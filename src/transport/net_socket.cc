/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>
#include <chrono>
#include <sstream>
#include <thread>

u_long getthreadid() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  u_long tid = std::stoul(ss.str());
  return tid;
}


double us_now() {
  auto t = std::chrono::high_resolution_clock::now();
  return t.time_since_epoch().count() / 1e3;  // convert to us
};


/* Init functions */
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      ncclNetIfs = findInterfaces(ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, ncclNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&ncclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

ncclResult_t ncclSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

ncclResult_t ncclSocketPciPath(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/net/%s/device", ncclNetIfNames+dev*MAX_IF_NAME_SIZE);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    INFO(NCCL_NET|NCCL_INIT, "Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS 128
// #define MAX_QUEUE_LEN MAX_REQUESTS
#define MAX_QUEUE_LEN 256
#define MIN_CHUNKSIZE (64*1024)
#define MAX_NQ 8
#define Q_SHARE 4 // for each 4 threads share one tasks queue
#define TASK_SIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

struct ncclSocketHandle {
  union socketAddress connectAddr;
  int nSocks;
  int nThreads;
};

struct ncclSocketTask {
  int reqIdx; // index of the request, which the task belongs to 
  int posIdx;
  int op;
  void* data;
  int size;
  // int fd;
  int offset;
  int used;
  ncclResult_t result;
};

struct ncclSocketRequest {
  int posIdx; // position of the request in the comm->requests[]
  int op;
  void* data;
  int size;
  int ctrlFd;
  int used;
  struct ncclSocketComm* comm;
  struct ncclSocketTask* tasks[MAX_QUEUE_LEN];
  int nSubs;
};

struct ncclSocketTaskQueue {
  int head;
  int next;
  struct ncclSocketTask* tasks;
  pthread_mutex_t qLock;
  pthread_cond_t qCond;
};

enum threadState {start, stop};

struct ncclSocketThreadResources {
  struct ncclSocketTaskQueue* sharedTaskQueue;
  enum threadState state;
  struct ncclSocketComm* comm;
  int fds[MAX_SOCKETS];
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
};

struct ncclSocketListenComm {
  int fd;
  int nSocks;
  int nThreads;
};

struct ncclSocketComm {
  int ctrlFd;
  int fds[MAX_SOCKETS];
  int nSocks;
  int nThreads;
  // int nextFd;
  int nTaskQ;
  int nextTaskQ; // put task in round robin way
  struct ncclSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct ncclSocketThreadResources threadResources[MAX_THREADS];
  struct ncclSocketTaskQueue tasksQueues[MAX_NQ];
};

void* persistentSendThread(void *args_) {
  INFO(NCCL_INIT|NCCL_NET, "entering send thread");
  // u_long tid = (int unsigned long)pthread_self();
  struct ncclSocketThreadResources* resource = (struct ncclSocketThreadResources*)args_;
  struct ncclSocketComm* comm = resource->comm;
  INFO(NCCL_INIT|NCCL_NET, "resource ptr %p, comm ptr %p", resource, comm);
  volatile enum threadState* state = &resource->state;
  struct ncclSocketTaskQueue* taskQueue = resource->sharedTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  int tasks4Fds[MAX_SOCKETS];
  int sentInfo[MAX_SOCKETS];
  for (int i = 0; i < MAX_SOCKETS; i++) {
    tasks4Fds[i] = -1; sentInfo[i] = 0;
  }
  int infoBuf[2]; // request idx, task idx
  int* myFds = resource->fds;
  INFO(NCCL_INIT|NCCL_NET, "starting persistentSendThread with task queue ptr %p", taskQueue);
  while (1) {
    int idle = 1;
    int mark = taskQueue->next; // mark newest task seen
    // int _op = 0;
    // assign tasks to fds if there is elements 
    if (taskQueue->head != taskQueue->next) {
      pthread_mutex_lock(&taskQueue->qLock);
      for (int i = 0; i < nSocksPerThread; i++) {
        if (tasks4Fds[i] == -1 && taskQueue->head != taskQueue->next && 
            (taskQueue->tasks + taskQueue->head)->used == 1) {
          // sock i does not have task && queue is not empty
          tasks4Fds[i] = taskQueue->head;
          taskQueue->head = (taskQueue->head + 1) % MAX_QUEUE_LEN;
        }
      }
      pthread_mutex_unlock(&taskQueue->qLock);
    }
    // send the info of the task if not yet
    for (int i = 0; i < nSocksPerThread; i++) {
      if (tasks4Fds[i] > 0 && sentInfo[i] != 1 && myFds[i] > 0) {
        struct ncclSocketTask* t = taskQueue->tasks + tasks4Fds[i];
        infoBuf[0] = t->reqIdx; infoBuf[1] = t->posIdx;
        socketSend(myFds[i], (void*)infoBuf, 2 * sizeof(int));
        sentInfo[i] = 1;
      }
    }
    // send data
    for (int i = 0; i < nSocksPerThread; i++) {
      if (tasks4Fds[i] > 0) {
        // has a task to do
        struct ncclSocketTask* t = taskQueue->tasks + tasks4Fds[i];
        if (t != NULL && t->used == 1 && t->offset < t->size) {
          t->result =
              socketProgress(t->op, myFds[i], t->data, t->size, &t->offset);
          if (t->result != ncclSuccess) {
            WARN("NET/Socket : socket progress error");
            return NULL;
          }
        }
        idle = 0;

        if (t->offset == t->size) {
          // task done
          tasks4Fds[i] = -1;
          sentInfo[i] = 0;
        }
      }
    }
    if (idle) {
      // pthread_mutex_lock(&resource->threadLock);
      pthread_mutex_lock(&taskQueue->qLock);
      while (mark == taskQueue->next && *state != stop) { // no new tasks, wait
        pthread_cond_wait(&taskQueue->qCond, &taskQueue->qLock);
      }
      pthread_mutex_unlock(&taskQueue->qLock);
    }
    if (*state == stop) return NULL;
  }
}

void* persistentRecvThread(void* args_) {
  INFO(NCCL_INIT|NCCL_NET, "entering recv thread");
  struct ncclSocketThreadResources* resource = (struct ncclSocketThreadResources*)args_;
  struct ncclSocketComm* comm = resource->comm;
  volatile enum threadState* state = &resource->state;
  struct ncclSocketTaskQueue* taskQueue = resource->sharedTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  int tasks4Fds[MAX_SOCKETS][2]; // record pairs of req-idx and task-idx
  // init to all -1
  for (int i = 0; i < MAX_SOCKETS; i++) {
    tasks4Fds[i][0] = -1; tasks4Fds[i][1] = -1;
  }
  int* myFds = resource->fds;
  int infoBuf[2] = {-1, -1}; // for receiving
  int infoSize = 2*sizeof(int);
  // return NULL;
  INFO(NCCL_INIT|NCCL_NET, "starting persistentRecvThread with task queue ptr %p", taskQueue);
  while (1) {
    int idle = 1;
    int mark = taskQueue->next; // mark newest task seen

    // recv task info, in asyn way
    for (int i = 0; i < nSocksPerThread; i++) {
      if (tasks4Fds[i][0] == -1) {
        // no task assign to myFds[i]
        // try to receive
        int offset = 0;
        int result = socketProgress(NCCL_SOCKET_RECV, myFds[i], infoBuf, infoSize, &offset);
        if (result != ncclSuccess) {
          WARN("NET/Socket : socket progress error");
          return NULL;
        }
        // if already receive some bytes, then continue receive
        if (offset > 0 && offset < infoSize) {
          while(offset < infoSize) {
            socketProgress(NCCL_SOCKET_RECV, myFds[i], infoBuf, infoSize, &offset);
          }
        }
        // if received the task TODO improve later
        if (offset > 0) {
          tasks4Fds[i][0] = infoBuf[0];
          tasks4Fds[i][1] = infoBuf[1];
          infoBuf[0] = -1; infoBuf[1] = -1;
        }
        idle = 0;
      }
    }

    // recv task data
    for (int i = 0; i < nSocksPerThread; i++) {
      if (tasks4Fds[i][0] > -1) {
        ncclSocketTask* t = comm->requests[tasks4Fds[i][0]].tasks[tasks4Fds[i][1]];
        if (t != NULL && t->used == 1 && t->offset < t->size) {
          t->result =
              socketProgress(t->op, myFds[i], t->data, t->size, &t->offset);
          if (t->result != ncclSuccess) {
            WARN("NET/Socket : socket progress error");
            return NULL;
          }
        }
        idle = 0;
        if (t->offset == t->size) {
          // task done, clear the flags in tasks4Fds
          tasks4Fds[i][0] = -1; tasks4Fds[i][1] = -1;
        }
      }
    }

    // check status for idle
    if (idle){
      pthread_mutex_lock(&taskQueue->qLock);
      while (mark == taskQueue->next && *state != stop) { // no new tasks, wait
        pthread_cond_wait(&taskQueue->qCond, &taskQueue->qLock);
      }
      pthread_mutex_unlock(&taskQueue->qLock);
    }

    if (*state == stop) {
      return NULL;
    }
  }
}

/* 
void* persistentSocketThread(void *args_) {
  struct ncclSocketThreadResources* resource = (struct ncclSocketThreadResources*)args_;
  struct ncclSocketComm* comm = resource->comm;
  volatile enum threadState* state = &resource->state;
  struct ncclSocketTaskQueue* myQueue = resource->sharedTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  u_long tid = (int unsigned long)pthread_self();
  double startTime;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    // int _op = 0;
    for (int i=0; i<MAX_QUEUE_LEN; i+=nSocksPerThread) {
      int repeat;
      
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclSocketTask* r = myQueue->tasks+i+j;
          startTime = us_now();
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = socketProgress(r->op, r->fd, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;
            printf("{\"pid\":0, \"tid\": %lu, \"name\":\"op-%d-s-%d\", \"ph\":\"X\", \"ts\":%f, \"dur\": %f},\n",
                    tid, r->op, r->size, startTime, us_now() - startTime);
          }
          
        }
      } while (repeat);
      
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next && *state != stop) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (*state == stop) return NULL;
  }
}
*/

ncclResult_t ncclSocketGetNsockNthread(int dev, int* ns, int* nt) {
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=1, autoNs=1;
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetIfNames+dev*MAX_IF_NAME_SIZE);
    char* rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s\n", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    }
end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
  return ncclSuccess;
}

ncclResult_t ncclSocketNewListenComm(struct ncclSocketListenComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

ncclResult_t ncclSocketInitComm(struct ncclSocketComm* comm, bool isRecv) {
  // called after created a communicator and connected  
  int qSize = comm->nThreads / Q_SHARE;
  qSize = qSize ? qSize : 1;
  comm->nTaskQ = qSize;
  // memory allocation for store tasks
  for (int i = 0 ; i < qSize; ++i){
    comm->tasksQueues[i].next = 0;
    comm->tasksQueues[i].head = 0;
    NCCLCHECK(ncclCalloc(&comm->tasksQueues[i].tasks, MAX_QUEUE_LEN));
    pthread_mutex_init(&comm->tasksQueues[i].qLock, NULL);
    pthread_cond_init(&comm->tasksQueues[i].qCond, NULL);
  }
  // create helper threads, and assign task queue to helper threads
  for (int i = 0; i < comm->nThreads; ++i) {
    struct ncclSocketThreadResources* res = comm->threadResources+i;
    int qidx = i % qSize;
    res->sharedTaskQueue = &comm->tasksQueues[qidx];
    res->comm = comm;
    INFO(NCCL_INIT, "res->sharedTaskQueue %p", res->sharedTaskQueue);
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    if (isRecv) {
      pthread_create(comm->helperThread+i, NULL, persistentRecvThread, res);
    } else {
      pthread_create(comm->helperThread+i, NULL, persistentSendThread, res);
    }
    res->state = start;
    // assign fds to thd res
    int nSockPerThread = comm->nSocks / comm->nThreads;
    for (int j = 0; j < MAX_SOCKETS; j++) {
      // initialize fds
      res->fds[j] = -1; 
    }
    for (int j = 0; j < nSockPerThread; j++) {
      res->fds[j] = comm->fds[i * nSockPerThread + j];
    }
  }
  INFO(NCCL_INIT|NCCL_NET, "ncclSocketInitComm done, ctrl fd %d (%s), nQueue %d", comm->ctrlFd, isRecv? "recv-comm":"send-comm", qSize);
  return ncclSuccess;
}

ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->ctrlFd = -1;
  for (int i=0; i < MAX_SOCKETS; i++) {
    (*comm)->fds[i] = -1;
  }
  // (*comm)->nextFd = 0;
  (*comm)->nextTaskQ = 0;
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  struct ncclSocketListenComm* comm;
  NCCLCHECK(ncclSocketNewListenComm(&comm));
  NCCLCHECK(GetSocketAddr(dev, &handle->connectAddr));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  NCCLCHECK(ncclSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  for (int i=0; i<comm->nSocks+1; i++) {
    int tmpFd, offset=0;
    NCCLCHECK(connectAddress(&tmpFd, &handle->connectAddr));
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, tmpFd, &i, sizeof(int), &offset));
    if (i == comm->nSocks) comm->ctrlFd = tmpFd;
    else comm->fds[i] = tmpFd;
  }
  *sendComm = comm;
  ncclSocketInitComm(comm, false);
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketListenComm* lComm = (struct ncclSocketListenComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  for (int i=0; i<rComm->nSocks+1; i++) {
    int tmpFd, sendSockIdx, offset=0;
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", tmpFd);
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, tmpFd, &sendSockIdx, sizeof(int), &offset));
    if (sendSockIdx == rComm->nSocks) rComm->ctrlFd = tmpFd;
    else rComm->fds[sendSockIdx] = tmpFd;
  }
  *recvComm = rComm;
  ncclSocketInitComm(rComm, true);
  return ncclSuccess;
}


ncclResult_t ncclSocketGetRequest(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketRequest** req) {
  u_long tid = getthreadid();
  double startTime = us_now();
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSocketRequest* r = comm->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlFd = comm->ctrlFd;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      r->posIdx = i;
      *req = r;
      // printf("ncclSocketGetRequest, assign to location %d", i);
      // printf("{\"pid\":0, \"tid\": %lu, \"name\":\"getReqAt-%d\", \"ph\":\"X\", \"ts\":%f, \"dur\": %f},\n",
      //   tid, i, startTime, us_now() - startTime);
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketGetTask(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketTask** req, int pidx, int selfPos) {
  int qidx = comm->nextTaskQ % comm->nTaskQ;
  // struct ncclSocketThreadResources* res = comm->threadResources+tid;
  struct ncclSocketTaskQueue* queue = &comm->tasksQueues[qidx];
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    WARN("NET/Socket : ncclSocketTaskQueue not initialized");
  }
  struct ncclSocketTask* t = queue->tasks+queue->next;
  if (t->used == 0) {
    t->op = op;
    t->data = data;
    t->size = size;
    // t->fd = comm->fds[comm->nextFd];
    t->offset = 0;
    t->result = ncclSuccess;
    comm->nextTaskQ = (comm->nextTaskQ + 1) % comm->nTaskQ;
    t->used = 1;
    t->reqIdx = pidx; // record the parent request position idx
    t->posIdx = selfPos; 
    *req = t;
    // pthread_mutex_lock(&res->threadLock);
    pthread_mutex_lock(&queue->qLock);
    queue->next = (queue->next+1) % MAX_QUEUE_LEN;
    pthread_cond_signal(&queue->qCond);
    // res->state = start;
    // pthread_cond_signal(&res->threadCond);
    // pthread_mutex_unlock(&res->threadLock);
    pthread_mutex_unlock(&queue->qLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      WARN("NET/Socket : message truncated : receiving %d bytes instead of %d", data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->used = 2; // done exchanging size
    // divide into subtasks
    // int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
    // fixed size for tasks
    int taskSize = TASK_SIZE; 
    int chunkOffset = 0, i = 0;
    while (chunkOffset < r->size) {
      int chunkSize = std::min(taskSize, r->size-chunkOffset);
      NCCLCHECK(ncclSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i, r->posIdx, i - 1));
      i++;
      chunkOffset += chunkSize;
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    int nCompleted = 0;
    for (int i=0; i<r->nSubs; i++) {
      struct ncclSocketTask* sub = r->tasks[i];
      if (sub->result != ncclSuccess) return sub->result;
      if (sub->offset == sub->size) nCompleted++;
    }
    if (nCompleted == r->nSubs) {
      if (size) *size = r->size;
      *done = 1;
      r->used = 0;
      for (int i=0; i<r->nSubs; i++) {
        struct ncclSocketTask* sub = r->tasks[i];
        sub->used = 0;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_RECV, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketFlush(void* recvComm, void* data, int size, void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketCloseListen(void* opaqueComm) {
  struct ncclSocketListenComm* comm = (struct ncclSocketListenComm*)opaqueComm;
  if (comm) {
    if (comm->fd != -1) close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->state = stop;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      // free(res->sharedTaskQueue.tasks);
    }
    // carefully address the free operation
    for (int i = 0; i < comm->nTaskQ; i++) {
      free(&comm->tasksQueues[i]);
    }
    if (comm->ctrlFd != -1) close(comm->ctrlFd);
    for (int i=0; i<comm->nSocks; i++) {
      if (comm->fds[i] != -1) close(comm->fds[i]);
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketPciPath,
  ncclSocketPtrSupport,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketRegMr,
  ncclSocketDeregMr,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketFlush,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketCloseListen
};
