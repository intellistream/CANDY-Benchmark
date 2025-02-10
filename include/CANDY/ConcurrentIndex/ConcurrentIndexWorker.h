/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/10
 *  Description:
 */

#ifndef CANDY_INCLUDE_CANDY_CONCURRENTINDEXWORKER_H_
#define CANDY_INCLUDE_CANDY_CONCURRENTINDEXWORKER_H_

#include <Utils/AbstractC20Thread.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <Utils/IntelliTensorOP.hpp>
#include <CANDY/IndexTable.h>
#include <CANDY/AbstractIndex.h>

namespace CANDY {

class ConcurrentIndexWorker {
 protected:
  int64_t forceDrop = 1;
 public:
  TensorListIdxQueuePtr reduceQueue;
  ConcurrentIndexWorker() {

  }

  ~ConcurrentIndexWorker() {

  }

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual bool ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &q, int64_t k);
};

typedef std::shared_ptr<class CANDY::ConcurrentIndexWorker> ConcurrentIndexWorkerPtr;

#define newConcurrentIndexWorker std::make_shared<CANDY::ConcurrentIndexWorker>
}

#endif CANDY_INCLUDE_CANDY_CONCURRENTINDEXWORKER_H_
