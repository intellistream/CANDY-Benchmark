#ifndef CANDY_INCLUDE_CANDY_CONCURRENTINDEX_H_
#define CANDY_INCLUDE_CANDY_CONCURRENTINDEX_H_

#include <Utils/ConfigMap.hpp>
#include <memory>
#include <vector>
#include <tuple>
#include <Utils/IntelliTensorOP.hpp>
#include <CANDY/AbstractIndex.h>

namespace CANDY {

class ConcurrentIndex : public CANDY::AbstractIndex {

using BatchIndex = size_t;
using QueryIndex = size_t;

using SearchResults = std::vector<torch::Tensor>;
using SearchRecord = std::tuple<BatchIndex, QueryIndex, SearchResults>;

 protected:
  AbstractIndexPtr myIndexAlgo = nullptr;
  std::string myConfigString = "";

  int64_t vecDim = 0;
  double writeRatio = 0.0;
  int64_t numThreads = 1;

 public:
  ConcurrentIndex() {

  }

  ~ConcurrentIndex() {

  }

  virtual bool loadInitialTensor(torch::Tensor &t);

  virtual void reset();

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual std::vector<SearchRecord> ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &qt, int64_t k);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
};

typedef std::shared_ptr<class CANDY::CongestionDropIndex> CongestionDropIndexPtr;

#define newConcurrentIndex std::make_shared<CANDY::ConcurrentIndex>
}

#endif // CANDY_INCLUDE_CANDY_CONCURRENTINDEX_H_
