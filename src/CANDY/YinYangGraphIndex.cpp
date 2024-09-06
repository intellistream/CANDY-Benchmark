/*! \file YinYangGraphIndex.cpp*/
//
// Created by tony on 25/05/23.
//

#include <CANDY/YinYangGraphIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>

static inline uint32_t nnzBits(uint32_t a) {
  if (a == 0) {
    return 1;
  }
  return 32 - __builtin_clz(a);
}
#ifndef ONLINEIVF_NEXT_POW_2
/**
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define ONLINEIVF_NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

#ifndef HASH
#define HASH(X, MASK, SKIP) (((X) & MASK) >> SKIP)
#endif
void CANDY::YinYangGraphIndex::genCrsIndices() {
  int64_t n = vecDim;
  // Probability distribution
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform
}

bool CANDY::YinYangGraphIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  FlatIndex::setConfig(cfg);
  maxConnection = cfg->tryI64("maxConnection", 256, true);

  candidateTimes = cfg->tryI64("candidateTimes", 1, true);
  skeletonRows = cfg->tryI64("skeletonRows",-1, true);
  rowNNZTensor = torch::zeros({initialVolume, 1}, torch::kInt64);

  /**
   ** @brief generate the rotation matrix for random projection
   */
  torch::manual_seed(114514);
  return true;
}
bool CANDY::YinYangGraphIndex::insertTensor(torch::Tensor &t) {

  return true;
}

std::vector<torch::Tensor> CANDY::YinYangGraphIndex::searchTensor(torch::Tensor &q, int64_t k) {
  int64_t rows = q.size(0);
  std::vector<torch::Tensor> ru((size_t) rows);

  return ru;
}