/*! \file HNSWTensorSim.h*/
//
// Created by tony on 24-11-25.
//

#ifndef CANDYBENCH_INCLUDE_CANDY_HNSWNAIVE_HNSWTENSORSIME_H_
#define CANDYBENCH_INCLUDE_CANDY_HNSWNAIVE_HNSWTENSORSIME_H_
#include <torch/torch.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <stdint.h>
namespace  CANDY {
/**
 * @class HNSWTensorSim
 * @brief Implementation of the HNSW (Hierarchical Navigable Small World) graph for approximate nearest neighbor search, using a tensor-based similarity structure.
 */
class HNSWTensorSim {
 public:
  /**
   * @brief init function
   * @param numElements Maximum number of elements that can be added to the graph.
   * @param maxDegree Maximum number of neighbors for each node.
   * @param efConstruction Number of candidates to evaluate during construction.
   * @param levelMultiplier Probability multiplier for determining the random level of a node.
   */
  void init(int64_t numElements, int64_t maxDegree, int64_t efConstruction, float levelMultiplier);
  HNSWTensorSim() {}

  /**
    * @brief Adds a new vector to the HNSW graph.
    * @param vector The vector to add, represented as a 1xD tensor.
    */
  void add(const torch::Tensor& vector);

  /// Searches for the nearest neighbors of a single query vector
  std::vector<int64_t> search(const torch::Tensor& query, int64_t k);

  /// Performs a batched search for multiple queries
  std::vector<torch::Tensor> multiQuerySearch(const torch::Tensor& queries, int64_t k);

  /// Retrieves the similarity tensor for inspection
  torch::Tensor getSimilarityTensor() const;

 private:
  int64_t numElements_; ///< Maximum number of elements in the graph
  int64_t maxDegree_; ///< Maximum number of neighbors per node
  int64_t efConstruction_; ///< Number of candidates to evaluate during graph construction
  float levelMultiplier_; ///< Probability multiplier for determining node levels
  int64_t currentNodeCount_; ///< Current number of nodes in the graph
  std::vector<torch::Tensor> vectors_; ///< List of vectors in the graph
  torch::Tensor similarityTensor_; ///< 2D tensor representing the graph's adjacency list

  /// Generates a random level for a node
  int64_t randomLevel();

  /// Adds a new node to the graph and connects it to its neighbors
  void addNode(int64_t id);

  /// Searches for neighbors of a query vector within a specific layer of the graph
  std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::less<>> searchLayer(const torch::Tensor& query, int64_t ef);
};

}
#endif //CANDYBENCH_INCLUDE_CANDY_HNSWNAIVE_HNSWTENSORSIME_H_
