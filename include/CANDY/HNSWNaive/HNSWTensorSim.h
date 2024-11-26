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
  int64_t numElements_; ///< Maximum number of elements
  int64_t maxDegree_; ///< Maximum degree for a node
  int64_t efConstruction_; ///< Parameter for controlling construction phase
  float levelMultiplier_; ///< Multiplier for determining levels
  int64_t currentNodeCount_; ///< Current number of nodes
  int64_t maxLevel_; ///< Maximum level in the hierarchy
  std::vector<torch::Tensor> vectors_; ///< Stored vectors
  torch::Tensor similarityTensor_; ///< Adjacency list as a 2D tensor
  std::vector<torch::Tensor> layerVectors_; ///< Vector of tensors, each containing the nodes in a layer

  /// Generate a random level for a node
  int64_t randomLevel();

  /// Add a node to the graph
  void addNode(int64_t id, int64_t level);

  /// Search within a specific layer
  std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> searchLayer(
      const torch::Tensor& query, int64_t entryPointId, int64_t ef, int64_t layer);

  /// Shrink connections to maintain the max degree
  std::vector<int64_t> shrinkConnections(const std::vector<std::pair<float, int64_t>>& candidates);

  /// Get neighbors for a node
  std::vector<int64_t> getNeighbors(int64_t id);

  void add_link(int64_t src, int64_t dest, int64_t level);
};

}
#endif //CANDYBENCH_INCLUDE_CANDY_HNSWNAIVE_HNSWTENSORSIME_H_
