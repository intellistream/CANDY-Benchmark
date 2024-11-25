//
// Created by tony on 24-11-25.
//

#include <CANDY/HNSWNaive/HNSWTensorSim.h>
#include <random>
#include <cmath>
#include <limits>

namespace CANDY {
void HNSWTensorSim::init(int64_t numElements, int64_t maxDegree, int64_t efConstruction, float levelMultiplier) {
  numElements_ = numElements;
  maxDegree_ = maxDegree;
  efConstruction_ = efConstruction;
  levelMultiplier_=levelMultiplier;
  currentNodeCount_=0;
  similarityTensor_ = torch::full({numElements_, maxDegree_}, -1, torch::kInt64);
}

/**
 * @brief Adds a new vector to the HNSW graph.
 */
void HNSWTensorSim::add(const torch::Tensor& vector) {
  if (currentNodeCount_ >= numElements_) {
    throw std::runtime_error("Exceeded the maximum number of elements.");
  }

  int64_t id = currentNodeCount_++;
  vectors_.push_back(vector.squeeze());

  if (currentNodeCount_ == 1) {
    return;
  }

  addNode(id);
}

/**
 * @brief Searches for the nearest neighbors of a single query vector.
 */
std::vector<int64_t> HNSWTensorSim::search(const torch::Tensor& query, int64_t k) {
  if (currentNodeCount_ == 0) {
    return {};
  }

  int64_t ef = std::max(efConstruction_, k);
  auto topCandidates = searchLayer(query, ef);

  std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::less<>> resultQueue;
  while (!topCandidates.empty()) {
    auto candidate = topCandidates.top();
    topCandidates.pop();
    resultQueue.emplace(candidate);
    if ((int64_t)resultQueue.size() > k) {
      resultQueue.pop();
    }
  }

  std::vector<int64_t> result;
  while (!resultQueue.empty()) {
    result.push_back(resultQueue.top().second);
    resultQueue.pop();
  }

  std::reverse(result.begin(), result.end());
  return result;
}

/**
 * @brief Performs a batched search for multiple queries.
 */
std::vector<torch::Tensor> HNSWTensorSim::multiQuerySearch(const torch::Tensor& queries, int64_t k) {
  if (queries.size(0) == 0) {
    return {};
  }

  int64_t n = queries.size(0);
  std::vector<torch::Tensor> results;

  for (int64_t i = 0; i < n; ++i) {
    auto query = queries[i];
    auto neighborIds = search(query, k);

    std::vector<torch::Tensor> neighbors;
    for (int64_t id : neighborIds) {
      neighbors.push_back(vectors_[id]);
    }

    if (!neighbors.empty()) {
      results.push_back(torch::stack(neighbors));
    } else {
      results.push_back(torch::empty({0, queries.size(1)}, queries.options()));
    }
  }

  return results;
}

/**
 * @brief Retrieves the similarity tensor for inspection.
 */
torch::Tensor HNSWTensorSim::getSimilarityTensor() const {
  return similarityTensor_;
}

/**
 * @brief Generates a random level for a node based on the level multiplier.
 */
int64_t HNSWTensorSim::randomLevel() {
  static std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  int64_t level = 0;
  while (distribution(generator) < levelMultiplier_ && level < maxDegree_) {
    level++;
  }
  return level;
}

/**
 * @brief Adds a new node to the graph and connects it to its neighbors.
 */
void HNSWTensorSim::addNode(int64_t id) {
  auto queryVector = vectors_[id];
  auto candidates = searchLayer(queryVector, efConstruction_);

  std::vector<int64_t> neighbors;
  while (!candidates.empty() && (int64_t)neighbors.size() < maxDegree_) {
    neighbors.push_back(candidates.top().second);
    candidates.pop();
  }

  for (int64_t i = 0; i < (int64_t)neighbors.size(); ++i) {
    similarityTensor_[id][i] = neighbors[i];
    int64_t neighborId = neighbors[i];
    for (int64_t j = 0; j < maxDegree_; ++j) {
      if (similarityTensor_[neighborId][j].item<int64_t>() == -1) {
        similarityTensor_[neighborId][j] = id;
        break;
      }
    }
  }
}

/**
 * @brief Searches for neighbors of a query vector within a specific layer of the graph.
 */
std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::less<>> HNSWTensorSim::searchLayer(
    const torch::Tensor& query, int64_t ef) {
  std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::less<>> topCandidates;
  std::unordered_set<int64_t> visited;

  auto innerProduct = [](const torch::Tensor& a, const torch::Tensor& b) {
    return -torch::dot(a, b).item<float>();
  };

  std::priority_queue<std::pair<float, int64_t>> candidates;
  candidates.emplace(innerProduct(query, vectors_[0]), 0);
  visited.insert(0);

  while (!candidates.empty()) {
    auto current = candidates.top();
    candidates.pop();

    topCandidates.emplace(current);
    if ((int64_t)topCandidates.size() > ef) {
      topCandidates.pop();
    }

    for (int64_t i = 0; i < maxDegree_; ++i) {
      int64_t neighbor = similarityTensor_[current.second][i].item<int64_t>();
      if (neighbor == -1 || visited.count(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      float similarity = innerProduct(query, vectors_[neighbor]);
      candidates.emplace(similarity, neighbor);
    }
  }

  return topCandidates;
}
} // CANDY