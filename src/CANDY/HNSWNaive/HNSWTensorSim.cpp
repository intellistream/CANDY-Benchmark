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
  maxLevel_ = 0;
}

/**
 * @brief Generate a random level for a node.
 */
int64_t HNSWTensorSim::randomLevel() {
  static std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  int64_t level = 0;
  while (distribution(generator) < levelMultiplier_) {
    level++;
  }
  return level;
}

void HNSWTensorSim::add_link(int64_t src, int64_t dest, int64_t level) {
    for(int64_t j=0; j<maxDegree_; j++){
        if(similarityTensor_[src][j].item<int64_t>()==-1){
            break; // in case of deletion
        }
        if(similarityTensor_[src][j].item<int64_t>()==dest){
            return; // check repetition
        }
    }

    if(similarityTensor_[src][maxDegree_-1].item<int64_t>()==-1){
        int64_t i=maxDegree_;
        while(i>0){
            if(similarityTensor_[src][i-1].item<int64_t>()!=-1){
                break;
            }
            i--;
        }
        // if there is empty slot
        similarityTensor_[src][i]=dest;
        return;
    }

    // need to prune some neighbors
    auto innerProduct = [](const torch::Tensor& a, const torch::Tensor& b) {
        return torch::dot(a, b).item<float>();
    };
    // maintain candidates as a maxheap
    std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> topCandidates;
    topCandidates.emplace(innerProduct(vectors_[src], vectors_[dest]), dest);
    for(int64_t j=0; j<maxDegree_; j++){
        if(similarityTensor_[src][j].item<int64_t>()==-1){
            continue;
        }
        auto nei = similarityTensor_[src][j].item<int64_t>();
        topCandidates.emplace(innerProduct(vectors_[src], vectors_[nei]), nei);
    }

    // prune
    std::vector<int64_t>  tempList;
    while(!topCandidates.empty()){
        auto v1 = topCandidates.top().second;
        auto dist_v1_q = topCandidates.top().first;
        topCandidates.pop();
        bool rng_ok = true;

        for(auto v2 : tempList){
            auto dist_v1_v2 = innerProduct(vectors_[v1], vectors_[v2]);
            if(dist_v1_v2 < dist_v1_q){
                // v1 can be visited from q->v2->v1 with little effort
                rng_ok = false;
                break;
            }
        }
        if(rng_ok){
            tempList.push_back(v1);
            if(tempList.size()>=maxDegree_){
                return;
            }
        }
    }







    int64_t i=0;
    for(i=0; i<tempList.size(); i++){
        similarityTensor_[src][i] = tempList[i];
    }
    while(i<maxDegree_){
        similarityTensor_[src][i]=-1;
        i++;
    }
    return;





}

/**
 * @brief Add a vector to the HNSW graph.
 */
void HNSWTensorSim::add(const torch::Tensor& vector) {
  if (currentNodeCount_ >= numElements_) {
    throw std::runtime_error("Maximum capacity reached.");
  }

  int64_t id = currentNodeCount_++;
  vectors_.push_back(vector);

  int64_t nodeLevel = randomLevel();
  if (nodeLevel > maxLevel_) {
    maxLevel_ = nodeLevel;
    layerVectors_.resize(maxLevel_ + 1);
  }

  if (layerVectors_.size() <= nodeLevel) {
    layerVectors_.resize(nodeLevel + 1);
  }

  if (layerVectors_[nodeLevel].numel() == 0) {
    layerVectors_[nodeLevel] = torch::tensor({id}, torch::kInt64);
  } else {
    layerVectors_[nodeLevel] = torch::cat({layerVectors_[nodeLevel], torch::tensor({id}, torch::kInt64)});
  }

  if (id == 0) {
    return;
  }

  int64_t entryPointId = 0;
  for (int64_t level = maxLevel_; level > nodeLevel; --level) {
    auto candidates = searchLayer(vector, entryPointId, 1, level);
    if (!candidates.empty()) {
      entryPointId = candidates.top().second;
    }
  }

  for (int64_t level = std::min(maxLevel_, nodeLevel); level >= 0; --level) {
    auto candidates = searchLayer(vector, entryPointId, efConstruction_, level);

    std::vector<std::pair<float, int64_t>> candidateList;
    while (!candidates.empty()) {
      candidateList.push_back(candidates.top());
      candidates.pop();
    }

    auto connections = shrinkConnections(candidateList);

    for (size_t i = 0; i < connections.size(); ++i) {
      //similarityTensor_[id][(int64_t)i] = connections[i];
      add_link(id, connections[i], level);
    }

    for (int64_t neighbor : connections) {
        add_link(neighbor, id, level);
//      for (int64_t j = 0; j < maxDegree_; ++j) {
//        if (similarityTensor_[neighbor][j].item<int64_t>() == -1) {
//          similarityTensor_[neighbor][j] = id;
//          break;
//        }
//      }
    }
  }
}

/**
 * @brief Perform a single-query search.
 */
std::vector<int64_t> HNSWTensorSim::search(const torch::Tensor& query, int64_t k) {
  if (currentNodeCount_ == 0) {
    return {};
  }

  int64_t entryPointId = 0;
  for (int64_t level = maxLevel_; level > 0; --level) {
    auto candidates = searchLayer(query, entryPointId, 1, level);
    if (!candidates.empty()) {
      entryPointId = candidates.top().second;
    }
  }

  auto candidates = searchLayer(query, entryPointId, k, 0);
  std::vector<int64_t> result;
  while (!candidates.empty()) {
    result.push_back(candidates.top().second);
    candidates.pop();
  }

  return result;
}

/**
 * @brief Perform a multi-query search.
 */
std::vector<torch::Tensor> HNSWTensorSim::multiQuerySearch(const torch::Tensor& queries, int64_t k) {
  int64_t n = queries.size(0);
  std::vector<torch::Tensor> results;
  for (int64_t i = 0; i < n; ++i) {
    auto resultIds = search(queries[i], k);
    std::vector<torch::Tensor> neighbors;
    for (auto id : resultIds) {
      neighbors.push_back(vectors_[id]);
    }
    results.push_back(torch::stack(neighbors));
  }
  return results;
}

/**
 * @brief Search within a specific layer.
 */std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> HNSWTensorSim::searchLayer(
    const torch::Tensor& query, int64_t entryPointId, int64_t ef, int64_t layer) {
  std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> topCandidates;
  std::unordered_set<int64_t> visited;

  auto innerProduct = [](const torch::Tensor& a, const torch::Tensor& b) {
    return torch::dot(a, b).item<float>();
  };

  std::priority_queue<std::pair<float, int64_t>> candidates;
  candidates.emplace(innerProduct(query, vectors_[entryPointId]), entryPointId);
  visited.insert(entryPointId);

  while (!candidates.empty()) {
    auto current = candidates.top();
    candidates.pop();

    topCandidates.emplace(current);
    if (topCandidates.size() > ef) {
      topCandidates.pop();
    }

    // Get nodes in the current layer
    const auto& layerNodes = layerVectors_[layer];
   // auto layerNodesAccessor = layerNodes.accessor<int64_t, 1>();

    for (int64_t i = 0; i < layerNodes.size(0); ++i) {
      int64_t neighbor = layerNodes[i].item<int64_t>();
      if (visited.count(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      float similarity = innerProduct(query, vectors_[neighbor]);
      candidates.emplace(similarity, neighbor);
    }
  }

  return topCandidates;
}
/**
 * @brief Shrink connections to maintain the max degree.
 */
std::vector<int64_t> HNSWTensorSim::shrinkConnections(const std::vector<std::pair<float, int64_t>>& candidates) {
  std::vector<int64_t> connections;
  for (const auto& candidate : candidates) {
    if (connections.size() >= maxDegree_) {
      break;
    }
    connections.push_back(candidate.second);
  }
  return connections;
}

/**
 * @brief Retrieve the similarity tensor for inspection.
 */
torch::Tensor HNSWTensorSim::getSimilarityTensor() const {
  return similarityTensor_;
}

} // CANDY