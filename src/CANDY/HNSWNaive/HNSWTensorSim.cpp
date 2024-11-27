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
  layerTensor_ = torch::full({numElements_}, -1, torch::kInt64); // 1D Tensor for layer information
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

  layerTensor_[id] = nodeLevel; // Record the layer for this node
  if (nodeLevel > maxLevel_) {
    maxLevel_ = nodeLevel;
    entryPointId_ = id;
  }
  //std::cout<<"ID="<<id<<"level = "<<nodeLevel<<std::endl;
  if (id == 1) {
    similarityTensor_[0][0] = 1;
    similarityTensor_[1][0] = 0;
    return; // No need to connect the first node
  }

  int64_t entryPointId = entryPointId_;
  for (int64_t level = maxLevel_; level > nodeLevel; --level) {
    auto candidates = searchLayer(vector, entryPointId, 1, level);
    if (!candidates.empty()) {
      entryPointId = candidates.top().second;
      //std::cout<<"candidate size "<<candidates.size()<<std::endl;
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
      add_link(id,connections[i],0);
    }

    for (int64_t neighbor : connections) {
      add_link(neighbor,id,0);
      /*for (int64_t j = 0; j < maxDegree_; ++j) {
        if (similarityTensor_[neighbor][j].item<int64_t>() == -1) {
          similarityTensor_[neighbor][j] = id;
          break;
        }
      }*/
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

  int64_t entryPointId = entryPointId_;
  int64_t level;
  for (level = maxLevel_; level > 0; --level) {
    auto candidates = searchLayer(query, entryPointId, 1, level);
   // std::cout<<"Layer"+std::to_string(level)+",candidates="+std::to_string(candidates.size())<<std::endl;
    if (!candidates.empty()) {
      entryPointId = candidates.top().second;
    }
  }


  auto candidates = searchLayer(query, entryPointId, k, level);
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
   // std::cout<<resultIds.size()<<std::endl;
    /*for (auto id : resultIds) {
     std::cout<<vectors_[id]<<std::endl;
    }*/
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
 */
std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, std::greater<>> HNSWTensorSim::searchLayer(
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

    auto furthest = topCandidates.empty()
                    ? std::make_pair(std::numeric_limits<float>::lowest(), static_cast<int64_t>(-1))
                    : topCandidates.top();
    if (topCandidates.size() >= ef && current.first < furthest.first) {
      break;
    }

    auto neighbors = getNeighbors(current.second, layer);
    if(neighbors.size()!=0) {
    //  std::cout<<"Neighbor len="+std::to_string(neighbors.size());
    }

    for (int64_t neighbor : neighbors) {
      if (visited.count(neighbor)) {
        continue;
      }
      visited.insert(neighbor);

      float similarity = innerProduct(query, vectors_[neighbor]);
      candidates.emplace(similarity, neighbor);

      if (topCandidates.size() < ef || similarity > furthest.first) {
        topCandidates.emplace(similarity, neighbor);
        if (topCandidates.size() > ef) {
          topCandidates.pop();
        }
      }
    }
  }

  return topCandidates;
}

/**
 * @brief Get neighbors for a node in a specific layer.
 */
std::vector<int64_t> HNSWTensorSim::getNeighbors(int64_t id, int64_t layer) {
  std::vector<int64_t> neighbors;
 /* if (layerTensor_[id].item<int64_t>() != layer) {
    return neighbors; // Node does not belong to the specified layer
  }*/
  for (int64_t i = 0; i < maxDegree_; ++i) {
    int64_t neighbor = similarityTensor_[id][i].item<int64_t>();
    if (neighbor != -1) {
      //std::cout<<"expect layer "+std::to_string(layer)+",  get"+std::to_string(layerTensor_[neighbor].item<int64_t>() )<<std::endl;
      if(layerTensor_[neighbor].item<int64_t>() != layer){

      }
      else {
        neighbors.push_back(neighbor);
      }

    }

  }
  return neighbors;
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