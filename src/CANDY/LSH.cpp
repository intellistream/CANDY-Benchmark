#include <CANDY/LSH.h>

using namespace CANDY;
using namespace std;

LSH::LSH() {}

LSH::~LSH() {}

vector<VectorIndex> LSH::searchIndex(const torch::Tensor &q, int64_t k) {
    auto query_num = q.size(0);

    vector<VectorIndex> result(query_num * k, -1);

    for (size_t slice_idx = 0; slice_idx < query_num; slice_idx++) {
        priority_queue<pair<double, VectorIndex>, vector<pair<double, VectorIndex>>, std::greater<pair<double, int>>> resultPriorityQueue;
        unordered_set<VectorIndex> resultSet;

        for (size_t i = 0; i < hashFunctionNum; i++) {
            auto hash_value = hashFunctions[i](q.slice(0, slice_idx, slice_idx + 1));
            auto range = hashTables[i].equal_range(hash_value);
            for (auto it = range.first; it != range.second; it++) {
                resultSet.insert(it->second);
            }
        }

        for (auto index : resultSet) {
            resultPriorityQueue.push({torch::dist(q.slice(0, slice_idx, slice_idx + 1).reshape(-1), tensorDatabase[index]->reshape(-1)).item<double>(), index});
        }

        for (size_t i = 0; i < k; i++) {
            if (resultPriorityQueue.empty()) {
                break;
            }
            result[slice_idx * k + i] = resultPriorityQueue.top().second;
            resultPriorityQueue.pop();
        }
    }

    return result;
}

bool LSH::setConfig(INTELLI::ConfigMapPtr cfg) {
    AbstractIndex::setConfig(cfg);
    hashFunctionNum = cfg->tryI64("hashFunctionNum", 10, true);
    vecDim = cfg->tryI64("vecDim", 768, true);
    hashFunctionNum = max(int64_t(ceil(log2(vecDim))), hashFunctionNum);
    assert(vecDim >= 2 && "vecDim must be at least 2");

    hashFunctions.resize(hashFunctionNum);
    hashTables.resize(hashFunctionNum);
    randomVectors.resize(hashFunctionNum);

    for (size_t i = 0; i < hashFunctionNum; i++) {
        randomVectors[i] = vectorGenerator.generate(vecDim);
        hashFunctions[i] = [i, this](const torch::Tensor &t) {
            auto random_vector = this->randomVectors[i];
            auto projection = torch::dot(random_vector.reshape(-1), t.reshape(-1)).item<double>();

            auto hash_value = 10 * atan(projection);
            return HashValue(round(hash_value));
        };
    }
    
    INTELLI_INFO("LSH setConfig success");

    return true;
}

bool LSH::insertTensor(const torch::Tensor &t) {
    // Get the number of tensors
    auto num = t.size(0);

    for (size_t i = 0; i < num; i++) {
        // insert tensor into database
        auto new_tensor = newTensor(t.slice(0, i, i + 1));
        if (new_tensor == nullptr) {
            INTELLI_ERROR("LSH insertTensor failed");
            return false;
        }

        VectorIndex index = tensorDatabase.size();

        if (deleteCache.size() > 0) {
            index = deleteCache.front();
            deleteCache.pop();
            tensorDatabase[index] = new_tensor;
        } else {
            tensorDatabase.push_back(new_tensor);
        }

        // Insert the tensor into hash tables
        for (size_t j = 0; j < hashFunctionNum; j++) {
            auto hash_value = hashFunctions[j](*new_tensor);
            hashTables[j].insert({hash_value, index});
        }
    }

    INTELLI_INFO("LSH insertTensor success");

    return true;
}

vector<INTELLI::TensorPtr> LSH::searchTensor(const torch::Tensor &q, int64_t k) {
    auto result = searchIndex(q, k);

    vector<INTELLI::TensorPtr> result_tensor;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] == -1) {
            result_tensor.push_back(nullptr);
        } else {
            result_tensor.push_back(tensorDatabase[result[i]]);
        }
    }

    INTELLI_INFO("LSH searchTensor success");

    return result_tensor;
}

bool LSH::deleteTensor(const torch::Tensor &t) {
    auto delete_index = searchIndex(t, 1);

    for (size_t i = 0; i < delete_index.size(); i++) {
        if (delete_index[i] != -1) {
            auto& tensorPtr = tensorDatabase[delete_index[i]];
            if (tensorPtr == nullptr) continue;

            if (tensorPtr->equal(t.slice(0, i, i + 1))) {
                // update deleteCache
                deleteCache.push(delete_index[i]);

                // updtate hashTables
                for (size_t j = 0; j < hashFunctionNum; j++) {
                    auto hash_value = hashFunctions[j](*tensorPtr);
                    auto range = hashTables[j].equal_range(hash_value);
                    for (auto it = range.first; it != range.second; it++) {
                        if (it->second == delete_index[i]) {
                            hashTables[j].erase(it);
                            break;
                        }
                    }
                }

                // release tensor in memory
                tensorPtr.reset();
                tensorDatabase[delete_index[i]] = nullptr;
            }
        }
    }

    INTELLI_INFO("LSH deleteTensor success");

    return true;
}