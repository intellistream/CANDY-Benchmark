#include <Utils/ComputeGT/StepwiseRecall.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>

bool COMPUTE_GT::readBatchFromFile(const std::string& filename, uint64_t& step, uint64_t& npts, uint64_t& ndims,
                                    std::vector<std::vector<float>>& queryVectors,
                                    std::vector<std::vector<float>>& annsResults) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening prediction file: " << filename << std::endl;
    return false;
  }

  if (!file.read(reinterpret_cast<char*>(&step), sizeof(uint64_t))) return false;
  if (!file.read(reinterpret_cast<char*>(&npts), sizeof(uint64_t))) return false;
  if (!file.read(reinterpret_cast<char*>(&ndims), sizeof(uint64_t))) return false;

  queryVectors.resize(npts, std::vector<float>(ndims));
  for (size_t i = 0; i < npts; i++) {
    if (!file.read(reinterpret_cast<char*>(queryVectors[i].data()), ndims * sizeof(float))) return false;
  }

  annsResults.resize(npts, std::vector<float>(ndims));
  for (size_t i = 0; i < npts; i++) {
    if (!file.read(reinterpret_cast<char*>(annsResults[i].data()), ndims * sizeof(float))) return false;
  }

  return true;
}

bool COMPUTE_GT::readGTFile(const std::string& filename, uint64_t& npts, uint64_t& ndims,
                              std::vector<std::vector<float>>& gtVectors) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening ground truth file: " << filename << std::endl;
    return false;
  }

  if (!file.read(reinterpret_cast<char*>(&npts), sizeof(uint64_t))) return false;
  if (!file.read(reinterpret_cast<char*>(&ndims), sizeof(uint64_t))) return false;

  gtVectors.resize(npts, std::vector<float>(ndims));
  for (size_t i = 0; i < npts; i++) {
    if (!file.read(reinterpret_cast<char*>(gtVectors[i].data()), ndims * sizeof(float))) return false;
  }

  return true;
}

double COMPUTE_GT::computeRecallWithQueryVec(const std::vector<std::vector<float>>& queryVectors,
                                              const std::vector<std::vector<float>>& annsResult,
                                              const std::vector<std::vector<float>>& gtVectors) {
  if (queryVectors.empty() || annsResult.empty() || gtVectors.empty()) {
    std::cerr << "Error: Input vectors cannot be empty." << std::endl;
    return 0.0;
  }

  size_t nqueries = queryVectors.size();
  size_t ndims = annsResult[0].size();
  size_t correct_count = 0;

  std::map<std::vector<float>, size_t> gtIndexMap;
  for (size_t i = 0; i < gtVectors.size(); ++i) {
    gtIndexMap[gtVectors[i]] = i;
  }

  for (size_t i = 0; i < nqueries; i++) {
    auto it = gtIndexMap.find(queryVectors[i]);
    if (it == gtIndexMap.end()) {
      std::cerr << "Warning: Query vector not found in GT file." << std::endl;
      continue;
    }
    size_t gtIndex = it->second;

    std::set<int> gtSet;
    for (const auto& val : gtVectors[gtIndex]) {
      gtSet.insert(static_cast<int>(val));
    }

    for (float val : annsResult[i]) {
      int predictedId = static_cast<int>(val);
      if (gtSet.find(predictedId) != gtSet.end()) {
        correct_count++;
      }
    }
  }

  return static_cast<double>(correct_count) / (nqueries * ndims);
}

void COMPUTE_GT::calcStepwiseRecall(const std::string& predFile, const std::string& gtFile, const std::string& outputFile) {
  uint64_t npts, ndims;
  std::vector<std::vector<float>> gtVectors;
  if (!readGTFile(gtFile, npts, ndims, gtVectors)) {
    std::cerr << "Failed to read ground truth file." << std::endl;
    return;
  }

  std::ifstream predStream(predFile, std::ios::binary);
  if (!predStream) {
    std::cerr << "Failed to open prediction file." << std::endl;
    return;
  }

  std::ofstream outFile(outputFile);
  if (!outFile) {
    std::cerr << "Failed to open output file." << std::endl;
    return;
  }

  while (true) {
    uint64_t step, batchNpts, batchNdims;
    std::vector<std::vector<float>> queryVectors, annsResult;
    if (!readBatchFromFile(predFile, step, batchNpts, batchNdims, queryVectors, annsResult)) {
      break;
    }

    if (batchNdims != ndims) {
      std::cerr << "Dimension mismatch detected." << std::endl;
      break;
    }

    double recall = computeRecallWithQueryVec(queryVectors, annsResult, gtVectors);
    outFile << "Step " << step << ": Recall = " << recall << std::endl;
    std::cout << "Step " << step << ": Recall = " << recall << std::endl;
  }

  predStream.close();
  outFile.close();

  std::cout << "Stepwise recall written to " << outputFile << std::endl;
}