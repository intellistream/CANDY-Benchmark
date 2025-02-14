#include <Utils/ComputeGT/StepwiseRecall.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>

bool COMPUTE_GT::readBatchFromFile(const std::string& filename, uint64_t& step, uint64_t& npts, uint64_t& ndims,
                                    std::vector<std::vector<float>>& predictedGT) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
      std::cerr << "Error opening prediction file: " << filename << std::endl;
      return false;
  }

  while (file.read(reinterpret_cast<char*>(&step), sizeof(uint64_t))) {
      file.read(reinterpret_cast<char*>(&npts), sizeof(uint64_t));
      file.read(reinterpret_cast<char*>(&ndims), sizeof(uint64_t));

      // 跳过查询向量部分
      file.seekg(npts * ndims * sizeof(float), std::ios::cur);

      
      predictedGT.resize(npts, std::vector<float>(ndims));
      for (size_t i = 0; i < npts; i++) {
          file.read(reinterpret_cast<char*>(predictedGT[i].data()), ndims * sizeof(float));
      }
  }
  return true;
}