#ifndef CALC_RECALL_STEPWISE_HPP
#define CALC_RECALL_STEPWISE_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace COMPUTE_GT {

struct StepRecall {
  uint64_t step;        
  double recall;       
};

void calculateStepwiseRecall(const std::string& predFile,
                              const std::string& gtFile,
                              const std::string& outputFile);

bool readBatchFromFile(const std::string& filename, 
                        uint64_t& step, uint64_t& npts, uint64_t& ndims,
                        std::vector<std::vector<float>>& predictedGT);

bool readGTFile(const std::string& filename, 
                  uint64_t& npts, uint64_t& ndims,
                  std::vector<std::vector<float>>& gtVectors);

double computeRecall(const std::vector<std::vector<float>>& predictedGT,
                     const std::vector<std::vector<float>>& groundTruth);

} 

#endif // CALC_RECALL_STEPWISE_HPP
