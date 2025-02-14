#ifndef UTILS_COMPUTE_GT_STEPWISE_RECALL_HPP
#define UTILS_COMPUTE_GT_STEPWISE_RECALL_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace COMPUTE_GT {

struct StepRecall {
  uint64_t step;        
  double recall;       
};

bool readBatchFromFile(const std::string& filename, uint64_t& step, uint64_t& npts, uint64_t& ndims,
                        std::vector<std::vector<float>>& queryVectors,
                        std::vector<std::vector<float>>& annsResults);

bool readGTFile(const std::string& filename, uint64_t& npts, uint64_t& ndims,
                  std::vector<std::vector<float>>& gtVectors);

double computeRecallWithQueryVec(const std::vector<std::vector<float>>& queryVectors,
                                    const std::vector<std::vector<float>>& annsResult,
                                    const std::vector<std::vector<float>>& gtVectors);

void calcStepwiseRecall(const std::string& predFile, const std::string& gtFile, 
                          const std::string& outputFile);
} 

#endif // UTILS_COMPUTE_GT_STEPWISE_RECALL_HPP
