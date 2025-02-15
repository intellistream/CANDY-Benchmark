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

bool readStepwiseFile(const std::string& filename, uint64_t& npts, uint64_t& ndims, 
                        std::vector<size_t>* indices, std::vector<std::vector<float>>& data, 
                        bool readIndices);

double computeRecallWithQueryVec(const std::vector<std::vector<float>>& queryVectors,
                                    const std::vector<std::vector<float>>& annsResult,
                                    const std::vector<std::vector<float>>& gtVectors);

void calcStepwiseRecall(const std::string& annsFile, const std::string& gtFile, 
                          const std::string& outputFile);
} 

#endif // UTILS_COMPUTE_GT_STEPWISE_RECALL_HPP
