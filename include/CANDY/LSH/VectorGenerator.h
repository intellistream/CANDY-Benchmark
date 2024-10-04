#ifndef CANDY_VECTOR_GENERATOR_H
#define CANDY_VECTOR_GENERATOR_H

#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <torch/torch.h>

namespace CANDY {
class NormalizationVectorGenerator {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<> dis;

public:
    NormalizationVectorGenerator() : gen(rd()), dis(0.0, 1.0) {}
    ~NormalizationVectorGenerator() {}

    /**
     * @brief generate a random vector with a given dimension
     * @param vecDim the dimension of the vector
     * @return a random vector
     */
    torch::Tensor generate(int vecDim);
};
}

#endif