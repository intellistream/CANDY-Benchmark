#include <CANDY/LSH/VectorGenerator.h>

namespace CANDY {

torch::Tensor NormalizationVectorGenerator::generate(int vecDim) {
    std::vector<double> new_vector(vecDim);
    torch::Tensor new_tensor = torch::zeros({1, vecDim});
    std::generate(new_vector.begin(), new_vector.end(), [this] { return dis(this->gen); });
 
    double norm = std::sqrt(std::inner_product(new_vector.begin(), new_vector.end(), new_vector.begin(), 0.0));
    for (size_t i = 0; i < vecDim; i++) {
        new_tensor[0][i] = new_vector[i] / norm;
    }

    return new_tensor;
}

} // namespace CANDY