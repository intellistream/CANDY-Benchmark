//
// Created by tony on 24-11-28.
//
#include <CANDY/Grid2DIndex/MLPGravastarModel.h>
#include <cmath>
#include <Utils/IntelliTensorOP.hpp>
#include <c10/util/Logging.h>
namespace CANDY {
torch::Tensor MLPGravastarModel::customLoss(torch::Tensor yOut,torch::Tensor yExpect){
  auto originalIp = yExpect;
  auto compIp = yOut;
  auto normalizedOri = INTELLI::IntelliTensorOP::l2Normalize(originalIp);
  auto normalizedSurface = INTELLI::IntelliTensorOP::l2Normalize(compIp);
  return torch::mse_loss(normalizedSurface, normalizedOri);
}
void  MLPGravastarModel::init(int64_t inputDim, int64_t outputDim, INTELLI::ConfigMapPtr extraConfig) {
  if (extraConfig != nullptr) {
    cudaDev = extraConfig->tryI64("cudaDev", -1, true);
    cudaDevInference =  extraConfig->tryI64("cudaDevInference", -1, true);
    MLTrainBatchSize = extraConfig->tryI64("MLTrainBatchSize", 128, true);
    learningRate = extraConfig->tryDouble("learningRate", 0.01, true);
    hiddenLayerDim = extraConfig->tryI64("hiddenLayerDim", outputDim, true);
    MLTrainEpochs = extraConfig->tryI64("MLTrainEpochs", 30, true);
  }
  model.init(inputDim, hiddenLayerDim, outputDim);
}
void MLPGravastarModel::trainModel(torch::Tensor &x1, int64_t logout) {
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learningRate)); // The learning rate is set to 0.001
  torch::Device device(torch::kCPU); // or torch::kCPU
  torch::Device deviceInference(torch::kCPU); // or torch::kCPU
  if (cudaDev > -1 && torch::cuda::is_available()) {
    // Move tensors to GPU 1
    device = torch::Device(torch::kCUDA, cudaDev);
  }
  if (cudaDevInference > -1 && torch::cuda::is_available()) {
    // Move tensors to GPU 1
    deviceInference = torch::Device(torch::kCUDA, cudaDevInference);
  }
  int64_t num_micro_batches, micro_batch_size;
  if (MLTrainBatchSize > 1) {
    num_micro_batches = x1.size(0) / MLTrainBatchSize;
    micro_batch_size = MLTrainBatchSize;
  } else {
    num_micro_batches = 1;
    micro_batch_size = x1.size(0);
  }
  int64_t epochDiv = MLTrainEpochs / 10;
  int64_t batchDiv = num_micro_batches / 10;
  int64_t maxIdx = x1.size(0);
  // Example training loop
  model.to(device);
  torch::manual_seed(999);
  for (int64_t epoch = 0; epoch != MLTrainEpochs; ++epoch) {
    model.train();
    torch::Tensor loss;
    double total_loss = 0.0;
    for (int64_t mb = 0; mb < num_micro_batches; ++mb) {
      // Calculate start and end indices for the current micro-batch
      int64_t start_idx = mb * micro_batch_size;
      int64_t end_idx = start_idx + micro_batch_size;
      if (end_idx > maxIdx) {
        end_idx = maxIdx;
      }
      // Slice the tensors to get the current micro-batch
      auto mb_x1 = x1.slice(/*dim=*/0, start_idx, end_idx).to(device);
      // Forward pass for the current micro-batch
      auto output1 = model.forward(mb_x1);
      auto originalIp = torch::matmul(mb_x1,mb_x1.t());
      auto gravastarSurface = torch::matmul(output1,output1.t());
      // Compute loss for the current micro-batch
      auto loss = customLoss(gravastarSurface,originalIp);
      // Backward pass
      optimizer.zero_grad(); // Zero gradients (do this once per batch, not per micro-batch)
      loss.backward(); // Accumulate gradients over all micro-batches
      total_loss += loss.item<double>();
      if (mb % batchDiv == 1&&logout) {
        LOG(INFO) << "Epoch [" << (epoch + 1) << "/" << MLTrainEpochs << "], Done batch " << mb<<"/"<<num_micro_batches
                  << std::endl;
      }
    }
    if (epoch % epochDiv == 1&&logout) {
      LOG(INFO) << "Epoch [" << (epoch + 1) << "/" << MLTrainEpochs << "], Loss: " << (total_loss / num_micro_batches)
                << std::endl;
    }
    // Optimization step (after processing all micro-batches)
    optimizer.step();
  }
  model.to(deviceInference);
}

torch::Tensor MLPGravastarModel::forward(torch::Tensor &input) {
  return model.forward((input));
}

}