/*! \file MLPGravastarModel.h*/
//
// Created by tony on 24-11-28.
//

#ifndef CANDYBENCH_INCLUDE_CANDY_GRID2DINDEX_MLPGRAVASTARMODEL_H_
#define CANDYBENCH_INCLUDE_CANDY_GRID2DINDEX_MLPGRAVASTARMODEL_H_
#include <torch/torch.h>
#include <iostream>
#include <Utils/ConfigMap.hpp>
#include <memory>
namespace CANDY {
/**
 * @ingroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */
/***
 * @class MLPGravastarModel CANDY/Grid2Index/MLPGravastarModel
 * @breif The model to fit a gravastar's surface using MLP
 */
class MLPGravastarModel {
 private:
  struct MLModel : torch::nn::Module {
    torch::nn::Linear inputLayer{nullptr};
    torch::nn::Linear outputLayer{nullptr};
   // torch::Tensor middleTensor;
    MLModel() {}
    ~MLModel() {}
    // Constructor with input_dim, hidden_dim, and output_dim
    void init(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
      torch::manual_seed(999);
      // Initialize the input layer and register it
      inputLayer = register_module("inputLayer", torch::nn::Linear(input_dim, hidden_dim));
      // Initialize the output layer and register it
      outputLayer = register_module("outputLayer", torch::nn::Linear(hidden_dim, output_dim));
    }
    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
      // Pass the input through the input layer followed by a ReLU activation
      x = torch::tanh(inputLayer->forward(x));
      x = outputLayer->forward(x);
      // Pass the result through the output layer
      return x;
    }
  };
  int64_t cudaDev = -1, hiddenLayerDim = 0, MLTrainBatchSize = 64, MLTrainEpochs = 10,cudaDevInference = -1;
  double learningRate = 0.01;
  MLModel model;
  torch::Tensor customLoss(torch::Tensor yOut,torch::Tensor yExpect);
 public:
  MLPGravastarModel() {}
  ~MLPGravastarModel() {}
  /**
   * @brief init the model class
   * @param inputDim the dimension of model ending input
   * @param outputDim the dimension of model ending output
   * @param extraConfig optional extra configs
   * @note accepted configurations
   * - cudaDevice the cuda device to build model, I64, default -1 (cpu only)
   * - cudaDeviceInference the cuda device for inference, I64, default -1 (cpu only)
   * - learningRate the learning rate for training, Double, default 0.01
   * - hiddenLayerDim the dimension of hidden layer, I64, default the same as output layer
   * - MLTrainBatchSize the batch size of ML training, I64, default 64
   * - MLTrainMargin the margin value in regulating variance used in training, Double, default 0
   * - MLTrainEpochs the number of epochs in training, I64, default 10
   */
  virtual void init(int64_t inputDim, int64_t outputDim, INTELLI::ConfigMapPtr extraConfig);
  /**
   * @brief the training function
   * @param x an 2D tensor sized [n*d]
   * @param logout whether or not log the training process out
   */
  virtual void trainModel(torch::Tensor &x1,int64_t logout=0);
  /**
   * @brief the forward function
   * @param input The input tensor
   * @return the output tensor for encoding
   */
  virtual torch::Tensor forward(torch::Tensor &input);
};
/**
 * @}
 */
}
#endif //CANDYBENCH_INCLUDE_CANDY_GRID2DINDEX_MLPGRAVASTARMODEL_H_
