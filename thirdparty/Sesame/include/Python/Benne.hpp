#ifndef SESAME_PYTHON_INCLUDE_BENNE_HPP_
#define SESAME_PYTHON_INCLUDE_BENNE_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/AlgorithmFactory.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Algorithm/Param.hpp"

#include <iostream>
#include <vector>

using namespace pybind11::literals;
using namespace SESAME;
using namespace std;
namespace py = pybind11;

class Benne {
public:
  Benne(int dim, int n_clusters, int obj, double distance_threshold,
        int queue_size_threshold, int dim_threshold, double variance_threshold,
        int outliers_num_threshold, double outliers_dist_threshold)
      : dim(dim), n_clusters(n_clusters), obj((BenneObj)obj),
        distance_threshold(distance_threshold),
        queue_size_threshold(queue_size_threshold),
        dim_threshold(dim_threshold), variance_threshold(variance_threshold),
        outliers_num_threshold(outliers_num_threshold),
        outliers_dist_threshold(outliers_dist_threshold) {
    param.algo = AlgoType::BenneType;
    param.landmark = 10;
    param.num_clusters = n_clusters;
    param.dim = dim;
    param.coreset_size = 2;
    param.obj = (BenneObj)obj;
    param.distance_threshold = distance_threshold;
    param.benne_threshold.queue_size = queue_size_threshold;
    param.benne_threshold.dim = dim_threshold;
    param.benne_threshold.variance = variance_threshold;
    param.benne_threshold.outliers_num = outliers_num_threshold;
    param.benne_threshold.outliers_dist = outliers_dist_threshold;
    algo = AlgorithmFactory::create(param);
    sinkPtr = GenericFactory::New<DataSink>(param);
    auto barrierPtr = GenericFactory::New<barrier<>>(1);
    sinkPtr->setBarrier(barrierPtr);
    sinkPtr->start(0);
    algo->Init();
  }

  Benne &fit(py::array_t<double> X) {
    algo = AlgorithmFactory::create(param);
    sinkPtr = GenericFactory::New<DataSink>(param);
    auto barrierPtr = GenericFactory::New<barrier<>>(1);
    sinkPtr->setBarrier(barrierPtr);
    sinkPtr->start(0);
    algo->Init();

    partial_fit(X);
    return *this;
  }

  py::array_t<int> fit_predict(py::array_t<double> X) {
    py::buffer_info buf = X.request();

    if (buf.ndim != 2)
      throw std::runtime_error("numpy.ndarray dims must be 2!");

    double *ptr = static_cast<double *>(buf.ptr);

    auto num_elements = buf.shape[0] * buf.shape[1];
    auto num_vectors = num_elements / dim;
    for (int i = 0; i < num_vectors; i++) {
      PointPtr point = std::make_shared<Point>(dim, id++, ptr + i * dim);
      inputs.push_back(point);
      algo->RunOnline(point);
    }

    algo->RunOffline(sinkPtr);
    std::vector<PointPtr> &results = sinkPtr->getResults(), predicts;
    UtilityFunctions::groupByCenters(inputs, results, predicts, param.dim);
    std::vector<int> labels;
    for (auto &point : predicts) {
      labels.push_back(point->clu_id);
    }
    return py::array_t<int>(labels.size(), labels.data());
  }

  py::dict get_params() {
    return py::dict("distance_threshold"_a = distance_threshold,
                    "queue_size_threshold"_a = queue_size_threshold,
                    "dim_threshold"_a = dim_threshold,
                    "variance_threshold"_a = variance_threshold,
                    "outliers_num_threshold"_a = outliers_num_threshold,
                    "outliers_dist_threshold"_a = outliers_dist_threshold,
                    "n_clusters"_a = n_clusters);
  }

  Benne &partial_fit(py::array_t<double> X) {
    py::buffer_info buf = X.request();

    if (buf.ndim != 2)
      throw std::runtime_error("numpy.ndarray dims must be 2!");

    double *ptr = static_cast<double *>(buf.ptr);

    auto num_elements = buf.shape[0] * buf.shape[1];
    auto num_vectors = num_elements / dim;
    for (int i = 0; i < num_vectors; i++) {
      PointPtr point = std::make_shared<Point>(dim, id++, ptr + i * dim);
      algo->RunOnline(point);
    }
    return *this;
  }

  py::array_t<int> predict(py::array_t<double> X) {
    py::buffer_info buf = X.request();

    if (buf.ndim != 2)
      throw std::runtime_error("numpy.ndarray dims must be 2!");

    double *ptr = static_cast<double *>(buf.ptr);

    vector<PointPtr> inputs;

    auto num_elements = buf.shape[0] * buf.shape[1];
    auto num_vectors = num_elements / dim;
    for (int i = 0; i < num_vectors; i++) {
      SESAME::PointPtr point =
          std::make_shared<SESAME::Point>(dim, id++, ptr + i * dim);
      inputs.push_back(point);
    }
    algo->RunOffline(sinkPtr);
    std::vector<PointPtr> &results = sinkPtr->getResults(), predicts;
    UtilityFunctions::groupByCenters(inputs, results, predicts, param.dim);
    std::vector<int> labels;
    for (auto &point : predicts) {
      labels.push_back(point->clu_id);
    }
    return py::array_t<int>(labels.size(), labels.data());
  }

  void set_output(bool output) { throw std::runtime_error("Not supported"); }

  void set_params(py::dict params) {
    throw std::runtime_error("Not supported");
  }

private:
  double distance_threshold = 0.5;
  int queue_size_threshold = 1;
  int dim_threshold = 30;
  double variance_threshold = 100.0;
  int outliers_num_threshold = 200;
  double outliers_dist_threshold = 50.0;
  BenneObj obj;
  int n_clusters = 2;
  int dim;
  uint64_t id = 0;
  SESAME::SesameParam param;
  SESAME::AlgorithmPtr algo;
  std::vector<PointPtr> inputs;
  DataSinkPtr sinkPtr;
};

#endif // SESAME_PYTHON_INCLUDE_BENNE_HPP_