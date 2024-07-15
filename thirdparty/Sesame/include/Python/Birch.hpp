#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/AlgorithmFactory.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"

#include <barrier>
#include <iostream>

using namespace pybind11::literals;
using namespace SESAME;
using namespace std;
namespace py = pybind11;

class Birch {
public:
  Birch(int dim, int n_clusters, double distance_threshold,
        int branching_factor)
      : distance_threshold(distance_threshold),
        branching_factor(branching_factor), n_clusters(n_clusters), dim(dim) {
    param.algo = AlgoType::BirchType;
    param.distance_threshold = distance_threshold;
    param.landmark = 10;
    param.max_in_nodes = branching_factor;
    param.max_leaf_nodes = branching_factor;
    param.num_clusters = n_clusters;
    param.dim = dim;
    algo = AlgorithmFactory::create(param);
    sinkPtr = GenericFactory::New<DataSink>(param);
    auto barrierPtr = GenericFactory::New<barrier<>>(1);
    sinkPtr->setBarrier(barrierPtr);
    sinkPtr->start(0);
    algo->Init();
  }

  Birch &fit(py::array_t<double> X) {
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
    // cout << "results size: " << results.size() << endl;
    // cout << "inputs size: " << inputs.size() << endl;
    // cout << "predicts size: " << predicts.size() << endl;
    std::vector<int> labels;
    for (auto &point : predicts) {
      labels.push_back(point->clu_id);
    }
    return py::array_t<int>(labels.size(), labels.data());
  }

  py::dict get_params() {
    return py::dict("distance_threshold"_a = distance_threshold,
                    "branching_factor"_a = branching_factor,
                    "n_clusters"_a = n_clusters);
  }

  Birch &partial_fit(py::array_t<double> X) {
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
  double distance_threshold;
  int branching_factor;
  int n_clusters;
  int dim;
  bool output;
  uint64_t id = 0;
  SesameParam param;
  AlgorithmPtr algo;
  std::vector<PointPtr> inputs;
  DataSinkPtr sinkPtr;
};
