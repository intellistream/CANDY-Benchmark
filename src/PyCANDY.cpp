//
// Created by tony on 12/04/24.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <Utils/IntelliLog.h>
#include <CANDY/AbstractIndex.h>
#include <DataLoader/DataLoaderTable.h>
#include <CANDY/IndexTable.h>
#include <CANDY/DAGNNIndex.h>
#include <include/papi_config.h>

#if CANDY_PAPI == 1
#include <Utils/ThreadPerfPAPI.hpp>
#endif
namespace py = pybind11;
using namespace INTELLI;
using namespace CANDY;
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
  return a + b;
}
py::dict configMapToDict(const std::shared_ptr<ConfigMap>& cfg) {
  py::dict d;
  auto i64Map=cfg->getI64Map();
  auto doubleMap=cfg->getDoubleMap();
  auto strMap=cfg->getStrMap();
  for (auto &iter : i64Map) {
    d[py::cast(iter.first)]=py::cast(iter.second);
  }
  for (auto &iter : doubleMap) {
    d[py::cast(iter.first)]=py::cast(iter.second);
  }
  for (auto &iter : strMap) {
    d[py::cast(iter.first)]=py::cast(iter.second);
  }
  return d;
}

// Function to convert Python dictionary to ConfigMap
std::shared_ptr<ConfigMap> dictToConfigMap(const py::dict& dict) {
  auto cfg = std::make_shared<ConfigMap>();
  for (auto item : dict) {
    auto key = py::str(item.first);
    auto value = item.second;
    // Check if the type is int
    if (py::isinstance<py::int_>(value)) {
      int64_t val = value.cast<int64_t>();
      cfg->edit(key,val);
      //std::cout << "Key: " << key.cast<std::string>() << " has an int value." << std::endl;
    }
      // Check if the type is float
    else if (py::isinstance<py::float_>(value)) {
      double val = value.cast<float>();
      cfg->edit(key,val);
    }
      // Check if the type is string
    else if (py::isinstance<py::str>(value)) {
      std::string val =py::str(value);
      cfg->edit(key,val);
    }
      // Add more type checks as needed
    else {
      std::cout << "Key: " << key.cast<std::string>() << " has a value of another type." << std::endl;
    }
  }
  return cfg;
}
AbstractIndexPtr createIndex(std::string nameTag) {
  IndexTable tab;
  auto ru= tab.getIndex(nameTag);
  if(ru==nullptr){
    INTELLI_ERROR("No index named "+nameTag+", return flat");
    nameTag="flat";
    return tab.getIndex(nameTag);
  }
  return ru;
}
AbstractDataLoaderPtr  creatDataLoader(std::string nameTag) {
  DataLoaderTable dt;
  auto ru= dt.findDataLoader(nameTag);
  if(ru==nullptr){
    INTELLI_ERROR("No index named "+nameTag+", return flat");
    nameTag="random";
    return dt.findDataLoader(nameTag);
  }
  return ru;
}

PYBIND11_MODULE(PyCANDY, m) {
  /**
   * @brief export the configmap class
   */
  m.attr("__version__") = "0.1.0";  // Set the version of the module
  py::class_<INTELLI::ConfigMap,std::shared_ptr<INTELLI::ConfigMap>>(m, "ConfigMap")
      .def(py::init<>())
      .def("edit", py::overload_cast<const std::string &, int64_t>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, double>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, std::string>(&INTELLI::ConfigMap::edit))
      .def("toString", &INTELLI::ConfigMap::toString,
           py::arg("separator") = "\t",
           py::arg("newLine") = "\n")
      .def("toFile", &ConfigMap::toFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n")
      .def("fromFile", &ConfigMap::fromFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n");
  m.def("configMapToDict", &configMapToDict, "A function that converts ConfigMap to Python dictionary");
  m.def("dictToConfigMap", &dictToConfigMap, "A function that converts  Python dictionary to ConfigMap");
  /***
   * @brief abstract index
   */
  py::class_<AbstractIndex,std::shared_ptr<AbstractIndex>>(m, "AbstractIndex")
      .def(py::init<>())
      .def("setTier", &AbstractIndex::setTier)
      .def("reset", &AbstractIndex::reset, py::call_guard<py::gil_scoped_release>())
      .def("setConfigClass", &AbstractIndex::setConfigClass, py::call_guard<py::gil_scoped_release>())
      .def("setConfig", &AbstractIndex::setConfig, py::call_guard<py::gil_scoped_release>())
      .def("startHPC", &AbstractIndex::startHPC)
      .def("insertTensor", &AbstractIndex::insertTensor)
      .def("loadInitialTensor", &AbstractIndex::loadInitialTensor)
      .def("deleteTensor", &AbstractIndex::deleteTensor)
      .def("reviseTensor", &AbstractIndex::reviseTensor)
      .def("searchIndex", &AbstractIndex::searchIndex)
      .def("rawData", &AbstractIndex::rawData)
      .def("searchTensor", &AbstractIndex::searchTensor)
      .def("endHPC", &AbstractIndex::endHPC)
      .def("setFrozenLevel", &AbstractIndex::setFrozenLevel)
      .def("offlineBuild", &AbstractIndex::offlineBuild)
      .def("waitPendingOperations", &AbstractIndex::waitPendingOperations)
      .def("loadInitialStringObject", &AbstractIndex::loadInitialStringObject)
      .def("insertStringObject", &AbstractIndex::insertStringObject)
      .def("deleteStringObject", &AbstractIndex::deleteStringObject)
      .def("searchStringObject", &AbstractIndex::searchStringObject)
      .def("searchTensorAndStringObject", &AbstractIndex::searchTensorAndStringObject)
      .def("loadInitialTensorAndQueryDistribution", &AbstractIndex::loadInitialTensorAndQueryDistribution);
  m.def("createIndex", &createIndex, "A function to create new index by name tag");
  m.def("createDataLoader", &creatDataLoader, "A function to create new data loader by name tag");
  m.def("add_tensors", &add_tensors, "A function that adds two tensors");

  py::class_<DynamicTuneHNSW::GraphStates, std::shared_ptr<DynamicTuneHNSW::GraphStates>>(m, "GraphStates")
    .def(py::init<>())
    .def_readonly("global_stat", &DynamicTuneHNSW::GraphStates::global_stat)
    .def_readonly("time_local_stat", &DynamicTuneHNSW::GraphStates::time_local_stat)
    .def_readonly("window_states", &DynamicTuneHNSW::GraphStates::window_states);
  py::class_<DynamicTuneHNSW::GlobalGraphStats, std::shared_ptr<DynamicTuneHNSW::GlobalGraphStats>>(m, "GlobalGraphStats")
    .def(py::init<>())
    .def_readonly("degree_sum", &DynamicTuneHNSW::GlobalGraphStats::degree_sum)
    .def_readonly("degree_variance", &DynamicTuneHNSW::GlobalGraphStats::degree_variance)
    .def_readonly("neighbor_distance_sum", &DynamicTuneHNSW::GlobalGraphStats::neighbor_distance_sum)
    .def_readonly("neighbor_distance_variance", &DynamicTuneHNSW::GlobalGraphStats::neighbor_distance_variance)
    .def_readonly("steps_taken_max", &DynamicTuneHNSW::GlobalGraphStats::steps_taken_max)
    .def_readonly("steps_taken_avg", &DynamicTuneHNSW::GlobalGraphStats::steps_taken_avg)
    .def_readonly("steps_expansion_average", &DynamicTuneHNSW::GlobalGraphStats::steps_expansion_average)
    .def_readonly("value_average", &DynamicTuneHNSW::GlobalGraphStats::value_average)
    .def_readonly("value_variance", &DynamicTuneHNSW::GlobalGraphStats::value_variance)
    .def_readonly("ntotal",&DynamicTuneHNSW::GlobalGraphStats::ntotal);

  py::class_<DynamicTuneHNSW::BatchDataStates, std::shared_ptr<DynamicTuneHNSW::BatchDataStates>>(m,"BatchDataStates")
    .def(py::init<>())
    .def_readonly("old_ntotal", &DynamicTuneHNSW::BatchDataStates::old_ntotal)
    .def_readonly("degree_sum_new", &DynamicTuneHNSW::BatchDataStates::degree_sum_new)
    .def_readonly("degree_variance_new",&DynamicTuneHNSW::BatchDataStates::degree_variance_new)
    .def_readonly("degree_variance_old", &DynamicTuneHNSW::BatchDataStates::degree_variance_old)
    .def_readonly("degree_sum_old", &DynamicTuneHNSW::BatchDataStates::degree_sum_old)
    .def_readonly("neighbor_distance_sum_new", &DynamicTuneHNSW::BatchDataStates::neighbor_distance_sum_new)
    .def_readonly("neighbor_distance_variance_new", &DynamicTuneHNSW::BatchDataStates::neighbor_distance_variance_new)
    .def_readonly("neighbor_distance_sum_old", &DynamicTuneHNSW::BatchDataStates::neighbor_distance_sum_old)
    .def_readonly("neighbor_distance_variance_old", &DynamicTuneHNSW::BatchDataStates::neighbor_distance_variance_old)
    .def_readonly("steps_taken_sum", &DynamicTuneHNSW::BatchDataStates::steps_taken_sum)
    .def_readonly("steps_taken_max", &DynamicTuneHNSW::BatchDataStates::steps_taken_max)
    .def_readonly("steps_expansion_sum", &DynamicTuneHNSW::BatchDataStates::steps_expansion_sum)
    .def_readonly("value_average", &DynamicTuneHNSW::BatchDataStates::value_average)
    .def_readonly("value_variance", &DynamicTuneHNSW::BatchDataStates::value_variance)
    .def_readonly("ntotal", &DynamicTuneHNSW::BatchDataStates::ntotal);

  py::class_<DynamicTuneHNSW::WindowStates, std::shared_ptr<DynamicTuneHNSW::WindowStates>>(m, "WindowStates")
    .def(py::init<>())
    .def("getCount", &DynamicTuneHNSW::WindowStates::get_count)
    .def_readonly("oldWindowSize", &DynamicTuneHNSW::WindowStates::oldWindowSize)
    .def_readonly("newWindowSize", &DynamicTuneHNSW::WindowStates::newWindowSize)
    .def_readonly("hierarchyWindowSize", &DynamicTuneHNSW::WindowStates::hierarchyWindowSize);

  py::class_<DAGNNIndex, std::shared_ptr<DAGNNIndex>>(m, "DAGNNIndex")
        .def(py::init<>())
        .def("setConfig",&DAGNNIndex::setConfig, py::call_guard<py::gil_scoped_release>())
        .def("insertTensor", &DAGNNIndex::insertTensor)
        .def("loadInitialTensor", &DAGNNIndex::loadInitialTensor)
        .def("searchIndex", &DAGNNIndex::searchIndex)
        .def("searchTensor", &DAGNNIndex::searchTensor)
        .def("getState", &DAGNNIndex::getState)
        .def("performAction", &DAGNNIndex::performAction)
        .def("getReward", &DAGNNIndex::getReward)
        .def("setTraining", &DAGNNIndex::setTraining);


  /**
   * @brief perf
   */
  py::class_<CANDY::AbstractDataLoader,std::shared_ptr<CANDY::AbstractDataLoader>>(m, "AbstractDataLoader")
      .def(py::init<>())
      .def("setConfig", &CANDY::AbstractDataLoader::setConfig)
      .def("getData", &CANDY::AbstractDataLoader::getData)
      .def("getQuery", &CANDY::AbstractDataLoader::getQuery);


#if CANDY_PAPI == 1
  py::class_<INTELLI::ThreadPerfPAPI,std::shared_ptr<INTELLI::ThreadPerfPAPI>>(m, "PAPIPerf")
      .def(py::init<>())
      .def("initEventsByCfg", &INTELLI::ThreadPerfPAPI::initEventsByCfg)
      .def("start", &INTELLI::ThreadPerfPAPI::start)
      .def("end", &INTELLI::ThreadPerfPAPI::end)
      .def("resultToConfigMap", &INTELLI::ThreadPerfPAPI::resultToConfigMap);
#endif
}
