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
py::dict configMapToDict(const std::shared_ptr<ConfigMap> &cfg) {
  py::dict d;
  auto i64Map = cfg->getI64Map();
  auto doubleMap = cfg->getDoubleMap();
  auto strMap = cfg->getStrMap();
  for (auto &iter : i64Map) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : doubleMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : strMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  return d;
}

// Function to convert Python dictionary to ConfigMap
std::shared_ptr<ConfigMap> dictToConfigMap(const py::dict &dict) {
  auto cfg = std::make_shared<ConfigMap>();
  for (auto item : dict) {
    auto key = py::str(item.first);
    auto value = item.second;
    // Check if the type is int
    if (py::isinstance<py::int_>(value)) {
      int64_t val = value.cast<int64_t>();
      cfg->edit(key, val);
      //std::cout << "Key: " << key.cast<std::string>() << " has an int value." << std::endl;
    }
      // Check if the type is float
    else if (py::isinstance<py::float_>(value)) {
      double val = value.cast<float>();
      cfg->edit(key, val);
    }
      // Check if the type is string
    else if (py::isinstance<py::str>(value)) {
      std::string val = py::str(value);
      cfg->edit(key, val);
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
  auto ru = tab.getIndex(nameTag);
  if (ru == nullptr) {
    INTELLI_ERROR("No index named " + nameTag + ", return flat");
    nameTag = "flat";
    return tab.getIndex(nameTag);
  }
  return ru;
}
AbstractDataLoaderPtr creatDataLoader(std::string nameTag) {
  DataLoaderTable dt;
  auto ru = dt.findDataLoader(nameTag);
  if (ru == nullptr) {
    INTELLI_ERROR("No index named " + nameTag + ", return flat");
    nameTag = "random";
    return dt.findDataLoader(nameTag);
  }
  return ru;
}

PYBIND11_MODULE(PyCANDY, m) {
  /**
   * @brief export the configmap class
   */
  m.attr("__version__") = "0.1.2";  // Set the version of the module
  py::class_<INTELLI::ConfigMap, std::shared_ptr<INTELLI::ConfigMap>>(m, "ConfigMap")
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
  py::class_<AbstractIndex, std::shared_ptr<AbstractIndex>>(m, "AbstractIndex")
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
      .def("loadInitialU64Object", &AbstractIndex::loadInitialU64Object)
      .def("insertStringObject", &AbstractIndex::insertStringObject)
      .def("insertU64Object", &AbstractIndex::insertU64Object)
      .def("deleteStringObject", &AbstractIndex::deleteStringObject)
      .def("deleteU64Object", &AbstractIndex::deleteU64Object)
      .def("searchStringObject", &AbstractIndex::searchStringObject)
      .def("searchU64Object", &AbstractIndex::searchU64Object)
      .def("searchTensorAndStringObject", &AbstractIndex::searchTensorAndStringObject)
      .def("loadInitialTensorAndQueryDistribution", &AbstractIndex::loadInitialTensorAndQueryDistribution)
      .def("resetIndexStatistics", &AbstractIndex::resetIndexStatistics)
      .def("getIndexStatistics", &AbstractIndex::getIndexStatistics);
  m.def("createIndex", &createIndex, "A function to create new index by name tag");
  m.def("createDataLoader", &creatDataLoader, "A function to create new data loader by name tag");
  m.def("add_tensors", &add_tensors, "A function that adds two tensors");
  /**
   * @brief perf
   */
  py::class_<CANDY::AbstractDataLoader, std::shared_ptr<CANDY::AbstractDataLoader>>(m, "AbstractDataLoader")
      .def(py::init<>())
      .def("setConfig", &CANDY::AbstractDataLoader::setConfig)
      .def("getData", &CANDY::AbstractDataLoader::getData)
      .def("getDataAt", &CANDY::AbstractDataLoader::getDataAt)
      .def("getQueryAt", &CANDY::AbstractDataLoader::getQueryAt)
      .def("getQuery", &CANDY::AbstractDataLoader::getQuery);
#if CANDY_PAPI == 1
  py::class_<INTELLI::ThreadPerfPAPI, std::shared_ptr<INTELLI::ThreadPerfPAPI>>(m, "PAPIPerf")
      .def(py::init<>())
      .def("initEventsByCfg", &INTELLI::ThreadPerfPAPI::initEventsByCfg)
      .def("start", &INTELLI::ThreadPerfPAPI::start)
      .def("end", &INTELLI::ThreadPerfPAPI::end)
      .def("resultToConfigMap", &INTELLI::ThreadPerfPAPI::resultToConfigMap);
#endif
}