//
// Created by tony on 12/04/24.
//
#include <gflags/gflags.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <Utils/IntelliLog.h>
#include <Utils/SPSCQueue.hpp>
#include <CANDY/AbstractIndex.h>


#include <CANDY/IndexTable.h>
#include <include/papi_config.h>
#if CANDY_PAPI == 1
#include <Utils/ThreadPerfPAPI.hpp>
#endif

#include <faiss/index_factory.h>
#include<puck/pyapi_wrapper/py_api_wrapper.h>

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
//AbstractIndexPtr createIndex(std::string nameTag) {
//  IndexTable tab;
//  auto ru = tab.getIndex(nameTag);
//  if (ru == nullptr) {
//    INTELLI_ERROR("No index named " + nameTag + ", return flat");
//    nameTag = "flat";
//    return tab.getIndex(nameTag);
//  }
//  return ru;
//}

AbstractIndexPtr createIndex(std::string nameTag, int64_t dim) {
    IndexTable tab;
    auto ru = tab.getIndex(nameTag);
    if (ru == nullptr) {
        INTELLI_ERROR("No index named " + nameTag + ", return flat");
        nameTag = "flat";
        return tab.getIndex(nameTag);
    }
    ConfigMapPtr cfg = newConfigMap();
    cfg->edit("vecDim", dim);
    ru->setConfig(cfg);

    return ru;
}

//AbstractDataLoaderPtr creatDataLoader(std::string nameTag) {
//  DataLoaderTable dt;
//  auto ru = dt.findDataLoader(nameTag);
//  if (ru == nullptr) {
//    INTELLI_ERROR("No index named " + nameTag + ", return flat");
//    nameTag = "random";
//    return dt.findDataLoader(nameTag);
//  }
//  return ru;
//}
static bool existRow(torch::Tensor base, torch::Tensor row) {
  for (int64_t i = 0; i < base.size(0); i++) {
    auto tensor1 = base[i].contiguous();
    auto tensor2 = row.contiguous();
    //std::cout<<"base: "<<tensor1<<std::endl;
    //std::cout<<"query: "<<tensor2<<std::endl;
    if (torch::equal(tensor1, tensor2)) {
      return true;
    }
  }
  return false;
}
double recallOfTensorList(std::vector<torch::Tensor> groundTruth, std::vector<torch::Tensor> prob) {
  int64_t truePositives = 0;
  int64_t falseNegatives = 0;
  for (size_t i = 0; i < prob.size(); i++) {
    auto gdI = groundTruth[i];
    auto probI = prob[i];
    for (int64_t j = 0; j < probI.size(0); j++) {
      if (existRow(gdI, probI[j])) {
        truePositives++;
      } else {
        falseNegatives++;
      }
    }
  }
  double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives);
  return recall;
}


template <class DT>
class NumpyIdxPair{
public:
    NumpyIdxPair(){};
    ~NumpyIdxPair(){};
    std::vector<int64_t> idx;
    py::array_t<DT> vectors;
    NumpyIdxPair(const py::array_t<DT> _vectors, const std::vector<int64_t> _idx){
        idx=_idx;
        vectors = _vectors;
    }
};

using NumpyIdxPairInt8 = NumpyIdxPair<int8_t>;
using NumpyIdxPairFloat = NumpyIdxPair<float>;

using NumpyIdxQueueInt8 = SPSCQueue<NumpyIdxPairInt8>;
using NumpyIdxQueueFloat = SPSCQueue<NumpyIdxPairFloat>;

template<class DT>
class SPSCWrapperNumpy{
public:
    SPSCWrapperNumpy(size_t capacity){
        queue = new SPSCQueue<NumpyIdxPair<DT>>(capacity);
    }
    SPSCQueue<NumpyIdxPair<DT>>* queue;
    void push(NumpyIdxPair<DT> &obj){
        queue->push(obj);
    }
    bool try_push(NumpyIdxPair<DT> &obj){
        return queue->try_push(obj);
    }
    NumpyIdxPair<DT> front(){

        auto temp = NumpyIdxPair<DT>(queue->front()->vectors, queue->front()->idx);
        return temp;
    }
    void pop(){
        queue->pop();
    }
    size_t capacity(){
        return queue->capacity();
    }
    size_t size(){
        return queue->size();
    }
    bool empty(){
        return queue->empty();
    }
};

class SPSCWrapperIdx{
public:
    SPSCWrapperIdx(size_t capacity){
        queue = new SPSCQueue<int64_t>(capacity);
    }
    SPSCQueue<int64_t>* queue;
    void push(int64_t &obj){
        queue->push(obj);
    }
    bool try_push(int64_t &obj){
        return queue->try_push(obj);
    }
    int64_t front(){

        auto temp = *(queue->front());
        return temp;
    }
    void pop(){
        queue->pop();
    }
    size_t capacity(){
        return queue->capacity();
    }
    size_t size(){
        return queue->size();
    }
    bool empty(){
        return queue->empty();
    }

};


#define COMPILED_TIME (__DATE__ " " __TIME__)
PYBIND11_MODULE(PyCANDYAlgo, m) {
  /**
   * @brief export the configmap class
   */
  m.attr("__version__") = "0.1.2";  // Set the version of the module
  m.attr("__compiled_time__") = COMPILED_TIME;  // Set the compile time of the module
  m.doc() = "This is top module - PyCANDYAlgo.";
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
      .def("insertTensorWithIds", &AbstractIndex::insertTensorWithIds)
      .def("loadInitialTensor", &AbstractIndex::loadInitialTensor)
      .def("loadInitialTensorWithIds", &AbstractIndex::loadInitialTensorWithIds)
      .def("deleteTensor", &AbstractIndex::deleteTensor)
      .def("deleteIndex", &AbstractIndex::deleteIndex)
      .def("reviseTensor", &AbstractIndex::reviseTensor)
      .def("searchIndex", &AbstractIndex::searchIndex)
      .def("searchIndexParam", &AbstractIndex::searchIndexParam)
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

  m.def("add_tensors", &add_tensors, "A function that adds two tensors");


  m.def("recallOfTensorList", &recallOfTensorList, "calculate the recall");

  /// faiss index APIs only
  py::class_<faiss::Index,std::shared_ptr<faiss::Index>>(m, "IndexFAISS")
         // .def(py::init<>())
          .def("add",&faiss::Index::add_arrays)
          .def("search",&faiss::Index::search_arrays)
          .def("train",&faiss::Index::train_arrays)
          .def("add_with_ids", &faiss::Index::add_arrays_with_ids)
        .def_readwrite("verbose", &faiss::Index::verbose);

  m.def("index_factory_ip", &faiss::index_factory_IP, "Create custom index from faiss with IP");

  m.def("index_factory_l2", &faiss::index_factory_L2, "Create custom index from faiss with IP");





#if CANDY_PAPI == 1
  py::class_<INTELLI::ThreadPerfPAPI, std::shared_ptr<INTELLI::ThreadPerfPAPI>>(m, "PAPIPerf")
      .def(py::init<>())
      .def("initEventsByCfg", &INTELLI::ThreadPerfPAPI::initEventsByCfg)
      .def("start", &INTELLI::ThreadPerfPAPI::start)
      .def("end", &INTELLI::ThreadPerfPAPI::end)
      .def("resultToConfigMap", &INTELLI::ThreadPerfPAPI::resultToConfigMap);
#endif


  auto m_puck = m.def_submodule("puck", "Puck Interface from Baidu.");
  py::class_<py_puck_api::PySearcher, std::shared_ptr<py_puck_api::PySearcher>>(m_puck, "PuckSearcher")
    .def(py::init<>())
    .def("init", &py_puck_api::PySearcher::init)
    .def("show",&py_puck_api::PySearcher::show)
    .def("build",&py_puck_api::PySearcher::build)
    .def("search",&py_puck_api::PySearcher::search)
    .def("batch_add",&py_puck_api::PySearcher::batch_add)
    .def("batch_delete",&py_puck_api::PySearcher::batch_delete);

    m_puck.def("update_gflag", &py_puck_api::update_gflag, "A function to update gflag");

    auto m_utils = m.def_submodule("utils", "Utility Classes from CANDY.");
    py::class_<NumpyIdxPair<float>,std::shared_ptr<NumpyIdxPair<float>>>(m_utils,"NumpyIdxPair")
            .def(py::init<>())
            .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>, std::vector<int64_t>>())
            .def_readwrite("vectors", &NumpyIdxPair<float>::vectors)
            .def_readwrite("idx", &NumpyIdxPair<float>::idx);

    py::class_<SPSCWrapperNumpy<float>, std::shared_ptr<SPSCWrapperNumpy<float>>>(m_utils, "NumpyIdxQueue")
        .def(py::init<const size_t>())
        .def("push", &SPSCWrapperNumpy<float>::push)
        .def("try_push", &SPSCWrapperNumpy<float>::try_push)
        .def("front", &SPSCWrapperNumpy<float>::front)
        .def("empty", &SPSCWrapperNumpy<float>::empty)
        .def("size", &SPSCWrapperNumpy<float>::size)
        .def("capacity", &SPSCWrapperNumpy<float>::capacity)
        .def("pop", &SPSCWrapperNumpy<float>::pop);
    py::class_<SPSCWrapperIdx, std::shared_ptr<SPSCWrapperIdx>>(m_utils, "IdxQueue")
            .def(py::init<const size_t>())
            .def("push", &SPSCWrapperIdx::push)
            .def("try_push", &SPSCWrapperIdx::try_push)
            .def("front", &SPSCWrapperIdx::front)
            .def("empty", &SPSCWrapperIdx::empty)
            .def("size", &SPSCWrapperIdx::size)
            .def("capacity", &SPSCWrapperIdx::capacity)
            .def("pop", &SPSCWrapperIdx::pop);


}