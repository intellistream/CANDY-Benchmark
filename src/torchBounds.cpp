//
// Created by tony on 05/01/24.
//

#include <vector>
#include <CANDY.h>
#include <iostream>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
#include <map>
std::map<std::string, CANDY::AbstractIndexPtr> torchBounding_idxMap;
std::map<std::string, INTELLI::ConfigMapPtr> torchBounding_cfgMap;

/**
*
* @brief The c++ bindings to creat an index at backend
* @param name the name of this index
* @param type the type of this index, keep the same as that in CANDY::IndexTable
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_create(string name, string type) {
  CANDY::IndexTable it;
  auto idx = it.getIndex(type);
  if (idx == nullptr) {
    return torch::zeros({1, 1});
  }
  torchBounding_idxMap[name] = idx;
  torchBounding_cfgMap[name] = newConfigMap();
  return torch::zeros({1, 1}) + 1.0;
}
/**
* @brief The c++ bindings to change the config map related to a specific index
* @param name the name of the index
 * @param key the key in the cfg
 * @param value the double value
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_editCfgDouble(string name, string key, double value) {
  if ((torchBounding_cfgMap.count(name) == 1)) // have this index
  {
    torchBounding_cfgMap[name]->edit(key, value);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to load the config map related to a specific index from file
* @param name the name of the index
 * @param fname the name of file
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_loadCfgFromFile(string name, string fname) {
  if ((torchBounding_cfgMap.count(name) == 1)) // have this index
  {
    torchBounding_cfgMap[name]->fromFile(fname);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}

/**
* @brief The c++ bindings to change the config map related to a specific index
* @param name the name of the index
 * @param key the key in the cfg
 * @param value the float value
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_editCfgFloat(string name, string key, float value) {
  if ((torchBounding_cfgMap.count(name) == 1)) // have this index
  {
    double v2 = value;
    torchBounding_cfgMap[name]->edit(key, v2);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to change the config map related to a specific index
* @param name the name of the index
 * @param key the key in the cfg
 * @param value the string value
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_editCfgStr(string name, string key, string value) {
  if ((torchBounding_cfgMap.count(name) == 1)) // have this index
  {
    torchBounding_cfgMap[name]->edit(key, value);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to change the config map related to a specific index
* @param name the name of the index
 * @param key the key in the cfg
 * @param value the I64 value
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_editCfgI64(string name, string key, int64_t value) {
  if ((torchBounding_cfgMap.count(name) == 1)) // have this index
  {
    torchBounding_cfgMap[name]->edit(key, value);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to init an index with its bounded config
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_init(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->setConfig(torchBounding_cfgMap[name]);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to insert tensor to an index
* @param name the name of the index
 * @param t the tensor
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_insert(string name, torch::Tensor t) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->insertTensor(t)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}

/**
* @brief The c++ bindings to search tensor
* @param name the name of the index
 * @param t the tensor
 * @param k the NNS
* @return the list of result tensors
*/
std::vector<torch::Tensor> CANDY_index_searchTensorList(string name, torch::Tensor t, int64_t k) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    auto tensors = torchBounding_idxMap[name]->searchTensor(t, k);
    { return tensors; }
  }
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::zeros({1, 1});
  return ru;
}

/**
* @brief The c++ bindings to delete tensor to an index
* @param name the name of the index
 * @param t the tensor
 * @param k the NNS
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_delete(string name, torch::Tensor t, int64_t k) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->deleteTensor(t, k)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}

/**
* @brief The c++ bindings to revise tensor to an index
* @param name the name of the index
 * @param t the tensor to be revised
 * @param w the revison
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_revise(string name, torch::Tensor t, torch::Tensor &w) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->reviseTensor(t, w)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to return rawData
* @param name the name of the index
* @return tensor of rawData
*/
torch::Tensor CANDY_index_rawData(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    return torchBounding_idxMap[name]->rawData();
  }
  return torch::zeros({1, 1});
}
/**
*
* @brief The c++ bindings to creat an index at backend
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_reset(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->reset();
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
*
* @brief The c++ bindings to start HPC features
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_startHPC(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->startHPC();
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
*
* @brief The c++ bindings to end HPC features
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_endHPC(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->endHPC();
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}

/**
*
* @brief The c++ bindings to set the frozen level of online updating internal state
* @param name the name of the index
 * @param frozenLv the level of frozen, 0 means freeze any online update in internal state
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_setFrozenLevel(string name, int64_t frozenLV) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->setFrozenLevel(frozenLV);
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}

/**
* @brief The c++ bindings to offlineBuild
* @param name the name of the index
 * @param t the tensor
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_offlineBuild(string name, torch::Tensor t) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->offlineBuild(t)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to load initial tensor
* @param name the name of the index
 * @param t the tensor
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_loadInitialTensor(string name, torch::Tensor t) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->loadInitialTensor(t)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to save a tensor into file
 * @param A the tensor
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_export_tensorToFile(torch::Tensor A, std::string fname) {
  if (IntelliTensorOP::tensorToFile(&A, fname)) {
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to load a tensor from file
* @param name the name of the index
* @return the tensor result
*/
torch::Tensor CANDY_export_tensorFromFile(std::string fname) {
  torch::Tensor A;
  if (IntelliTensorOP::tensorFromFile(&A, fname)) {
    return A;
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to load initial tensor along with string objects
* @param name the name of the index
 * @param t the tensor
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_loadInitialString(string name, torch::Tensor t, std::vector<std::string> s) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->loadInitialStringObject(t, s)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to insert tensor to an index with its binded strings
* @param name the name of the index
 * @param t the tensor
 * @param s the vector of string, List[str] in python
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_insertString(string name, torch::Tensor t, std::vector<std::string> s) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->insertStringObject(t, s)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}

/**
* @brief The c++ bindings to delete tensor to an index and its string object
* @param name the name of the index
 * @param t the tensor
 * @param k the NNS
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_deleteString(string name, torch::Tensor t, int64_t k) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    if (torchBounding_idxMap[name]->deleteStringObject(t, k)) { return torch::zeros({1, 1}) + 1.0; }
  }
  return torch::zeros({1, 1});
}
/**
* @brief The c++ bindings to search binded string of given tensor
* @param name the name of the index
 * @param t the tensor
 * @param k the NNS
* @return List[List[str]], for each rows
*/
std::vector<std::vector<std::string>> CANDY_index_searchStringList(string name, torch::Tensor &q, int64_t k) {

  assert(k > 0);
  assert(q.size(1));
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    auto rut = torchBounding_idxMap[name]->searchStringObject(q, k);
    { return rut; }
  }
  std::vector<std::vector<std::string>> ru(1);
  ru[0] = std::vector<std::string>(0);
  return ru;
}
std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> CANDY_index_searchTensorAndStringList(
    string name,
    torch::Tensor &q,
    int64_t k) {

  assert(k > 0);
  assert(q.size(1));
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    auto ru = torchBounding_idxMap[name]->searchTensorAndStringObject(q, k);
    { return ru; }
  }
  auto ruT = CANDY_index_searchTensorList(name, q, k);
  auto ruS = CANDY_index_searchStringList(name, q, k);
  std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> ru(ruT, ruS);
  return ru;
}

//std::map<std::string, CANDY::AbstractDataLoaderPtr> torchBounding_dataLoaderMap;
std::map<std::string, INTELLI::ConfigMapPtr> torchBounding_cfgDlMap;
///**
//*
//* @brief The c++ bindings to creat an dataLoader at backend
//* @param name the name of this dataLoader
//* @param type the type of this dataLoader, keep the same as that in CANDY::IndexTable
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_create(string name, string type) {
//  CANDY::DataLoaderTable it;
//  auto idx = it.findDataLoader(type);
//  if (idx == nullptr) {
//    return torch::zeros({1, 1});
//  }
//  torchBounding_dataLoaderMap[name] = idx;
//  torchBounding_cfgDlMap[name] = newConfigMap();
//  return torch::zeros({1, 1}) + 1.0;
//}
///**
//* @brief The c++ bindings to change the config map related to a specific dataLoader
//* @param name the name of the dataLoader
// * @param key the key in the cfg
// * @param value the double value
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_editCfgDouble(string name, string key, double value) {
//  if ((torchBounding_cfgDlMap.count(name) == 1)) // have this dataLoader
//  {
//    torchBounding_cfgDlMap[name]->edit(key, value);
//    return torch::zeros({1, 1}) + 1.0;
//  }
//  return torch::zeros({1, 1});
//}
//
///**
//* @brief The c++ bindings to change the config map related to a specific dataLoader
//* @param name the name of the dataLoader
// * @param key the key in the cfg
// * @param value the float value
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_editCfgFloat(string name, string key, float value) {
//  if ((torchBounding_cfgDlMap.count(name) == 1)) // have this dataLoader
//  {
//    double v2 = value;
//    torchBounding_cfgDlMap[name]->edit(key, v2);
//    return torch::zeros({1, 1}) + 1.0;
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to change the config map related to a specific dataLoader
//* @param name the name of the dataLoader
// * @param key the key in the cfg
// * @param value the string value
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_editCfgStr(string name, string key, string value) {
//  if ((torchBounding_cfgDlMap.count(name) == 1)) // have this dataLoader
//  {
//    torchBounding_cfgDlMap[name]->edit(key, value);
//    return torch::zeros({1, 1}) + 1.0;
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to change the config map related to a specific dataLoader
//* @param name the name of the dataLoader
// * @param key the key in the cfg
// * @param value the I64 value
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_editCfgI64(string name, string key, int64_t value) {
//  if ((torchBounding_cfgDlMap.count(name) == 1)) // have this dataLoader
//  {
//    torchBounding_cfgDlMap[name]->edit(key, value);
//    return torch::zeros({1, 1}) + 1.0;
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to init an dataLoader with its bounded config
//* @param name the name of the dataLoader
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_init(string name) {
//  if ((torchBounding_dataLoaderMap.count(name) == 1)) // have this dataLoader
//  {
//    torchBounding_dataLoaderMap[name]->setConfig(torchBounding_cfgDlMap[name]);
//    return torch::zeros({1, 1}) + 1.0;
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to get data tensor from the specified data loader
//* @param name the name of the dataLoader
// * @param t the tensor
//* @return tensor 1x1, [1] for success
//*/
//torch::Tensor CANDY_dataLoader_getData(string name) {
//  if ((torchBounding_dataLoaderMap.count(name) == 1)) // have this dataLoader
//  {
//    return torchBounding_dataLoaderMap[name]->getData();
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to get query tensor from the specified data loader
//* @param name the name of the dataLoader
//* @return the first result tensor
//*/
//torch::Tensor CANDY_dataLoader_getQuery(string name) {
//  if ((torchBounding_dataLoaderMap.count(name) == 1)) // have this dataLoader
//  {
//    return torchBounding_dataLoaderMap[name]->getQuery();
//  }
//  return torch::zeros({1, 1});
//}
///**
//* @brief The c++ bindings to load tensor from fvecs file
//* @param name the name of file
//* return the result tensor
//*/
//torch::Tensor CANDY_tensorFromFVECS(string name) {
//  return CANDY::FVECSDataLoader::tensorFromFVECS(name);
//}
///**
//* @brief The c++ bindings to load tensor from HDF5 file
//* @param name the name of file
//* @param attr the attribute
//* return the result tensor
//*/
//torch::Tensor CANDY_tensorFromHDF5(string name, string attr) {
//#if CANDY_HDF5 == 1
//  return CANDY::HDF5DataLoader::tensorFromHDF5(name, attr);
//#else
//  return torch::zeros({1, 1});
//#endif
//}
/**
*
* @brief The c++ bindings to wait pending operations features
* @param name the name of the index
* @return tensor 1x1, [1] for success
*/
torch::Tensor CANDY_index_waitPendingOperations(string name) {
  if ((torchBounding_idxMap.count(name) == 1)) // have this index
  {
    torchBounding_idxMap[name]->waitPendingOperations();
    return torch::zeros({1, 1}) + 1.0;
  }
  return torch::zeros({1, 1});
}

/**
 * @brief Declare the function to pytorch
 * @note The of lib is myLib
 */
TORCH_LIBRARY(CANDY, m2) {
  m2.def("index_create", CANDY_index_create);
  m2.def("index_loadCfgFromFile", CANDY_index_loadCfgFromFile);
  m2.def("index_editCfgStr", CANDY_index_editCfgStr);
  m2.def("index_ediCfgI64", CANDY_index_editCfgI64);
  m2.def("index_editCfgI64", CANDY_index_editCfgI64);
  m2.def("index_editCfgDouble", CANDY_index_editCfgDouble);
  m2.def("index_init", CANDY_index_init);
  m2.def("index_insert", CANDY_index_insert);
  m2.def("index_offlineBuild", CANDY_index_offlineBuild);
  m2.def("index_loadInitial", CANDY_index_loadInitialTensor);
  m2.def("index_delete", CANDY_index_delete);
  m2.def("index_revise", CANDY_index_revise);
  m2.def("index_search", CANDY_index_searchTensorList);
  m2.def("index_reset", CANDY_index_reset);
  m2.def("index_rawData", CANDY_index_rawData);
  m2.def("index_setFrozenLevel", CANDY_index_setFrozenLevel);
  m2.def("index_startHPC", CANDY_index_startHPC);
  m2.def("index_endHPC", CANDY_index_endHPC);
  m2.def("index_waitPending", CANDY_index_waitPendingOperations);
  m2.def("tensorToFile", CANDY_export_tensorToFile);
  m2.def("tensorFromFile", CANDY_export_tensorFromFile);
  //m2.def("dataLoader_create", CANDY_dataLoader_create);
  //m2.def("dataLoader_editCfgDouble", CANDY_dataLoader_editCfgDouble);
  //m2.def("dataLoader_editCfgStr", CANDY_dataLoader_editCfgStr);
  //m2.def("dataLoader_ediCfgI64", CANDY_dataLoader_editCfgI64);
  //m2.def("dataLoader_getData", CANDY_dataLoader_getData);
  //m2.def("dataLoader_getQuery", CANDY_dataLoader_getQuery);
  //m2.def("dataLoader_init", CANDY_dataLoader_init);
  m2.def("tensorFromFVECS", CANDY_tensorFromFVECS);
  m2.def("tensorFromHDF5", CANDY_tensorFromHDF5);
  m2.def("index_loadInitialString", CANDY_index_loadInitialString);
  m2.def("index_insertString", CANDY_index_insertString);
  m2.def("index_deleteString", CANDY_index_deleteString);
  m2.def("index_searchString", CANDY_index_searchStringList);
  m2.def("index_searchTensorAndString", CANDY_index_searchTensorAndStringList);
  //m2.def("myVecSub", myVecSub);
}
