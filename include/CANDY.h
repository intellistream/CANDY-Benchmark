/*! \file CANDY.h*/
//
// Created by tony on 22/12/23.
//

#ifndef INTELLISTREAM_CANDY_H
#define INTELLISTREAM_CANDY_H

#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <memory>
/**
* @mainpage Introduction
* This project is an index library and benchmark kit for online vector management, covering various AKNN algos, datasets, online insert benchmark, and examples for more fancy downstream tasks.
* @section supported data format
* The api interface is torch::Tensor for both c++ and python, and we also include support for loading the following data formats from file
* - *.fvecs, (http://corpus-texmex.irisa.fr/) using @ref FVECSDataLoader, a static public class function @ref tensorFromFVECS is also provided
* - *.h5, *.hdf5 (https://github.com/HDFGroup/hdf5) using @ref HDF5DataLoader, a static public class function @ref tensorFromHDF5 is also provided
    * - experimental feature, should using -DENABLE_HDF5=ON in cmake
    * - not support compression yet
* @section sec_name_index Built-in name tags
* @subsection subsec_tag_index Of index approaches (Please go to class @ref IndexTable for more details)
 * - flat @ref FlatIndex
 * - parallelPartition @ref ParallelPartitionIndex
 * - onlinePQ @ref OnlinePQIndex
 * - onlineIVFLSH @ref OnlineIVFLSHIndex
 * - HNSWNaive @ref HNSWNaiveIndex
 * - faiss @ref FaissIndex
 * - congestionDrop @ref CongestionDropIndex
 * - bufferedCongestionDrop @ref BufferedCongestionDropIndex
 * - flatAMMIP @ref FlatAMMIPIndex
* @subsection subsec_tag_loader Of data loaders (Please go to class @ref DataLoaderTable for more details)
* - random @ref RandomDataLoader
* - fvecs @ref FVECSDataLoader
* - hdf5 @ref HDF5DataLoader
* - zipf @ref ZipfDataLoader
* - expFamily @ref ExpFamilyDataLoader
* - exp, the exponential distribution in  @ref ExpFamilyDataLoader
* - beta, the beta distribution in  @ref ExpFamilyDataLoader
* - gaussian, the beta distribution in  @ref ExpFamilyDataLoader
* - poisson, the poisson distribution in  @ref ExpFamilyDataLoader
* @section sec_benchmark Built-in benchmarks
* @subsection subsec_onlineInsert The online insert benchmark
* This benchmark program evaluates the inserting latency and recall of a specified index, the usage is
* ./onlineInsert <name of config file>
* @note required parameters
* - vecDim, the dimension of vector, I64, default 768,
* - vecVolume, the volume of row tensors, I64, default value depends on the DataLoader
* - eventRateTps, the event rate of tuples, each tuple is a row, default 100
* - querySize, the size of your query, I64, default value depends on the DataLoader
* - cutOffTimeSeconds, the setting time to cut off execution after given seconds, default -1 (no cut off), I64
* - batchSize, the size of batch, I64, default equal to the vecVolume
* - staticDataSet, turn on this to force data to be static and make everything already arrived, I64, default 0
* - indexTag, the name tag of index class, String, default flat
* - dataLoaderTag,  the name tag of data loader class, String, default random
* - initialRows, the rows of initially loaded tensors, I64, default 0 (streaming at the begining)
* - waitPendingWrite, wether or not wait for pending writes before start a query, I64, default 0 (NOT)
* see also @ref DataLoaderTable, @ref IndexTable
* * @subsection subsec_onlineCUD The online create, update, delete benchmark (Still working, not support all ANNS)
* This is an upgraded version of onlineInsert @ref subsec_onlineInsert, which will firstly construct AKNN, then delete some tensor, and
* finally conduct the insert, the usage is
* ./onlineCUD <name of config file>
* @note additional parameters compared with onlineInsert
* - deleteRows, the number of rows you want to delete before insert, I64, default 0
* @subsection subsec_multiRW The sequential multiple Read write  benchmark
* This benchmark program evaluates the inserting latency and recall of a specified index, but with multiple RW sequences
* ./multiRW <name of config file>
* @note additional parameters compared with @ref subsec_onlineInsert
* - numberOfRWSeq, the number of RW sequences, will divide both data base tensor and query tensor by this factor, I64, default 1
* @section subsec_extend_cpp_operator How to extend a index algorithm (pure static c++ based)
* - go to the src/CANDY and include/CANDY
* - copy the example class, such as FlatIndex, rename it, and implement your own index class
*  - copy the cpp and h
*  - rename the cpp and h
*  - automatically conduct the IDE-full-replace over the template by your own name in cpp and h
*  - define your own function
*  - @note Please use this copy-and-replace policy rather than creat your own, unless you know the doxygen comment style
    *  very well and can always keep it!!!
*  - @warning  This copy-and-replace policy will also prevent from wrong parameter types of interface functions, please
*  DO KEEP THE INTERFACE PARAMETER UNDER THE SAME TYPE!!!!!!!!!!!
* - register our class with a tag to src/CANDY/IndexTable.cpp
* - edit the CMakelist.txt at src/CANDY to include your new algo and recompile
* - remember to add a test bench, you can refer to FlatIndexTest.cpp at test/SystemTest for example
* @section subsec_edit_test How to add a single point test
* - follow and copy the SimpleTest.cpp to create your own, say A.cpp
* - register A.cpp to test/CMakeLists.txt, please follow how we deal with the SketchTest.cpp
* - assuming you have made A.cpp into a_test, append  ./a_test "--success" to the last row of .github/workflows/cmake.yml
*
* @section python_doc Python Documents
* - Please find the class named @ref Candy_Python for python APIs (old style)
* - Please enable pybind build and install the *.so to system path, you can import PyCANDY, see benchmark/scripts/PyCANDY for details
**/
/**
*
*/
//The groups of modules
/**
 * @mainpage Code Structure
 *  @section Code_Structure  Code Structure
 */
/**
 * @subsection code_stru_dataloader DataLoader
 * This folder contains the loader  under different generation rules
 * @defgroup CANDY_DataLOADER The data loaders of CANDY
 * @{
 * We define the generation classes of DATA. here
 **/
/**
 * @}
 */

/**
* @subsection code_stru_cppalgo BODY
* This folder contains the main body
* @defgroup CANDY_lib The main body of CANDY's indexing approaches
* @{
**/
#include <CANDY/IndexTable.h>
/**
* @defgroup   CANDY_lib_bottom The bottom tier of indexing alorithms
* @{
**/
#include <CANDY/AbstractIndex.h>
#include <CANDY/FlatIndex.h>

/**
 * @}
 */


/**
* @defgroup   CANDY_lib_container The upper tier of indexing alorithms, can be container of other indexing ways
* @{
**/
#include <CANDY/ParallelPartitionIndex.h>
#include <include/ray_config.h>
#if CANDY_RAY == 1
#include <CANDY/DistributedPartitionIndex.h>
#endif
/**
 * @}
 */

/**
 * @}
 */

/**
* @subsection code_stru_dataloader DL
* This folder contains the dataloader
* @defgroup  CANDY_DataLOADER The data loader of CANDY
* @{
* We define the data loader classes . here
**/
#include <DataLoader/AbstractDataLoader.h>
#include <DataLoader/DataLoaderTable.h>
#include <DataLoader/RandomDataLoader.h>
#include <DataLoader/FVECSDataLoader.h>
//#include <include/hdf5_config.h>
//#if CANDY_HDF5 == 1
//#include <DataLoader/HDF5DataLoader.h>
//#endif
/**
 * @}
 *
 */
/**
 *  @subsection code_stru_utils Utils
* This folder contains the public utils shared by INTELISTREAM team and some third party dependencies.
 **/
/**
* @defgroup INTELLI_UTIL Shared Utils
* @{
*/
#include <Utils/ConfigMap.hpp>
#include <Utils/Meters/MeterTable.h>
/**
 * @ingroup INTELLI_UTIL
* @defgroup INTELLI_UTIL_OTHERC20 Other common class or package under C++20 standard
* @{
* This package covers some common C++20 new features, such as std::thread to ease the programming
*/
#include <Utils/C20Buffers.hpp>
#include <Utils/ThreadPerf.hpp>
#include <include/papi_config.h>
#if CANDY_PAPI == 1
#include <Utils/ThreadPerfPAPI.hpp>
#endif
#include <Utils/IntelliLog.h>
#include <Utils/UtilityFunctions.h>
//#include <Utils/BS_thread_pool.hpp>
#include <Utils/IntelliTensorOP.hpp>
#include <Utils/IntelliTimeStampGenerator.h>

/**
 * @}
 */
/**
 *
 * @}
 */

#endif
