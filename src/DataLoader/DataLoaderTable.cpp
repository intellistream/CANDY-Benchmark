//
// Created by tony on 10/05/23.
//

#include <DataLoader/DataLoaderTable.h>
#include <DataLoader/RandomDataLoader.h>
#include <DataLoader/ZipfDataLoader.h>
#include <DataLoader/ExpFamilyDataLoader.h>
#include <DataLoader/FVECSDataLoader.h>
#include <DataLoader/RBTDataLoader.h>

namespace CANDY {
static CANDY::AbstractDataLoaderPtr genExpFamilyLoader(INTELLI::ConfigMapPtr cfgHijack, std::string tag) {
  auto expLd = newExpFamilyDataLoader();
  cfgHijack->edit("distributionOverwrite", tag);
  expLd->hijackConfig(cfgHijack);
  return expLd;
}
/**
 * @note revise me if you need new loader
 */
CANDY::DataLoaderTable::DataLoaderTable() {
  loaderMap["null"] = newAbstractDataLoader();
  loaderMap["random"] = newRandomDataLoader();
  loaderMap["fvecs"] = newFVECSDataLoader();
  loaderMap["zipf"] = newZipfDataLoader();
  loaderMap["expFamily"] = newExpFamilyDataLoader();
  /**
   * @brief more specific loader oin exp family
   */
  INTELLI::ConfigMapPtr cfgHijack = newConfigMap();
  loaderMap["exp"] = genExpFamilyLoader(cfgHijack, "exp");
  loaderMap["beta"] = genExpFamilyLoader(cfgHijack, "beta");
  loaderMap["gaussian"] = genExpFamilyLoader(cfgHijack, "gaussian");
  loaderMap["poisson"] = genExpFamilyLoader(cfgHijack, "poisson");
  loaderMap["rbt"] = newRBTDataLoader();
}

} // Algorithms