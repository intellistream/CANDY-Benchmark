/*! \file SPDKSSD.h*/
//
// Created by tony on 23/07/24.
//

#ifndef CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_SPDKSSD_H_
#define CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_SPDKSSD_H_
#include <include/spdk_config.h>
#if CANDY_SPDK == 1
/**
 *  @defgroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <spdk/env.h>
#include <spdk/nvme.h>
#include <mutex>
namespace CANDY {
/**
 * @class SPDKSSD CANDY/FlatSSDGPUIndex/SPDKSSD.h
 * @brief A class to allow random R/W on SSD, possible for concurrent Write on different sectors.
 */

class SPDKSSD {
 public:

  /**
 * @brief Constructor that initializes the SPDK environment and connects to the NVMe controller, will just use the first available ssd
 */
  SPDKSSD(void);

  /**
   * @brief Destructor that frees the SPDK resources.
   */
  ~SPDKSSD();

  /**
   * @brief Writes data to the SSD.
   * @param buffer Data buffer to write.
   * @param size Size of the data buffer.
   * @param offset Offset in the SSD where data should be written.
   * @param qpair Queue pair to use for the write operation.
   */
  void write(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair);

  /**
   * @brief Reads data from the SSD.
   * @param buffer Data buffer to read into.
   * @param size Size of the data buffer.
   * @param offset Offset in the SSD from where data should be read.
   * @param qpair Queue pair to use for the read operation.
   */
  void read(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair);
  /**
   * @brief set the max size of zmalloc DMA
   * @param sz  size in bytes
   */
  void setDmaSize(int64_t sz);
  /**
   * @brief Allocates a new I/O queue pair.
   * @return Pointer to the allocated queue pair.
   */
  struct spdk_nvme_qpair *allocQpair();
  /**
   * * @brief To setup the environment
   */
  void setupEnv();
  /**
   * * @brief To clean the environment
   */
  void cleanEnv();
  /**
 * @brief Get the size of a sector
 * @return size in byte
 */
  size_t getSectorSize(void);
  /**
 * @brief Get the size of a name space
 * @return size in byte
 */
  size_t getNameSpaceSize(void);

  /**
   * @brief  Get the size of huge page
   * @return the size in bytes
   */
  size_t getHugePageSize(void);
 private:
  std::mutex m_mut;
  struct spdk_nvme_ctrlr *ctrlr; ///< Pointer to the NVMe controller.
  struct spdk_nvme_ns *ns; ///< Pointer to the NVMe namespace.
  int64_t maxDma = 1024000;
  size_t sector_size; ///< Sector size of the NVMe namespace.
  size_t ns_size; ///< Size of the NVMe namespace.
  bool isSet = false;
  /**
   * @brief Writes data to the SSD.
   * @param buffer Data buffer to write.
   * @param size Size of the data buffer.
   * @param offset Offset in the SSD where data should be written.
   * @param qpair Queue pair to use for the write operation.
   */
  void writeInline(const void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair);

  /**
   * @brief Reads data from the SSD.
   * @param buffer Data buffer to read into.
   * @param size Size of the data buffer.
   * @param offset Offset in the SSD from where data should be read.
   * @param qpair Queue pair to use for the read operation.
   */
  void readInline(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair);
  /**
   * @brief Completion callback for read operations.
   * @param arg User argument.
   * @param completion Completion status.
   */
  static void read_complete(void *arg, const struct spdk_nvme_cpl *completion);

  /**
   * @brief Completion callback for write operations.
   * @param arg User argument.
   * @param completion Completion status.
   */
  static void write_complete(void *arg, const struct spdk_nvme_cpl *completion);
};
} // CANDY
/**
 * @}
 */
#endif //CANDY_INCLUDE_CANDY_FLATSSDGPUINDEX_SPDKSSD_H_
#endif