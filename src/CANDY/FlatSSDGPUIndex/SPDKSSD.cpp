//
// Created by tony on 23/07/24.
//

#include <CANDY/FlatSSDGPUIndex/SPDKSSD.h>
//#include <Utils/IntelliLog.h>
#include <fstream>
#include <spdk/env.h>
#include <sys/statvfs.h>
#include <regex>
#if CANDY_SPDK == 1
namespace CANDY {
spdk_nvme_ctrlr *g_spdk_ctrlr = nullptr;
static bool probe_cb_auto(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts) {
  printf("Attaching to %s\n", trid->traddr);
  return true;
}

static void attach_cb_auto(void *cb_ctx,
                           const struct spdk_nvme_transport_id *trid,
                           struct spdk_nvme_ctrlr *ctrlr0,
                           const struct spdk_nvme_ctrlr_opts *opts) {
  g_spdk_ctrlr = ctrlr0;
}
SPDKSSD::SPDKSSD() {

}

SPDKSSD::~SPDKSSD() {
  //spdk_nvme_detach(ctrlr);
}
void SPDKSSD::setDmaSize(int64_t sz) {
  maxDma = sz;
}
size_t SPDKSSD::getNameSpaceSize() {
  return ns_size;
}
size_t SPDKSSD::getSectorSize() {
  return sector_size;
}
int64_t SPDKSSD::getTotalDiskWrite() {
  return totalDiskWrite;
}
int64_t SPDKSSD::getTotalUserWrite() {
  return totalUserWrite;
}

int64_t SPDKSSD::getTotalDiskRead() {
  return totalDiskRead;
}
int64_t SPDKSSD::getTotalUserRead() {
  return totalUserRead;
}

size_t SPDKSSD::getHugePageSize() {
  // size_t max_alloc_size = 0;
  std::ifstream infile("/proc/meminfo");
  std::string line;
  std::regex hugepages_size_regex("Hugepagesize:\\s+(\\d+) kB");
  // size_t hugepages_free = 0;
  size_t hugepages_size_kb = 0;

  while (std::getline(infile, line)) {
    std::smatch match;
    if (std::regex_search(line, match, hugepages_size_regex)) {
      hugepages_size_kb = std::stoul(match[1].str());
    }
  }

  return hugepages_size_kb * 1000;
}
size_t SPDKSSD::getFreeHugePages() {

  std::ifstream infile("/proc/meminfo");
  std::string line;
  std::regex hugepages_regex("HugePages_Free:\\s+(\\d+)");
  size_t hugepages_free = 0;

  while (std::getline(infile, line)) {
    std::smatch match;
    if (std::regex_search(line, match, hugepages_regex)) {
      hugepages_free = std::stoul(match[1].str());
    }
  }
  return hugepages_free;
}
void SPDKSSD::clearStatistics() {
  totalDiskWrite = 0;
  totalDiskRead = 0;
  totalUserRead = 0;
  totalUserWrite = 0;
}
void SPDKSSD::setupEnv() {
  if (isSet) {
    return;
  }
  isSet = true;
  struct spdk_env_opts opts;
  spdk_env_opts_init(&opts);
  opts.name = "spdk_example";
  if (spdk_env_init(&opts) < 0) {
    fprintf(stderr, "Unable to initialize SPDK environment\n");
    return;
  }
  if (spdk_nvme_probe(NULL, NULL, probe_cb_auto, attach_cb_auto, NULL) != 0) {
    fprintf(stderr, "spdk_nvme_probe() failed\n");
    spdk_env_fini();
  }
  ctrlr = g_spdk_ctrlr;
  if (ctrlr == NULL) {
    fprintf(stderr, "No NVMe controllers found\n");
    spdk_env_fini();
  }

  ns = spdk_nvme_ctrlr_get_ns(ctrlr, spdk_nvme_ctrlr_get_first_active_ns(ctrlr));
  if (ns == NULL) {
    fprintf(stderr, "Namespace not found\n");
    spdk_nvme_detach(ctrlr);
    spdk_env_fini();
  }
  sector_size = spdk_nvme_ns_get_sector_size(ns);
  ns_size = spdk_nvme_ns_get_size(ns);
  setDmaSize(getHugePageSize() / 4);
  // std::cout<<"Free huge pages"<<getFreeHugePages()<<std::endl;
  //INTELLI_INFO("Sector size="+std::to_string(sector_size)+"name space size="+std::to_string(ns_size));
  // exit(-1);
}
void SPDKSSD::cleanEnv() {
  spdk_nvme_detach(ctrlr);
  spdk_env_fini();
}
struct spdk_nvme_qpair *SPDKSSD::allocQpair() {
  while (!m_mut.try_lock());
  struct spdk_nvme_qpair *qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, nullptr, 0);
  if (!qpair) {
    throw std::runtime_error("Failed to allocate I/O queue pair");
  }
  m_mut.unlock();
  return qpair;
}
void SPDKSSD::write(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair) {
  /*int64_t remainSize = size;
  uint64_t tempOffset = offset;
  uint64_t bufferPos = 0;
  uint8_t *tempBuffer = (uint8_t *)buffer;
  while (remainSize>0) {
    int64_t writeSize = std::min(remainSize,maxDma);
    writeInline(&tempBuffer[bufferPos],writeSize,tempOffset,qpair);
    tempOffset += writeSize;
    bufferPos += writeSize;
    remainSize -= writeSize;
  }*/
  totalUserWrite += size;
  writeInline(buffer, size, offset, qpair);
}
void SPDKSSD::read(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair) {
  /*int64_t remainSize = size;
  uint64_t tempOffset = offset;
  uint64_t bufferPos = 0;
  uint8_t *tempBuffer = (uint8_t *)buffer;
  while (remainSize>0) {
    int64_t readSize = std::min(remainSize,maxDma);
    readInline(&tempBuffer[bufferPos],readSize,tempOffset,qpair);
    tempOffset += readSize;
    bufferPos += readSize;
    remainSize -= readSize;
  }*/
  totalUserRead += size;
  readInline(buffer, size, offset, qpair);
}
void SPDKSSD::writeInline(const void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair) {
  if (offset + size > ns_size) {
    throw std::out_of_range("Write exceeds namespace size");
  }

  // uint64_t sector_offset = offset / sector_size;
  //size_t sector_aligned_size = ((offset + size + sector_size - 1) / sector_size) * sector_size;
  /*void* temp_buffer = spdk_zmalloc(sector_aligned_size, 0,  NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
   if (!temp_buffer) {
     throw std::runtime_error("Failed to allocate temporary buffer");
   }*/
  size_t max_chunk_size = maxDma;
  size_t remaining_size = size;
  const uint8_t *current_buffer = static_cast<const uint8_t *>(buffer);
  uint64_t current_offset = offset;
  size_t chunk_size = std::min(remaining_size, max_chunk_size);
  uint64_t sector_offset = current_offset / sector_size;
  size_t sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;
  void *temp_buffer = spdk_zmalloc(sector_aligned_size + sector_size, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  while (remaining_size > 0) {
    chunk_size = std::min(remaining_size, max_chunk_size);
    sector_offset = current_offset / sector_size;
    sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;

    if (!temp_buffer) {
      throw std::runtime_error("Failed to allocate temporary buffer");
    }

    // Read existing data if not aligned
    if (current_offset % sector_size != 0 || chunk_size % sector_size != 0) {
      read(temp_buffer, sector_aligned_size + sector_size, (sector_offset) * sector_size, qpair);
    }
    size_t insideCopyOffset = current_offset % sector_size;
    // Copy user data into the temp buffer

    std::memcpy(static_cast<uint8_t *>(temp_buffer) + (insideCopyOffset), current_buffer, chunk_size);
    size_t lbaCnt = sector_aligned_size / sector_size;
    if (insideCopyOffset) {
      lbaCnt += 1;
    }
    int rc = spdk_nvme_ns_cmd_write(ns,
                                    qpair,
                                    temp_buffer,
                                    sector_offset,
                                    lbaCnt,
                                    write_complete,
                                    nullptr,
                                    0);
    while (rc != 0) {
      rc = spdk_nvme_ns_cmd_write(ns,
                                  qpair,
                                  temp_buffer,
                                  sector_offset,
                                  lbaCnt,
                                  write_complete,
                                  nullptr,
                                  0);
    }
    totalDiskWrite += sector_aligned_size;
    while (!spdk_nvme_qpair_process_completions(qpair, 0));

    remaining_size -= chunk_size;
    current_buffer += chunk_size;
    current_offset += chunk_size;
  }
  spdk_free(temp_buffer);


  // Read existing data if not aligned
  /*if (offset % sector_size != 0 || size % sector_size != 0) {
    read(temp_buffer, sector_aligned_size, sector_offset * sector_size, qpair);
  }

  // Copy user data into the temp buffer
  std::memcpy(static_cast<uint8_t*>(temp_buffer) + (offset % sector_size), buffer, size);

  int rc = spdk_nvme_ns_cmd_write(ns, qpair, temp_buffer, sector_offset, sector_aligned_size / sector_size, write_complete, nullptr, 0);
 // int rc = spdk_nvme_ns_cmd_write(ns, qpair, temp_buffer, 0, 1, write_complete, nullptr, 0);
  if (rc != 0) {
    spdk_free(temp_buffer);
    throw std::runtime_error("Write command failed");
  }

  while (!spdk_nvme_qpair_process_completions(qpair, 0));

  spdk_free(temp_buffer);*/
}

void SPDKSSD::readInline(void *buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair *qpair) {
  if (offset + size > ns_size) {
    throw std::out_of_range("Read exceeds namespace size");
  }
  size_t max_chunk_size = maxDma;
  size_t remaining_size = size;
  uint8_t *current_buffer = static_cast<uint8_t *>(buffer);
  uint64_t current_offset = offset;
  size_t chunk_size = std::min(remaining_size, max_chunk_size);
  uint64_t sector_offset = current_offset / sector_size;
  size_t sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;
  void *temp_buffer = spdk_zmalloc(sector_aligned_size + sector_size, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  while (remaining_size > 0) {
    chunk_size = std::min(remaining_size, max_chunk_size);
    sector_offset = current_offset / sector_size;
    sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;
    if (!temp_buffer) {
      throw std::runtime_error("Failed to allocate temporary buffer");
    }
    size_t insideCopyOffset = current_offset % sector_size;
    size_t lbaCnt = sector_aligned_size / sector_size;
    if (insideCopyOffset) {
      lbaCnt += 1;
    }
    int rc = spdk_nvme_ns_cmd_read(ns,
                                   qpair,
                                   temp_buffer,
                                   sector_offset,
                                   lbaCnt,
                                   read_complete,
                                   nullptr,
                                   0);
    while (rc != 0) {
      rc = spdk_nvme_ns_cmd_read(ns,
                                 qpair,
                                 temp_buffer,
                                 sector_offset,
                                 lbaCnt,
                                 read_complete,
                                 nullptr,
                                 0);
    }
    totalDiskRead += sector_aligned_size;
    while (!spdk_nvme_qpair_process_completions(qpair, 0));

    uint8_t *copyDest = (uint8_t *) temp_buffer;
    std::memcpy(current_buffer, &copyDest[insideCopyOffset], chunk_size);

    remaining_size -= chunk_size;
    current_buffer += chunk_size;
    current_offset += chunk_size;
  }
  spdk_free(temp_buffer);
  /*uint64_t sector_offset = offset / sector_size;
  size_t sector_aligned_size = ((offset + size + sector_size - 1) / sector_size) * sector_size;
  void* temp_buffer = spdk_zmalloc(sector_aligned_size, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  if (!temp_buffer) {
    throw std::runtime_error("Failed to allocate temporary buffer");
  }

  int rc = spdk_nvme_ns_cmd_read(ns, qpair, temp_buffer, sector_offset, sector_aligned_size / sector_size, read_complete, nullptr, 0);
  //int rc = spdk_nvme_ns_cmd_write(ns, qpair, temp_buffer, 0, 1, read_complete, nullptr, 0);
  if (rc != 0) {
    spdk_free(temp_buffer);
    throw std::runtime_error("Read command failed");
  }

  while (!spdk_nvme_qpair_process_completions(qpair, 0));

  std::memcpy(buffer, static_cast<uint8_t*>(temp_buffer) + (offset % sector_size), size);

  spdk_free(temp_buffer);*/
}

void SPDKSSD::read_complete(void *arg, const struct spdk_nvme_cpl *completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    throw std::runtime_error("Read operation failed");
  }
}

void SPDKSSD::write_complete(void *arg, const struct spdk_nvme_cpl *completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    throw std::runtime_error("Write operation failed");
  }
}

} // CANDY
#endif