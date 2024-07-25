//
// Created by tony on 23/07/24.
//

#include <CANDY/FlatSSDGPUIndex/SPDKSSD.h>
//#include <Utils/IntelliLog.h>
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

size_t SPDKSSD::getNameSpaceSize() {
  return ns_size;
}
size_t SPDKSSD::getSectorSize() {
  return sector_size;
}
void  SPDKSSD::setupEnv() {
  if(isSet) {
    return;
  }
  isSet = true;
  struct spdk_env_opts opts;
  spdk_env_opts_init(&opts);
  opts.name = "spdk_example";
  if (spdk_env_init(&opts) < 0) {
    fprintf(stderr, "Unable to initialize SPDK environment\n");
    return ;
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
 // INTELLI_INFO("Sector size="+std::to_string(sector_size)+"name space size="+std::to_string(ns_size));
 // exit(-1);
}
void SPDKSSD::cleanEnv() {
  spdk_nvme_detach(ctrlr);
  spdk_env_fini();
}
struct spdk_nvme_qpair* SPDKSSD::allocQpair() {
  while (!m_mut.try_lock());
  struct spdk_nvme_qpair* qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, nullptr, 0);
  if (!qpair) {
    throw std::runtime_error("Failed to allocate I/O queue pair");
  }
  m_mut.unlock();
  return qpair;
}
void SPDKSSD::write(const void* buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair* qpair) {
  if (offset + size > ns_size) {
    throw std::out_of_range("Write exceeds namespace size");
  }

 // uint64_t sector_offset = offset / sector_size;
  //size_t sector_aligned_size = ((offset + size + sector_size - 1) / sector_size) * sector_size;
 /*void* temp_buffer = spdk_zmalloc(sector_aligned_size, 0,  NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  if (!temp_buffer) {
    throw std::runtime_error("Failed to allocate temporary buffer");
  }*/
  size_t max_chunk_size = 128 * 1024 * 1024; // 128MB chunk size
  size_t remaining_size = size;
  const uint8_t* current_buffer = static_cast<const uint8_t*>(buffer);
  uint64_t current_offset = offset;

  while (remaining_size > 0) {
    size_t chunk_size = std::min(remaining_size, max_chunk_size);
    uint64_t sector_offset = current_offset / sector_size;
    size_t sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;
    void* temp_buffer = spdk_zmalloc(sector_aligned_size, 0,  NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    if (!temp_buffer) {
      throw std::runtime_error("Failed to allocate temporary buffer");
    }

    // Read existing data if not aligned
    if (current_offset % sector_size != 0 || chunk_size % sector_size != 0) {
      read(temp_buffer, sector_aligned_size, sector_offset * sector_size, qpair);
    }

    // Copy user data into the temp buffer
    std::memcpy(static_cast<uint8_t*>(temp_buffer) + (current_offset % sector_size), current_buffer, chunk_size);

    int rc = spdk_nvme_ns_cmd_write(ns, qpair, temp_buffer, sector_offset, sector_aligned_size / sector_size, write_complete, nullptr, 0);
    if (rc != 0) {
      spdk_free(temp_buffer);
      throw std::runtime_error("Write command failed");
    }

    while (!spdk_nvme_qpair_process_completions(qpair, 0));

    spdk_free(temp_buffer);(temp_buffer);

    remaining_size -= chunk_size;
    current_buffer += chunk_size;
    current_offset += chunk_size;
  }



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

void SPDKSSD::read(void* buffer, size_t size, uint64_t offset, struct spdk_nvme_qpair* qpair) {
  if (offset + size > ns_size) {
    throw std::out_of_range("Read exceeds namespace size");
  }
  size_t max_chunk_size = 128 * 1024 * 1024; // 128MB chunk size
  size_t remaining_size = size;
  uint8_t* current_buffer = static_cast<uint8_t*>(buffer);
  uint64_t current_offset = offset;

  while (remaining_size > 0) {
    size_t chunk_size = std::min(remaining_size, max_chunk_size);
    uint64_t sector_offset = current_offset / sector_size;
    size_t sector_aligned_size = ((chunk_size + sector_size - 1) / sector_size) * sector_size;
    void* temp_buffer = spdk_zmalloc(sector_aligned_size, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    if (!temp_buffer) {
      throw std::runtime_error("Failed to allocate temporary buffer");
    }

    int rc = spdk_nvme_ns_cmd_read(ns, qpair, temp_buffer, sector_offset, sector_aligned_size / sector_size, read_complete, nullptr, 0);
    if (rc != 0) {
      spdk_free(temp_buffer);
      throw std::runtime_error("Read command failed");
    }

    while (!spdk_nvme_qpair_process_completions(qpair, 0));

    std::memcpy(current_buffer, static_cast<uint8_t*>(temp_buffer) + (current_offset % sector_size), chunk_size);

    spdk_free(temp_buffer);

    remaining_size -= chunk_size;
    current_buffer += chunk_size;
    current_offset += chunk_size;
  }
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

void SPDKSSD::read_complete(void* arg, const struct spdk_nvme_cpl* completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    throw std::runtime_error("Read operation failed");
  }
}

void SPDKSSD::write_complete(void* arg, const struct spdk_nvme_cpl* completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    throw std::runtime_error("Write operation failed");
  }
}

} // CANDY
#endif