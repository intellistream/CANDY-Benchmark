//
// Created by tony on 19/07/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <Utils/ConfigMap.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <spdk/env.h>
#include <spdk/nvme.h>
#include <spdk/stdinc.h>
#include <CANDY/FlatSSDGPUIndex/SPDKSSD.h>
#include <iostream>
#include <thread>
#include <Utils/IntelliLog.h>
using namespace std;

/**
 * @brief Thread function to perform write operations.
 * @param ssd Reference to the SPDKSSD object.
 * @param data Data buffer to write.
 * @param offset Offset in the SSD where data should be written.
 */
void thread_write_pack(CANDY::SPDKSSD& ssd, const std::vector<uint8_t>& data, uint64_t offset) {
  struct spdk_nvme_qpair* qpair = ssd.allocQpair();
  ssd.write(data.data(), data.size(), offset, qpair);
  spdk_nvme_ctrlr_free_io_qpair(qpair);
}
using namespace CANDY;
int smain() {
  SPDKSSD ssd; // Change to your NVMe device's PCIe address
  try {
    ssd.setupEnv();
    INTELLI_INFO("Sector size="+std::to_string(ssd.getSectorSize())+", name space size="+std::to_string(ssd.getNameSpaceSize()));
    const size_t data_size = 10000;
    std::vector<uint8_t> data1(data_size, 0xAA);
    std::vector<uint8_t> data2(data_size, 0xBB);

    std::thread t1(thread_write_pack, std::ref(ssd), std::ref(data1), 512);
    std::thread t2(thread_write_pack, std::ref(ssd), std::ref(data2), 512 + data_size);

    t1.join();
    t2.join();

    std::vector<uint8_t> read_data1(data_size);
    std::vector<uint8_t> read_data2(data_size);

    struct spdk_nvme_qpair* qpair = ssd.allocQpair();

    ssd.read(read_data1.data(), read_data1.size(), 512, qpair);
    ssd.read(read_data2.data(), read_data2.size(), 512 + data_size, qpair);

    spdk_nvme_ctrlr_free_io_qpair(qpair);

    for (size_t i = 0; i < read_data1.size(); ++i) {
      if (read_data1[i] != 0xAA) {
        std::cerr << "Data mismatch at byte " << i << " in read_data1" << std::endl;
        return 1;
      }
    }

    for (size_t i = 0; i < read_data2.size(); ++i) {
      if (read_data2[i] != 0xBB) {
        std::cerr << "Data mismatch at byte " << i << " in read_data2" << std::endl;
        return 1;
      }
    }

    std::cout << "Data verification successful" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  ssd.cleanEnv();
  return 0;
}
TEST_CASE("Test Wrapper", "[short]")
{
  int a = 0;
  // place your test here
  smain();
  REQUIRE(a == 0);
}