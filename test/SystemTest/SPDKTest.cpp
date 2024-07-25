//
// Created by tony on 19/07/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <iostream>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
#include <Utils/ConfigMap.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <spdk/env.h>
#include <spdk/nvme.h>
#include <spdk/stdinc.h>
#include <CANDY/FlatSSDGPUIndex/SPDKSSD.h>
#include <CANDY/FlatSSDGPUIndex/DiskMemBuffer.h>
#include <iostream>
#include <thread>
#include <Utils/IntelliLog.h>
#include <CANDY/FlatSSDGPUIndex.h>
#include <CANDY/FlatAMMIPIndex.h>
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
/*
TEST_CASE("Test tensor RW", "[short]")
{
  SPDKSSD ssd;
  int temp = 0;
  PlainDiskMemBufferTU dmBuffer;
  ssd.setupEnv();
  dmBuffer.init(4,2,0,512,&ssd);
  torch::manual_seed(114514);
  auto a = torch::rand({4,4});
  INTELLI_INFO("Here is tensor");
  std::cout<<a<<std::endl;
  auto a02=a.slice(0,0,2);
  auto a24=a.slice(0,2,4);
  dmBuffer.appendTensor(a02);
  dmBuffer.appendTensor(a24);
  auto get02 = dmBuffer.getTensor(0,2);
  INTELLI_INFO("Here is row 0 to 2");
  std::cout<<get02<<std::endl;
  auto get13=dmBuffer.getTensor(1,3);
  INTELLI_INFO("Here is row 1 to 3");
  std::cout<<get13<<std::endl;
  auto get24=dmBuffer.getTensor(2,4);
  INTELLI_INFO("Here is row 2 to 4");
  std::cout<<get24<<std::endl;
  INTELLI_INFO("Now delete row 1");
  dmBuffer.deleteTensor(1,2);
  INTELLI_INFO("done");
  get13 = dmBuffer.getTensor(1,3);
  INTELLI_INFO("Here is row 1 to 3 after delete");
  std::cout<<get13<<std::endl;
  REQUIRE(temp == 0);
}*/
TEST_CASE("Test tensor RW", "[short]")
{
  SPDKSSD ssd;
  int temp = 0;
  torch::manual_seed(7758258);
  auto a = torch::rand({4,4});
  auto a02=a.slice(0,0,2);
  auto a24=a.slice(0,2,4);
  auto a34=a.slice(0,3,4);
  INTELLI_INFO("Here is tensor");
  std::cout<<a<<std::endl;
  INTELLI::ConfigMapPtr cfg = newConfigMap();
  cfg->edit("vecDim", (int64_t) 4);
  cfg->edit("metricType", "IP");
  cfg->edit("SSDBufferSize",(int64_t)2);
  auto ssdIdx = newFlatSSDGPUIndex();
  auto flatIdx = newFlatAMMIPIndex();
  ssdIdx->setConfig(cfg);
  ssdIdx->startHPC();
  ssdIdx->insertTensor(a02);
  ssdIdx->insertTensor(a24);
  auto ru =ssdIdx->searchTensor(a34,2);
  INTELLI_INFO("Here is search on row 3");
  std::cout<<ru[0]<<std::endl;
  ssdIdx->endHPC();
  INTELLI_INFO("Validate ...");
  flatIdx->setConfig(cfg);
  flatIdx->insertTensor(a02);
  flatIdx->insertTensor(a24);
  ru =flatIdx->searchTensor(a34,2);
  INTELLI_INFO("Here is search on row 3, using faiss");
  std::cout<<ru[0]<<std::endl;


  REQUIRE(temp == 0);
}