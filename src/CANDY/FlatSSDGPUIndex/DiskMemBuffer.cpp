//
// Created by tony on 24/07/24.
//
#include <CANDY/FlatSSDGPUIndex/DiskMemBuffer.h>
#if CANDY_SPDK == 1
namespace CANDY {
void PlainDiskMemBufferTU::init(int64_t vecDim,
                                int64_t bufferSize,
                                int64_t _tensorBegin,
                                int64_t _u64Begin,
                                SPDKSSD *_ssdPtr,
                                int64_t _dmaSize) {
  //diskInfo.
  diskInfo.vecDim = vecDim;
  bsize = bufferSize;
  tensorBegin = _tensorBegin;
  u64Begin = _u64Begin;
  dmaSize = _dmaSize;
  if (bufferSize > 0) {
    cacheT.buffer = torch::zeros({bufferSize, vecDim});
    cacheT.beginPos = 0;
    cacheT.endPos = bufferSize;
    cacheU.buffer = std::vector<uint64_t>(bufferSize, 0);
    cacheU.beginPos = 0;
    cacheU.endPos = bufferSize;
    //std::cout<<cacheT.buffer;
    isDirtyT = false;
    isDirtyU = false;
  }
  ssdPtr = _ssdPtr;
  ssdQpair = ssdPtr->allocQpair();

}
int64_t PlainDiskMemBufferTU::getMemoryWriteCntTotal() {
  return memoryWriteCntTotal;
}
int64_t PlainDiskMemBufferTU::getMemoryWriteCntMiss() {
  return memoryWriteCntMiss;
}
int64_t PlainDiskMemBufferTU::getMemoryReadCntTotal() {
  return memoryReadCntTotal;
}
int64_t PlainDiskMemBufferTU::getMemoryReadCntMiss() {
  return memoryReadCntMiss;
}
void PlainDiskMemBufferTU::clearStatistics() {
  memoryWriteCntTotal = 0;
  memoryWriteCntMiss = 0;
  memoryReadCntTotal = 0;
  memoryReadCntMiss = 0;
}

int64_t PlainDiskMemBufferTU::size() {
  return diskInfo.vecCnt;
}
torch::Tensor PlainDiskMemBufferTU::getTensor(int64_t startPos, int64_t endPos) {


  //int64_t remainSizeRu;
  int64_t diskOffset;
  /**
   * @brief no buffer, direct disk read
   */
  if (bsize <= 0) {
    diskOffset = startPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
    int64_t readSize = endPos - startPos;
    auto ru = torch::zeros({readSize, (int64_t) diskInfo.vecDim}).contiguous();
    ssdPtr->read(ru.data_ptr(), readSize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
    return ru;
  }
  auto inMemTensor = cacheT.buffer.contiguous();
  memoryReadCntTotal++;
  /**
  * @brief totally inside memory
  */
  if (startPos >= cacheT.beginPos && endPos <= cacheT.endPos) {
    torch::Tensor ru = inMemTensor.slice(0, startPos - cacheT.beginPos, endPos - cacheT.beginPos);
    return ru;
  }
    /**
    * @brief partially inside memory
    */
    /*else if(startPos>=cacheT.beginPos&&endPos>cacheT.endPos) {
      torch::Tensor ru =torch::zeros({endPos-startPos,(int64_t)diskInfo.vecDim});
      ru.slice(0,0,cacheT.endPos-startPos) = inMemTensor.slice(0,startPos-cacheT.beginPos,cacheT.endPos-cacheT.beginPos);
      remainSizeRu = (endPos-cacheT.endPos);
      diskOffset = cacheT.endPos*diskInfo.vecDim*sizeof(float)+512+tensorBegin;
      ssdPtr->read(inMemTensor.data_ptr(),bsize*diskInfo.vecDim*sizeof(float),diskOffset,ssdQpair);
      cacheT.buffer = inMemTensor;
      ru.slice(0,cacheT.endPos-startPos,endPos-startPos) = inMemTensor.slice(0,0,remainSizeRu);
      cacheT.beginPos = cacheT.endPos;
      cacheT.endPos += bsize;
      return  ru;
    }*/
  else {
    memoryReadCntMiss++;
    /**
    * @brief flush the old
    */
    if (isDirtyT) {
      diskOffset = cacheT.beginPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
      ssdPtr->write(inMemTensor.data_ptr(), bsize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
      isDirtyT = false;
    }
    /**
     * @brief read something new
     */
    diskOffset = startPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
    ssdPtr->read(inMemTensor.data_ptr(), bsize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
    cacheT.beginPos = startPos;
    cacheT.endPos = startPos + bsize;
    cacheT.buffer = inMemTensor;
    torch::Tensor ru = inMemTensor.slice(0, startPos - cacheT.beginPos, endPos - cacheT.beginPos);
    return ru;
  }

}
std::vector<uint64_t> PlainDiskMemBufferTU::getU64(int64_t startPos, int64_t endPos) {
  auto &inMemBuffer = cacheU.buffer;
  //int64_t remainSizeRu;
  int64_t diskOffset;
  int64_t bufferBegin = u64Begin;
  // Totally inside memory
  if (startPos >= cacheU.beginPos && endPos <= cacheU.endPos) {
    std::vector<uint64_t> ru(inMemBuffer.begin() + (startPos - cacheU.beginPos),
                             inMemBuffer.begin() + (endPos - cacheU.beginPos));
    return ru;
  }
    // Partially inside memory
    /*else if (startPos >= cacheU.beginPos && endPos > cacheU.endPos) {
      std::vector<uint64_t> ru(endPos - startPos, 0);
      std::copy(inMemBuffer.begin() + (startPos - cacheU.beginPos),
                inMemBuffer.end(),
                ru.begin());
      remainSizeRu = (endPos - cacheU.endPos);
      diskOffset = cacheU.endPos * sizeof(uint64_t) + 512 + bufferBegin;
      ssdPtr->read(reinterpret_cast<void*>(inMemBuffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);
      std::copy(inMemBuffer.begin(),
                inMemBuffer.begin() + remainSizeRu,
                ru.begin() + (cacheU.endPos - startPos));
      cacheU.buffer.assign(inMemBuffer.begin(), inMemBuffer.begin() + bsize);
      cacheU.beginPos = cacheU.endPos;
      cacheU.endPos += bsize;
      return ru;
    }*/
    // Not in memory - Flush and read new data
  else {
    if (isDirtyU) {
      diskOffset = cacheU.beginPos * sizeof(uint64_t) + 512 + bufferBegin;
      ssdPtr->write(reinterpret_cast<void *>(inMemBuffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);
      isDirtyU = false;
    }

    diskOffset = startPos * sizeof(uint64_t) + 512 + bufferBegin;
    ssdPtr->read(reinterpret_cast<void *>(inMemBuffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);
    cacheU.buffer.assign(inMemBuffer.begin(), inMemBuffer.begin() + bsize);
    cacheU.beginPos = startPos;
    cacheU.endPos = startPos + bsize;
    std::vector<uint64_t> ru(inMemBuffer.begin() + (startPos - cacheU.beginPos),
                             inMemBuffer.begin() + (endPos - cacheU.beginPos));
    return ru;
  }
}
bool PlainDiskMemBufferTU::reviseTensor(int64_t startPos, torch::Tensor &t) {
  int64_t endPos = startPos + t.size(0);
  //int64_t remainSizeRu;
  int64_t diskOffset;
  /**
   * @brief no buffer, direct disk write
   */
  if (bsize <= 0) {
    diskOffset = startPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
    int64_t writeSize = t.size(0);
    auto writeTensor = t.contiguous();
    ssdPtr->write(writeTensor.data_ptr(), writeSize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
    return true;
  }
  isDirtyT = true;
  memoryWriteCntTotal++;

  /**
 * @brief totally inside memory
 */
  if (startPos >= cacheT.beginPos && endPos <= cacheT.endPos) {
    cacheT.buffer.slice(0, startPos - cacheT.beginPos, endPos - cacheT.beginPos) = t;
    return true;
  }
    /**
     * @brief partially inside memory
     */
    /* else if(startPos>=cacheT.beginPos&&endPos>cacheT.endPos) {
       cacheT.buffer.slice(0,startPos-cacheT.beginPos,endPos-cacheT.beginPos) = t.slice(0,0,cacheT.endPos-startPos);
       auto inMemTensor = cacheT.buffer.contiguous();
       diskOffset = cacheT.beginPos*diskInfo.vecDim*sizeof(float)+512+tensorBegin;

       ssdPtr->write(inMemTensor.data_ptr(),bsize*diskInfo.vecDim*sizeof(float),diskOffset,ssdQpair);

       remainSizeRu = (endPos-cacheT.endPos);
       diskOffset = cacheT.endPos*diskInfo.vecDim*sizeof(float)+512+tensorBegin;

       ssdPtr->read(inMemTensor.data_ptr(),bsize*diskInfo.vecDim*sizeof(float),diskOffset,ssdQpair);
       inMemTensor.slice(0,0,remainSizeRu)=t.slice(0,cacheT.endPos-startPos,endPos-startPos);
       cacheT.beginPos = cacheT.endPos;
       cacheT.endPos += bsize;
       cacheT.buffer = inMemTensor;
       return true;
     }*/
  else {
    memoryWriteCntMiss++;
    diskOffset = cacheT.beginPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
    /**
     * @brief flush the old
     */
    auto inMemTensor = cacheT.buffer.contiguous();
    ssdPtr->write(inMemTensor.data_ptr(), bsize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
    /**
    * @brief read the new
    */
    diskOffset = startPos * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
    ssdPtr->read(inMemTensor.clone().data_ptr(), bsize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
    cacheT.beginPos = startPos;
    cacheT.endPos = startPos + bsize;
    cacheT.buffer = inMemTensor;
    cacheT.buffer.slice(0, startPos - cacheT.beginPos, endPos - cacheT.beginPos) = t;
    return true;
  }
}
bool PlainDiskMemBufferTU::reviseU64(int64_t startPos, std::vector<uint64_t> &data) {
  int64_t endPos = startPos + data.size();
  int64_t remainSizeRu;
  int64_t diskOffset;
  isDirtyU = true;
  int64_t bufferBegin = u64Begin;
  // Totally inside memory
  if (startPos >= cacheU.beginPos && endPos <= cacheU.endPos) {
    std::copy(data.begin(), data.end(), cacheU.buffer.begin() + (startPos - cacheU.beginPos));
    return true;
  }
    // Partially inside memory
  else if (startPos >= cacheU.beginPos && endPos > cacheU.endPos) {
    std::copy(data.begin(),
              data.begin() + (cacheU.endPos - startPos),
              cacheU.buffer.begin() + (startPos - cacheU.beginPos));

    diskOffset = cacheU.beginPos * sizeof(uint64_t) + 512 + bufferBegin;
    ssdPtr->write(reinterpret_cast<void *>(cacheU.buffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);

    remainSizeRu = (endPos - cacheU.endPos);
    diskOffset = cacheU.endPos * sizeof(uint64_t) + 512 + bufferBegin;
    ssdPtr->read(reinterpret_cast<void *>(cacheU.buffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);

    std::copy(data.begin() + (cacheU.endPos - startPos), data.end(), cacheU.buffer.begin());
    cacheU.beginPos = cacheU.endPos;
    cacheU.endPos += bsize;
    return true;
  }
    // Not in memory - Flush and read new data
  else {
    diskOffset = cacheU.beginPos * sizeof(uint64_t) + 512 + bufferBegin;
    ssdPtr->write(reinterpret_cast<void *>(cacheU.buffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);

    diskOffset = startPos * sizeof(uint64_t) + 512 + bufferBegin;
    ssdPtr->read(reinterpret_cast<void *>(cacheU.buffer.data()), bsize * sizeof(uint64_t), diskOffset, ssdQpair);

    cacheU.beginPos = startPos;
    cacheU.endPos = startPos + bsize;
    std::copy(data.begin(), data.end(), cacheU.buffer.begin() + (startPos - cacheU.beginPos));
    return true;
  }
}
bool PlainDiskMemBufferTU::appendTensor(torch::Tensor &t) {
  int64_t startPos = diskInfo.vecCnt;
  // int64_t diskOffset;
  reviseTensor(startPos, t);
  // diskOffset = tensorBegin;
  diskInfo.vecCnt += t.size(0);
  return true;
}
bool PlainDiskMemBufferTU::appendU64(std::vector<uint64_t> &data) {
  int64_t startPos = diskInfo.vecCnt;
  reviseU64(startPos, data);
  diskInfo.u64Cnt += data.size();
  return true;
}

bool PlainDiskMemBufferTU::deleteTensor(int64_t startPos, int64_t endPos) {
  int64_t delSize = endPos - startPos;
  auto inMemTensor = torch::zeros({delSize, (int64_t) diskInfo.vecDim}).contiguous();
  int64_t diskOffset;
  /***
   * @brief read the tail valid tensor
   */
  diskOffset = (diskInfo.vecCnt - delSize) * diskInfo.vecDim * sizeof(float) + 512 + tensorBegin;
  ssdPtr->read(inMemTensor.data_ptr(), delSize * diskInfo.vecDim * sizeof(float), diskOffset, ssdQpair);
  //std::cout<<"Tail tensor"<<inMemTensor;
  reviseTensor(startPos, inMemTensor);
  diskInfo.vecCnt -= delSize;
  return true;
}
bool PlainDiskMemBufferTU::deleteU64(int64_t startPos, int64_t endPos) {
  int64_t delSize = endPos - startPos;
  std::vector<uint64_t> inMemBuffer(delSize, 0);
  int64_t diskOffset;
  int64_t bufferBegin = u64Begin;
  // Read the tail valid data
  diskOffset = (diskInfo.u64Cnt - delSize) * sizeof(uint64_t) + 512 + bufferBegin;
  ssdPtr->read(reinterpret_cast<void *>(inMemBuffer.data()), delSize * sizeof(uint64_t), diskOffset, ssdQpair);

  reviseU64(startPos, inMemBuffer);
  diskInfo.vecCnt -= delSize;
  return true;
}

void PlainDiskMemBufferTU::clear() {
  ssdPtr->write(&diskInfo, sizeof(DiskHeader), tensorBegin, ssdQpair);
  //spdk_nvme_ctrlr_free_io_qpair(ssdQpair);
}
}
#endif