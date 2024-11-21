//
// Created by tony on 24-11-21.
//
#include <CANDY/GravistarIndex/Gravistar.h>
namespace CANDY {
void Gravistar_DarkMatter::init(int64_t vecDim,
                                int64_t bufferSize,
                                int64_t _tensorBegin,
                                int64_t _u64Begin,
                                int64_t _dmaSize) {
  dmBuffer.init(vecDim,bufferSize,_tensorBegin,_u64Begin,_dmaSize);
}

}