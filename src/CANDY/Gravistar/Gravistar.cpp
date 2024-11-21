//
// Created by tony on 24-11-21.
//
#include <CANDY/GravistarIndex/Gravistar.h>
namespace CANDY {
void Gravistar_DarkMatter::init(int64_t _vecDim,
                                int64_t _bufferSize,
                                int64_t _tensorBegin,
                                int64_t _u64Begin,
                                int64_t _dmaSize) {
  dmBuffer.init(_vecDim,_bufferSize,_tensorBegin,_u64Begin,_dmaSize);
  bufferSize = _bufferSize;
  vecDim = _vecDim;
}

bool CANDY::Gravistar_DarkMatter::insertTensor(torch::Tensor &t) {
  if (t.size(0)+dmBuffer.size() > bufferSize) {
    return false;
  } else {
    return dmBuffer.appendTensor(t);
  }
}
torch::Tensor CANDY::Gravistar_DarkMatter::getTensor(int64_t startPos, int64_t endPos) {
  return dmBuffer.getTensor(startPos,endPos);
}
void CANDY::Gravistar_DarkMatter::setToLastTier(bool val) {
  lastTier = val;
}
bool  CANDY::Gravistar_DarkMatter::isLastTier() {
  return  lastTier;
}


}