//
// Created by tony on 19/03/24.
//

#include <CANDY/CANDYObject.h>

namespace CANDY {
void CANDYObject::setStr(std::string str) {
  objStr = str;
  objSize = str.size();
}
std::string CANDYObject::getStr() {
  return objStr;
}
void CANDYBoundObject::setTensor(torch::Tensor &ts) {
  boundTensor=ts;
}
torch::Tensor CANDYBoundObject::getTensor() {
  return boundTensor;
}

} // CANDY