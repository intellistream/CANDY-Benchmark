/*! \file CANDYObject.h*/
//
// Created by tony on 19/03/24.
//

#ifndef CANDY_INCLUDE_CANDYOBJECT_H_
#define CANDY_INCLUDE_CANDYOBJECT_H_
#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>
namespace CANDY {
/**
 * @ingroup  CANDY_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class CANDYObject CANDY/RAMIAObject.h
 * @brief A generic object class to link string or void * pointers
 * @todo to finish the functions of setting void * pointers
 */
class CANDYObject {
 public:
  CANDYObject() {}
  ~CANDYObject() {}
  std::string objStr;
  void *objPointer = nullptr;
  int64_t objSize = 0;
  int64_t objId = -1;
  /**
   * @brief to set the string
   * @param str the string
   * @return void
   */
  void setStr(std::string str);
  /**
   * @brief to get the string
   * @return the objStr
   */
  std::string getStr();
};
/**
 * CANDYBoundObject CANDY/RAMIAObject.h
 * @brief An object with bound tensor
 */
class CANDYBoundObject: public CANDYObject{
 public:
  CANDYBoundObject(){}
  ~CANDYBoundObject(){}
  torch::Tensor boundTensor;
  /**
  * @brief to set the bound tensor
  * @param ts the tensor
  * @return void
  */
  void setTensor( torch::Tensor &ts);
  /**
   * @brief to get the tensor
   * @return the tensor
   */
  torch::Tensor getTensor();
};
/**
 * @ingroup  CANDY_lib_bottom
 * @typedef CANDYObjectPtr
 * @brief The class to describe a shared pointer to @ref  CANDYObject

 */
typedef std::shared_ptr<class CANDY::CANDYObject> CANDYObjectPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newAbstractIndex
 * @brief (Macro) To creat a new @ref  CANDYObject shared pointer.
 */
#define newCANDYObject std::make_shared<CANDY::CANDYObject>
/**
 * @}
 */
} // CANDY

#endif //CANDY_INCLUDE_CANDYOBJECT_H_
