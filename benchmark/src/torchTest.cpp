#include <iostream>
#include <torch/torch.h>
#include <CANDY.h>
using namespace INTELLI;
int main() {
  // Create a 2D tensor
  torch::Tensor tensor = torch::randn({3, 4});
  std::cout << "Original Tensor:\n" << tensor << "\n";

  // Create a new row to append
  torch::Tensor newRow = torch::randn({3, tensor.size(1)});

  IntelliTensorOP::appendRows(&tensor, &newRow);

  std::cout << "Tensor after appending a new row:\n" << tensor << "\n";
  std::vector<int64_t> rowsToDelete = {1, 3};
  IntelliTensorOP::deleteRows(&tensor, rowsToDelete);
  std::cout << "Tensor after deleting rows " << ":\n" << tensor << std::endl;
  auto tp = newTensor(torch::randn({3, 4}));
  std::cout << "Tensor ptr before delete row:\n" << *tp << "\n";
  IntelliTensorOP::deleteRow(tp, 2);
  std::cout << "Tensor ptr after delete row:\n" << *tp << "\n";
  auto tp2 = newTensor(tp->clone());
  // Split the original tensor at the insertion row
  IntelliTensorOP::insertRows(tp, tp2, 1);
  // Concatenate the parts with the tensor to insert in between
  std::cout << "Tensor ptr after insert row:\n" << *tp << "\n";
  auto tp3 = newTensor(torch::rand({2, 4}) + 1.0);
  IntelliTensorOP::editRows(tp, tp3, 1);
  std::cout << "Tensor ptr after editing row:\n" << *tp << "\n";
  int64_t lastNNZ = tp->size(0) - 1;
  //std::cout<<lastNNZ;
  std::vector<int64_t> rowsToDelete2 = {0, 2, 1};
  std::cout << lastNNZ;
  IntelliTensorOP::deleteRowsBufferMode(tp, rowsToDelete2, &lastNNZ);
  std::cout << "Tensor ptr after deletling row 0 and 2 in bufferMode:\n" << *tp << "\n";
  auto tp4 = newTensor(torch::rand({2, 4}) + 1.0);
  for (int i = 0; i < 2; i++) {
    std::cout << lastNNZ;
    IntelliTensorOP::appendRowsBufferMode(tp, tp4, &lastNNZ);
    std::cout << "Tensor ptr after apend " + std::to_string(i) + "th in bufferMode:\n" << *tp << "\n";
  }
  IntelliTensorOP::tensorToFile(tp.get(), "db.rbt");
  torch::Tensor loadedTensor;
  IntelliTensorOP::tensorFromFile(&loadedTensor, "db.rbt");
  std::cout << "Tensor loaded from file\n" << loadedTensor << "\n";
  //std::cout<<lastNNZ;
  return 0;
}