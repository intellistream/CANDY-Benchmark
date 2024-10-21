//
// Created by tony on 10/05/23.
//

#include <DataLoader/RBTDataLoader.h>
#include <Utils/IntelliTensorOP.hpp>
#include <iostream>
#include <fstream>
#include <Utils/IntelliLog.h>
//do nothing in Random class

bool CANDY::RBTDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  vecVolume = cfg->tryI64("vecVolume", 1000, true);
  querySize = cfg->tryI64("querySize", 500, true);
  queryPath = cfg->tryString("queryPath", "datasets/rbt/example.rbt", true);
  dataPath = cfg->tryString("dataPath", "datasets/rbt/exampleQ.rbt", true);
  useSeparateQuery = cfg->tryI64("useSeparateQuery", 1, true);
  if(!useSeparateQuery) {
    queryPath = dataPath;
  }
  dataSizes = CANDY::RBTDataLoader::getSizesFromRBT(dataPath);
  if(vecDim!=dataSizes[1]) {
    INTELLI_ERROR("Miss matched dimension, expect "+ to_string(vecDim)+", but get"+to_string(dataSizes[1]));
  }
  if(vecVolume>dataSizes[0]) {
    INTELLI_ERROR("Don not have enough rows");
  }
  return true;
}
torch::Tensor CANDY::RBTDataLoader::getQuery() {
  return CANDY::RBTDataLoader::readRowsFromRBT(queryPath,0,querySize);
}
torch::Tensor CANDY::RBTDataLoader::getDataAt(int64_t startPos, int64_t endPos) {
  return CANDY::RBTDataLoader::readRowsFromRBT(dataPath,startPos,endPos);
}
torch::Tensor CANDY::RBTDataLoader::getData() {
  return CANDY::RBTDataLoader::readRowsFromRBT(dataPath,0,std::min(vecVolume,(int64_t)1000));
}

int64_t CANDY::RBTDataLoader::size() {
  return dataSizes[0];
}
int64_t CANDY::RBTDataLoader::getDimension() {
  return dataSizes[1];
}

int64_t CANDY::RBTDataLoader::createRBT(std::string fname,torch::Tensor &t){
  if(INTELLI::IntelliTensorOP::tensorToFile(&t,fname)) {
    return true;
  }
  return false;
}
int64_t CANDY::RBTDataLoader::appendTensorToRBT(std::string fname, torch::Tensor &t) {
  std::fstream file(fname, std::ios::binary | std::ios::in | std::ios::out);
  if (!file) {
    return -1;
  }
  // Step 1: Read the header (first two int64_t values)
  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if(t.size(1)!= cols) {
    file.close();
    return -2;
  }
  //std::cout << "Original rows: " << rows << ", columns: " << cols << std::endl;

  // Step 2: Append the tensor to the end of the file
  file.seekp(0, std::ios::end);
  auto data = t.data_ptr<float>();
  auto num_elements = t.numel();
  file.write(reinterpret_cast<char*>(data), num_elements * sizeof(float));

  // Step 3: Update the header with the new number of rows
  int64_t new_rows = rows + t.size(0);
  file.seekp(0, std::ios::beg);
  file.write(reinterpret_cast<char*>(&new_rows), sizeof(new_rows));

  //std::cout << "Updated rows: " << new_rows << ", columns: " << cols << std::endl;

  file.close();
  return 0;
}
torch::Tensor CANDY::RBTDataLoader::readRowsFromRBT(std::string fname, int64_t startPos, int64_t endPos) {
  std::ifstream file(fname, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  // Read the header to get the rows and columns
  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

  if (startPos < 0 || endPos > rows || startPos > endPos) {
    throw std::out_of_range("Invalid row range");
  }

  // Calculate the number of rows to read and the starting position
  int64_t num_rows =  endPos-startPos;
  std::streampos fileOffset = sizeof(rows) + sizeof(cols) + startPos * cols * sizeof(float);
  // Move to the starting position
  file.seekg(fileOffset);
  torch::Tensor ru = torch::zeros({num_rows,cols}).contiguous();
  // Read the data into a tensor
  file.read(reinterpret_cast<char*>(ru.data_ptr()), ru.numel() * sizeof(float));
  file.close();
  return  ru;
}
std::vector<int64_t> CANDY::RBTDataLoader::getSizesFromRBT(std::string fname) {
  std::ifstream file(fname, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }
  // Read the header to get the rows and columns
  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.close();
  std::vector<int64_t> ru(2);
  ru[0] = rows;
  ru [1] = cols;
  return ru;
}