//
// Created by tony on 24-11-28.
//

#include <CANDY/Grid2DIndex/Grid2DOfTensor.h>

namespace CANDY {
void Grid2DOfTensor::init(int64_t numberOfGrids) {
  numberOfGrids_ = numberOfGrids;
  dataGrid = std::vector<std::vector<GridUnitOfTensorPtr>>(numberOfGrids_);
}
int64_t Grid2DOfTensor::insertItemToGrid(int64_t x,int64_t y,int64_t item){
  if(x>=numberOfGrids_||y>=numberOfGrids_){
    return -1;
  }
  auto rowGrids = dataGrid[x];
  for (auto gridXy:rowGrids){
    if(gridXy->idx_==y) {
      gridXy->data_ =  torch::cat({gridXy->data_, torch::tensor({item}, torch::kInt64)});
      return 1;
    }
  }
  auto newGridUnit = newGridUnitOfTensor();
  newGridUnit->idx_=y;
  newGridUnit->data_=torch::tensor({item}, torch::kInt64);
  rowGrids.push_back(newGridUnit);
  return 1;
}

GridUnitOfTensorPtr Grid2DOfTensor::getExactGridUnit(int64_t x, int64_t y) {
  if(x>=numberOfGrids_||y>=numberOfGrids_){
    return nullptr;
  }
  auto rowGrids = dataGrid[x];
  for (auto gridXy:rowGrids){
    if(gridXy->idx_==y) {
      return gridXy;
    }
  }
  return nullptr;
}
std::vector<GridUnitOfTensorPtr> Grid2DOfTensor::getApproximateGridUnitsSquare(int64_t x,int64_t y,int64_t extension,int64_t skipExact){
  std::vector<GridUnitOfTensorPtr> results;
  int64_t xMin = std::max(x-extension,(int64_t)0);
  int64_t xMax = std::min(x+extension,(int64_t)numberOfGrids_);
  int64_t yMin = std::max(y-extension,(int64_t)0);
  int64_t yMax = std::min(y+extension,(int64_t)numberOfGrids_);
  for (int64_t tx=xMin;tx<xMax;tx++){
    auto rowGrids = dataGrid[tx];
    for (auto gridXy:rowGrids){
      /**
       * @brief skip logic of [x,y]
       */
      if(gridXy->idx_==y&&tx==x) {
       if(!skipExact) {
         results.push_back(gridXy);
       }
      }
      else if(yMin<=gridXy->idx_&&gridXy->idx_<yMax) {
        results.push_back(gridXy);
      }
    }
  }
  return results;
}

} // CANDY