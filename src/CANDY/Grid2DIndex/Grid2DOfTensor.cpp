//
// Created by tony on 24-11-28.
//

#include <CANDY/Grid2DIndex/Grid2DOfTensor.h>
#include <Utils/IntelliLog.h>
namespace CANDY {
void Grid2DOfTensor::init(int64_t numberOfGrids) {
  numberOfGrids_ = numberOfGrids;
  dataGrid = std::vector<std::vector<GridUnitOfTensorPtr>>(numberOfGrids_);
}
int64_t Grid2DOfTensor::insertItemToGrid(int64_t x,int64_t y,int64_t item){
  if(x>=numberOfGrids_||y>=numberOfGrids_){
    INTELLI_INFO("invalid grid");
    return -1;
  }
  auto rowGrids = dataGrid[x];
  for (auto gridXy:rowGrids){
    if(gridXy->idx_==y) {
      gridXy->data_ =  torch::cat({gridXy->data_, torch::tensor({item}, torch::kInt64)});
      //INTELLI_INFO("Hit the same grid "+ to_string(x)+":"+ to_string(y)+"Cnt= "+ to_string(gridXy->data_.size(0)));
      return 1;
    }
  }
  auto newGridUnit = newGridUnitOfTensor();
  newGridUnit->idx_=y;
  newGridUnit->data_=torch::tensor({item}, torch::kInt64);
  dataGrid[x].push_back(newGridUnit);
  INTELLI_INFO("create new grid "+ to_string(x)+","+ to_string(y));
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
torch::Tensor CANDY::Grid2DOfTensor::getApproximateIndiciesUntilK(int64_t x, int64_t y, int64_t k) {
  if (numberOfGrids_ <= 0) {
    throw std::runtime_error("Grid2DOfTensor is not initialized.");
  }

  torch::Tensor collectedIndices = torch::empty({0}, torch::kInt64); // Initialize an empty tensor for indices
  int64_t extension = 0;

  // Start scanning from the central cell (x, y)
  while (collectedIndices.size(0) < k) {
    // For the current extension, scan the new border cells only
    int64_t xMin = std::max(x - extension, int64_t(0));
    int64_t xMax = std::min(x + extension, numberOfGrids_ - 1);
    int64_t yMin = std::max(y - extension, int64_t(0));
    int64_t yMax = std::min(y + extension, numberOfGrids_ - 1);

    // Scan top and bottom borders of the square
    if (extension > 0) {
      for (int64_t i = xMin; i <= xMax; ++i) {
        // Top border (yMin)
        if (yMin < numberOfGrids_) {
          for (const auto& gridUnit : dataGrid[i]) {
            if (gridUnit->idx_ == yMin && gridUnit->data_.defined()) {
              collectedIndices = torch::cat({collectedIndices, gridUnit->data_}, 0);
              if (collectedIndices.size(0) >= k) {
                return collectedIndices;
              }
            }
          }
        }

        // Bottom border (yMax)
        if (yMax < numberOfGrids_ && yMax != yMin) { // Avoid double scan if yMax == yMin
          for (const auto& gridUnit : dataGrid[i]) {
            if (gridUnit->idx_ == yMax && gridUnit->data_.defined()) {
              collectedIndices = torch::cat({collectedIndices, gridUnit->data_}, 0);
              if (collectedIndices.size(0) >= k) {
                return collectedIndices;
              }
            }
          }
        }
      }
    }

    // Scan left and right borders of the square
    for (int64_t j = yMin; j <= yMax; ++j) {
      // Left border (xMin)
      if (xMin < numberOfGrids_) {
        for (const auto& gridUnit : dataGrid[xMin]) {
          if (gridUnit->idx_ == j && gridUnit->data_.defined()) {
            collectedIndices = torch::cat({collectedIndices, gridUnit->data_}, 0);
            if (collectedIndices.size(0) >= k) {
              return collectedIndices;
            }
          }
        }
      }

      // Right border (xMax)
      if (xMax < numberOfGrids_ && xMax != xMin) { // Avoid double scan if xMax == xMin
        for (const auto& gridUnit : dataGrid[xMax]) {
          if (gridUnit->idx_ == j && gridUnit->data_.defined()) {
            collectedIndices = torch::cat({collectedIndices, gridUnit->data_}, 0);
            if (collectedIndices.size(0) >= k) {
              return collectedIndices;
            }
          }
        }
      }
    }

    // Increment extension for the next layer
    ++extension;

    // Break if we've exceeded all possible grid cells
    if (extension > numberOfGrids_) {
      break;
    }
  }

  // Ensure we return at least `k` elements or all available indices
  return collectedIndices;
}

} // CANDY