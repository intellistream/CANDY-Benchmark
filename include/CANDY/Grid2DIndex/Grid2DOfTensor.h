/*! \file Grid2DOfTensor.h*/
//
// Created by tony on 24-11-28.
//

#ifndef CANDYBENCH_INCLUDE_CANDY_GRID2DTENSORINDEX_GRID2DOFTENSOR_H_
#define CANDYBENCH_INCLUDE_CANDY_GRID2DTENSORINDEX_GRID2DOFTENSOR_H_
#include <torch/torch.h>
#include <vector>
#include <stdint.h>
#include <memory>
namespace CANDY {
/**
 * @ingroup  CANDY_lib_bottom_sub The support classes for index approaches
 * @{
 */
/**
* @class GridUnitOfTensor Grid2DIndex/Grid2DOfTensor
* @brief a minimal grid unit
*/
class GridUnitOfTensor {
 public:
  GridUnitOfTensor() {}
  ~GridUnitOfTensor() {}
  int64_t idx_ = -1;
  torch::Tensor data_;
};

typedef std::shared_ptr<class CANDY::GridUnitOfTensor> GridUnitOfTensorPtr;
#define newGridUnitOfTensor std::make_shared<CANDY::GridUnitOfTensor>
/**
 * @class Grid2DOfTensor Grid2DIndex/Grid2DOfTensor
 * @brief the whole class of a grid
 */
class Grid2DOfTensor {
 protected:
  std::vector<std::vector<GridUnitOfTensorPtr>> dataGrid;
  int64_t numberOfGrids_=-1;
 public:
  Grid2DOfTensor() {}
  ~Grid2DOfTensor() {}
  /**
   * @brief init the whole 2D grid
   * @param numberOfGrids number of grids in each dimension
   */
 void init(int64_t numberOfGrids);
  /**
   * @brief insert an idx to [x,y[
   * @param x the first coordinate
   * @param y the second coordinate
   * @param item the data item
   * @return 1 for success
   */
  int64_t insertItemToGrid(int64_t x,int64_t y,int64_t item);
  /**
   * @brief get the exact grid at [x,y]
   * @param x the first coordinate
   * @param y the second coordinate
   * @return the grid unit, nullptr if nothing
   */
  GridUnitOfTensorPtr getExactGridUnit(int64_t x,int64_t y);
  /**
  * @brief get the approximate grids at the square of  [x+-extension,y+-extension]
  * @param x the first coordinate
  * @param y the second coordinate
  * @param extension the extension range
  * @param skipExact whether or not skip the [x,y] grid
  * @return the vector of grid units
  */
  std::vector<GridUnitOfTensorPtr> getApproximateGridUnitsSquare(int64_t x,int64_t y,int64_t extension,int64_t skipExact=1);
  /**
  * @brief get the approximate indicies at the square near  [x,y] until we have at least k results
  * @param x the first coordinate
  * @param y the second coordinate
  * @param k the expected number of data indicies
  * @return the indicies tensor sized >=k
  */
  torch::Tensor getApproximateIndiciesUntilK(int64_t x,int64_t y,int64_t k);
};
/**
 * @}
 */
} // CANDY

#endif //CANDYBENCH_INCLUDE_CANDY_GRID2DTENSORINDEX_GRID2DOFTENSOR_H_
