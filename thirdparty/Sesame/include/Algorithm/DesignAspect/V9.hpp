//
// Created by 1124a on 2022/4/10.
//

#ifndef COVERTBIRCH_FILE_INCLUDE_ALGORITHM_DESIGNASPECT_V9_HPP_
#define COVERTBIRCH_FILE_INCLUDE_ALGORITHM_DESIGNASPECT_V9_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/CharacteristicVector.hpp"
#include "Algorithm/DataStructure/DensityGrid.hpp"
#include "Algorithm/DataStructure/GridCluster.hpp"
#include "Algorithm/WindowModel/LandmarkWindow.hpp"
#include "Sinks/DataSink.hpp"
#include "Utils/BenchmarkUtils.hpp"

#include <omp.h>

/* This algorithm is composed by landmark window + Grids + Outlier detection
 * with buffer + no refinement Note: we have removed all of the unnecessary
 * modules since subsituting damped window with the landmark one makes the
 * algorithm much more simple
 * 1. the dm, dl, gap under landmark window is unchanged, which is cm, cl, cm -
 * cl, following the assumption and calculation equation of 25 in DStream paper
 * (equation 11 is not needed since the weight will not decay in landmark
 * window).
 * 2. the removeSporadic and checkIfSporadic function can be removed since the
 * algorithm does not use outlier detection with timer
 * 3. at this time, we donnot need to recalculate the parameter and thus
 * ifReCalculate and reCalculateParameter function can also be removed
 * 4. we remove outlier_density_thresholdFunction function and set lambda = = 1
 * for transfer convenience. Even if the weight is updated it is still the same.
 * 5. In original DStream, it does not specifically set a buffer to store the
 * outlier grid, however, it separate those outlier with the normal grid using a
 * "SPARSE" label and still store them in the grid_list rather than deleting
 * them. So we still treat it as using outlier buffer
 * */
namespace SESAME {

typedef std::unordered_map<DensityGrid, CharacteristicVector, GridKeyHash,
                           EqualGrid>
    HashMap;
class V9 : public Algorithm {
public:
  int currentTimeStamp = 0;
  int lastLandmark = 0;
  int gap;   // Time gap between calls to the offline component
  double dm; // Density threshold for dense grids; controlled by cm
  double dl; //  Density threshold for sparse grids; controlled by cl
  HashMap gridList;
  // Store the deleted sporadic grids: <coordinate, deleteTime>
  std::vector<GridCluster> clusterList;    // A list of all Grid Clusters
  std::vector<GridCluster> newClusterList; // A list of grid clusters used when
                                           // re-clustering an existing cluster.
  std::vector<double> minVals; // The minimum value seen for a numerical dim;
                               // used to calculate N
  std::vector<double> maxVals; // The maximum value seen for a numerical dim;
                               // used to calculate N
  bool init = false;
  std::vector<int> Coord;
  std::vector<PointPtr> onlineCenters;
  int q = 0;

  V9(param_t &cmd_params);
  ~V9();
  void Init();
  void RunOnline(PointPtr input) override;
  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  void GridListUpdate(const std::vector<int> &coordinate);
  void initialClustering();
  void adjustClustering();
  bool adjustLabels();
  bool inspectChangedGrids();
  void calculateGridCoord(PointPtr point);
  HashMap adjustForSparseGrid(const DensityGrid &grid,
                              CharacteristicVector characteristicVec,
                              int gridClass);
  HashMap adjustForDenseGrid(const DensityGrid &grid,
                             CharacteristicVector characteristicVec,
                             int gridClass);
  HashMap adjustForTransitionalGrid(const DensityGrid &grid,
                                    CharacteristicVector characteristicVec,
                                    int gridClass);
  HashMap reCluster(GridCluster &gridCluster);
  HashMap adjustNewLabels(const HashMap &newGridList);
  void mergeClusters(int smallCluster, int bigCluster);
  void cleanClusters();
  HashMap cleanNewClusters(HashMap newGridList);
  HashMap mergeNewClusters(HashMap newGridList, int smallCluster,
                           int bigCluster);
  void updateGridListDensity();
  static void mergeGridList(HashMap &gridList, const HashMap &otherList);
  bool checkIfSporadic(CharacteristicVector characteristicVec);
  void removeSporadic();
};
} // namespace SESAME
#endif // COVERTBIRCH_FILE_INCLUDE_ALGORITHM_DESIGNASPECT_V9_HPP_
