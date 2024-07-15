#ifndef SESAME_INCLUDE_ALGORITHM_DSTREAM_HPP_
#define SESAME_INCLUDE_ALGORITHM_DSTREAM_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/CharacteristicVector.hpp"
#include "Algorithm/DataStructure/DensityGrid.hpp"
#include "Algorithm/DataStructure/GridCluster.hpp"
#include "Utils/BenchmarkUtils.hpp"

#include <omp.h>

/* Used Parameter in DStream:
double lambda: user defined parameter lambda in damped window
double beta: control the time interval of deleting the same sporadic grid
double cm: controls the threshold for dense grids
double cl: controls the threshold for sparse grid, require cm > cl
double grid_width: width of grid (default with the same width in every
dimension) double gap: self-updated based on cm and cl according to eq 26 in the
paper Note: 1.Since it is unrealistic to set the number of grids in the total
feature space with a fixed value ahead of time, in this implementation, we
timely adjust N during the clustering procedure: pi_(maxVals - minVals) /
grid_width. 2.For simplicity, we directly use the data index as the its arriving
timestamp.
**/

namespace SESAME {
class DStream;
typedef std::unordered_map<DensityGrid, CharacteristicVector, GridKeyHash,
                           EqualGrid>
    HashMap;
class DStream : public Algorithm {
public:
  DampedWindowPtr dampedWindow;
  double startTime = 0.0;
  int currentTimeStamp;
  int gap;    // Time gap between calls to the offline component
  double dm;  // Density threshold for dense grids; controlled by cm
  double dl;  //  Density threshold for sparse grids; controlled by cl
  int NGrids; // The number of density grids ,with an initial value 0
  HashMap gridList;
  std::unordered_map<DensityGrid, int, GridKeyHash, EqualGrid> deletedGrids;
  // Store the deleted sporadic grids: <coordinate, deleteTime>
  std::vector<GridCluster> clusterList;    // A list of all Grid Clusters
  std::vector<GridCluster> newClusterList; // A list of grid clusters used when
                                           // re-clustering an existing cluster.
  std::vector<double> minVals; // The minimum value seen for a numerical dim;
                               // used to calculate N
  std::vector<double> maxVals; // The maximum value seen for a numerical dim;
                               // used to calculate N
  bool init = false;

  DStream(param_t &cmd_params);
  ~DStream();
  void Init() override;
  void RunOnline(PointPtr input) override;
  void RunOffline(DataSinkPtr sinkPtr) override;

private:
  bool recalculateN = false; // flag indicating whether N needs to be
                             // recalculated after this instance
  std::vector<int> Coord;
  void ifReCalculate(PointPtr point);
  void reCalculateParameter();
  void GridListUpdate(std::vector<int> coordinate);
  void initialClustering();
  void adjustClustering();
  bool adjustLabels();
  bool inspectChangedGrids();
  HashMap adjustForSparseGrid(DensityGrid grid,
                              CharacteristicVector characteristicVec,
                              int gridClass);
  HashMap adjustForDenseGrid(DensityGrid grid,
                             CharacteristicVector characteristicVec,
                             int gridClass);
  HashMap adjustForTransitionalGrid(DensityGrid grid,
                                    CharacteristicVector characteristicVec,
                                    int gridClass);
  void removeSporadic();
  HashMap reCluster(GridCluster gridCluster);
  HashMap adjustNewLabels(HashMap newGridList);
  void mergeClusters(int smallCluster, int bigCluster);
  void cleanClusters();
  HashMap cleanNewClusters(HashMap newGridList);
  HashMap mergeNewClusters(HashMap newGridList, int smallCluster,
                           int bigCluster);
  double outlier_density_thresholdFunction(int tg, double cl,
                                           double decayFactor, int NGrids);
  bool checkIfSporadic(CharacteristicVector characteristicVec);
  void updateGridListDensity();
  static void mergeGridList(HashMap &gridList, const HashMap &otherList);
  // HashMap putHashMap(HashMap gList, const DensityGrid& g,
  // CharacteristicVector cv);
};

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_DSTREAM_HPP_
