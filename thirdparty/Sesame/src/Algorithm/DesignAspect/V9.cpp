//
// Created by 1124a on 2021/10/27.
//

#include "Algorithm/DesignAspect/V9.hpp"
#include "Algorithm/DataStructure/DataStructureFactory.hpp"
#include "Algorithm/WindowModel/WindowFactory.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

using namespace std;

SESAME::V9::V9(param_t &cmd_params) {
  this->param = cmd_params;
  param.lambda = 1;
  sum_timer.Tick();
  gap = (int)(param.cm - param.cl);
  dm = param.cm;
  dl = param.cl;
  minVals = std::vector<double>(param.dim, DBL_MAX);
  maxVals = std::vector<double>(param.dim, DBL_MIN);
  Coord = std::vector<int>(param.dim, 0);
}

SESAME::V9::~V9() = default;

void SESAME::V9::Init() {}

void SESAME::V9::calculateGridCoord(PointPtr point) {
  for (int i = 0; i < param.dim; i++) {
    auto feature = point->getFeatureItem(i);
    if (feature > maxVals[i]) {
      maxVals[i] = feature;
    } else if (feature < minVals[i]) {
      minVals[i] = feature;
    }
    Coord[i] = point->getFeatureItem(i) / param.grid_width;
  }
}

void SESAME::V9::RunOnline(PointPtr input) {
  win_timer.Tick();
  this->currentTimeStamp = input->index;
  if (input->getIndex() != 0 and input->getIndex() % param.landmark == 0) {
    lastLandmark = input->getIndex();
    for (auto iter = 0; iter != this->clusterList.size(); iter++) {
      PointPtr point = GenericFactory::New<Point>(param.dim, iter);
      auto count = 0;
      for (auto &iterGrid : this->clusterList.at(iter).grids) {
        for (int iterDim = 0; iterDim < param.dim; iterDim++) {
          if (count == 0)
            point->setFeatureItem(0, iterDim);
          point->setFeatureItem(point->getFeatureItem(iterDim) +
                                    iterGrid.first.coordinates[iterDim],
                                iterDim);
          if (count == this->clusterList.at(iter).grids.size() - 1) {
            point->setFeatureItem(point->getFeatureItem(iterDim) / param.dim,
                                  iterDim);
          }
        }
        double weight = gridList.find(iterGrid.first)->second.gridDensity;
        point->setWeight(point->getWeight() + weight);
        count++;
      }
      onlineCenters.push_back(point);
    }
    // clean the old clustering information and reinitial every middle variable
    minVals = std::vector<double>(param.dim, DBL_MAX);
    maxVals = std::vector<double>(param.dim, DBL_MIN);
    clusterList = std::vector<GridCluster>();
    newClusterList = std::vector<GridCluster>();
    Coord = std::vector<int>(param.dim, 0);
    gap = max(1.0, param.cm - param.cl);
    gridList = HashMap();
    init = false;
  }
  win_timer.Tock();
  ds_timer.Tick();
  calculateGridCoord(input);
  GridListUpdate(Coord); // tempCoord
  if (!init && (currentTimeStamp - lastLandmark) == gap) {
    initialClustering();
    init = true;
  }
  if (currentTimeStamp != lastLandmark and
      (currentTimeStamp - lastLandmark) % gap == 0) {
    removeSporadic();
    adjustClustering();
  }
  ds_timer.Tock();
  lat_timer.Add(input->toa);
}

void SESAME::V9::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  int cluID = 0;
  for (const auto &point : onlineCenters) {
    point->setClusteringCenter(cluID++);
    sinkPtr->put(point);
  }
  for (auto iter = 0; iter != this->clusterList.size(); iter++) {
    PointPtr point = GenericFactory::New<Point>(param.dim, iter);
    auto count = 0;
    for (auto &iterGrid : this->clusterList.at(iter).grids) {
      for (int iterDim = 0; iterDim < param.dim; iterDim++) {
        if (count == 0)
          point->setFeatureItem(0, iterDim);
        point->setFeatureItem(point->getFeatureItem(iterDim) +
                                  iterGrid.first.coordinates[iterDim],
                              iterDim);
        if (count == this->clusterList.at(iter).grids.size() - 1) {
          point->setFeatureItem(point->getFeatureItem(iterDim) / param.dim,
                                iterDim);
        }
      }
      double weight = gridList.find(iterGrid.first)->second.gridDensity;
      point->setWeight(point->getWeight() + weight);
      count++;
    }
    point->setClusteringCenter(cluID++);
    sinkPtr->put(point);
  }
  ref_timer.Tock();
  sum_timer.Tock();
}

/* Update the grid list of V9 when data inserting into the grid
 * */
void SESAME::V9::GridListUpdate(const std::vector<int> &coordinate) {
  CharacteristicVector characteristicVec;
  DensityGrid grid(coordinate);
  // 3. If (g not in grid_list) insert dg to grid_list
  auto it = this->gridList.find(grid);
  q = this->gridList.size();
  if (it == gridList.end()) {
    characteristicVec =
        CharacteristicVector(currentTimeStamp, 0, 1.0, -1, false, dl, dm);
    this->gridList.insert(std::make_pair(grid, characteristicVec));
  }
  // 4. Update the characteristic vector of dg
  else {
    it->second.densityWithNew(currentTimeStamp, param.lambda);
    it->second.updateTime = currentTimeStamp;
  }
}

/**
 * Implements the procedure given in Figure 3 of Chen and Tu 2007
 */
void SESAME::V9::initialClustering() {
  // 1. Update the density of all grids in grid_list
  // Timer: online grid
  updateGridListDensity();
  // 2. Assign each dense grid to a distinct cluster
  // and
  // 3. Label all other grids as NO_CLASS
  auto gridIter = this->gridList.begin();
  HashMap newGridList;
  while (gridIter != gridList.end()) {
    DensityGrid grid = gridIter->first;
    CharacteristicVector characteristicVecOfG = gridIter->second;
    if (characteristicVecOfG.attribute == DENSE) {
      int gridClass = this->clusterList.size();
      characteristicVecOfG.label = gridClass;
      GridCluster gridCluster = GridCluster(gridClass);
      gridCluster.addGrid(grid);
      this->clusterList.push_back(gridCluster);
      //  // SESAME_INFO(" was dense (class "<<gridClass<<")" <<"Position is "<<
      //  this->clusterList.size());
    } else
      characteristicVecOfG.label = NO_CLASS;
    newGridList.insert(std::make_pair(grid, characteristicVecOfG));
    ++gridIter;
  }
  this->gridList = newGridList;
  // 4. Make changes to grid labels by doing:
  //    a. For each cluster c
  //    b. For each outside grid g of c
  //    c. For each neighbouring grid h of g
  //    d. If h belongs to c', label c and c' with
  //       the label of the largest cluster
  //    e. Else if h is transitional, assign it to c
  //    f. While changes can be made
  while (adjustLabels())
    ; // while changes are being made
}
/**
 * Makes first change available to it by following the steps:
 * For each cluster c<
 * For each outside grid g of c
 * For each neighbouring grid h of g
 * If h belongs to c', label c and c' with the label of the largest cluster
 * Else if h is transitional, assign it to c
 * @return TRUE if a change was made to any cluster's labels, FALSE otherwise
 */
bool SESAME::V9::adjustLabels() {
  // bool adjust=false;
  // a. For each cluster c
  for (GridCluster &gridCluster : this->clusterList) {
    // // SESAME_INFO("Adjusting from cluster "<<gridCluster.clusterLabel<<",
    // standby..."); b. for each grid, dg, of c
    for (auto gridIter = gridCluster.grids.begin();
         gridIter != gridCluster.grids.end(); gridIter++) {
      {
        const DensityGrid &grid = gridIter->first;
        bool inside = gridIter->second;
        // // SESAME_INFO(" Inspecting density grid, grid, standby...");

        // b. for each OUTSIDE grid of cluster
        if (!inside) {
          //  // SESAME_INFO(" Density grid dg is outside!");
          // c. for each neighbouring grid, of current iter grid
          for (const DensityGrid &gridNeighbourhood : grid.getNeighbours()) {
            auto it2 = this->gridList.find(gridNeighbourhood);
            if (it2 != gridList.end()) {
              auto it1 = this->gridList.find(grid);
              CharacteristicVector characteristicVec1 = it1->second;
              CharacteristicVector characteristicVec2 = it2->second;
              int class1 = characteristicVec1.label;
              int class2 = characteristicVec2.label;
              // ...and if neighbouring grid isn't already in the same cluster
              // as grid...
              if (class1 != class2) {
                // If neighbouring grid is in cluster c', merge c and c' into
                // the larger of the two
                if (class2 != NO_CLASS) {
                  if (this->clusterList.at(class1).grids.size() <
                      this->clusterList.at(class2).grids.size())
                    mergeClusters(class1, class2);
                  else
                    mergeClusters(class2, class1);
                  return true;
                }
                // If gridNeighbourhood is transitional and 'outside' of the
                // cluster, assign it to cluster
                else if (characteristicVec2.isTransitional(dm, dl)) {
                  characteristicVec2.label = class1;
                  gridCluster.addGrid(gridNeighbourhood);
                  this->clusterList.at(class1) = gridCluster;
                  if (it1 != gridList.end())
                    it1->second = characteristicVec2;
                  else
                    this->gridList.insert(
                        std::make_pair(grid, characteristicVec2));
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

/**
 * Iterates through grid_list and updates the density for each density grid
 * therein. Also marks each density grid as unvisited for this call to
 * adjustClustering.
 */
void SESAME::V9::updateGridListDensity() {
  for (auto &iter : this->gridList) {
    iter.second.isVisited = false;
    iter.second.UpdateAllDensity(currentTimeStamp, param.lambda, dl, dm);
  }
}

/**
 * Performs the periodic adjustment of clusters every 'gap' timesteps.
 * Implements the procedure given in Figure 4 of Chen and Tu 2007
 *
 * @see moa.clusterers.V9.V9#gap
 */
void SESAME::V9::adjustClustering() {
  // SESAME_INFO("ADJUST CLUSTERING CALLED ");
  // 1. Update the density of all grids in grid_list
  updateGridListDensity();
  // 2. For each grid dg whose attribute is changed since last call
  //    a. If dg is sparse
  //    b. If dg is dense
  //    c. If dg is transitional
  while (inspectChangedGrids())
    ;
}

/**
 * Inspects each density grid in grid_list whose attribute has changed since the
 * last call to adjustClustering. Implements lines 3/4/7/19 of the procedure
 * given in Figure 4 of Chen and Tu 2007.
 *
 * @return TRUE if any grids are updated; FALSE otherwise.
 */
bool SESAME::V9::inspectChangedGrids() {
  HashMap newGridList;
  auto gridIter = this->gridList.begin();
  int a = 0;
  while (gridIter != gridList.end()) //&& newGridList.empty()
  {
    const DensityGrid &grid = gridIter->first;
    const CharacteristicVector &characteristicVec = gridIter->second;
    int gridClass = characteristicVec.label;
    if (characteristicVec.attChange &&
        !characteristicVec.isVisited) // grid.isVisited
    {                                 // grid.isVisited=true;
      gridIter->second.isVisited = true;
      newGridList.insert(std::make_pair(grid, characteristicVec));
      if (characteristicVec.attribute == SPARSE)
        mergeGridList(newGridList,
                      adjustForSparseGrid(grid, characteristicVec, gridClass));
      else if (characteristicVec.attribute == DENSE)
        mergeGridList(newGridList,
                      adjustForDenseGrid(grid, characteristicVec, gridClass));
      else // TRANSITIONAL
        mergeGridList(newGridList, adjustForTransitionalGrid(
                                       grid, characteristicVec, gridClass));
    }
    ++gridIter;
    a++;
  }
  // If there are grids in new grid list, update the corresponding grids in
  // grid_list and clean up the cluster list
  if (!newGridList.empty()) {
    mergeGridList(this->gridList, newGridList);
    cleanClusters();
    return true;
  } else
    return false;
}

/**
 * Adjusts the clustering of a sparse density grid. Implements lines 5 and 6
 * from Figure 4 of Chen and Tu 2007.
 *
 * @param dg the sparse density grid being adjusted
 * @param cv the characteristic vector of dg
 * @param dgClass the cluster to which dg belonged
 *
 * @return a HashMap containing density grids for update after this iteration
 */
SESAME::HashMap
SESAME::V9::adjustForSparseGrid(const DensityGrid &grid,
                                CharacteristicVector characteristicVec,
                                int gridClass) {
  HashMap newGridList;
  if (gridClass != NO_CLASS) {
    //// SESAME_INFO("It is removed from cluster "<<gridClass<<".");
    for (auto gridCluster : clusterList) {
      if (gridCluster.clusterLabel == gridClass) {
        gridCluster.removeGrid(grid);
        characteristicVec.label = NO_CLASS;
        newGridList.insert(std::make_pair(grid, characteristicVec));
        if (!gridCluster.grids.empty() && !gridCluster.isConnected())
          mergeGridList(newGridList, reCluster(gridCluster));
      }
    }
  }
  return newGridList;
}

/**
 * Reclusters a grid cluster into two (or more) constituent clusters when it has
 * been identified that the original cluster is no longer a grid group. It does
 * so by echoing the initial clustering procedure over only those grids in gc.
 * @param gridCluster the grid cluster to be re clustered
 * @return a HashMap<DensityGrid, CharacteristicVector> containing density grids
 * for update after this iteration
 */
SESAME::HashMap SESAME::V9::reCluster(GridCluster &gridCluster) {
  // SESAME_INFO("Now re-cluster!");
  HashMap newGridList;
  auto gcIter = gridCluster.grids.begin();
  // // SESAME_INFO("ReCluster called for cluster "<<gridCluster.clusterLabel);
  // Assign every dense grid in gc to its own cluster, assign all other grids to
  // NO_CLASS
  while (gcIter != gridCluster.grids.end()) {
    DensityGrid grid = gcIter->first;
    CharacteristicVector characteristicVecOfGrid =
        this->gridList.find(grid)->second;
    if (characteristicVecOfGrid.attribute == DENSE) {
      int gridClass = (int)newClusterList.size();
      characteristicVecOfGrid.label = gridClass;
      GridCluster newCluster(gridClass);
      newCluster.addGrid(grid);
      newClusterList.push_back(newCluster);
    } else
      characteristicVecOfGrid.label = NO_CLASS;
    newGridList.insert(std::make_pair(grid, characteristicVecOfGrid));
    gcIter++;
  }

  bool changesMade;
  // While changes can be made...
  do {
    changesMade = false;
    HashMap gridListAdjusted = adjustNewLabels(newGridList);
    if (!gridListAdjusted.empty()) {
      // SESAME_INFO("grid list is adjusted for sparse!");
      mergeGridList(newGridList, gridListAdjusted);
      changesMade = true;
    }
  } while (changesMade);

  // Update the cluster list with the newly formed clusters
  gridCluster.grids.clear();
  this->clusterList.at(gridCluster.clusterLabel) = gridCluster;
  for (GridCluster &cluster : newClusterList)
    this->clusterList.push_back(cluster);
  return newGridList;
}

SESAME::HashMap
SESAME::V9::adjustNewLabels(const SESAME::HashMap &newGridList) {
  HashMap gridListAdjusted;
  // a. For each cluster c
  for (GridCluster &gridCluster : newClusterList) {
    for (auto &gridIter : gridCluster.grids) {
      DensityGrid grid = gridIter.first;
      bool inside = gridIter.second;

      // b. for each OUTSIDE grid, dg, of c
      if (!inside) {
        // c. for each neighbouring grid, neighbourGrid, of dg

        for (DensityGrid &neighbourGrid : grid.getNeighbours()) {
          if (newGridList.find(neighbourGrid) != newGridList.end()) {
            CharacteristicVector characteristicVec1 =
                newGridList.find(neighbourGrid)->second;
            CharacteristicVector characteristicVec2 =
                newGridList.find(neighbourGrid)->second;
            int class1 = characteristicVec1.label;
            int class2 = characteristicVec2.label;

            // ...and if neighbourGrid isn't already in the same cluster as
            // dg...
            if (class1 != class2) {
              GridCluster cluster1 = newClusterList.at(class1);
              // If dgprime is in cluster c', merge c and c' into the larger of
              // the two
              if (class2 != NO_CLASS) {
                GridCluster cluster2 = newClusterList.at(class2);
                // System.out.println("C is "+class1+" and C' is "+class2+".");
                if (cluster1.grids.size() < cluster2.grids.size())
                  mergeGridList(gridListAdjusted,
                                mergeNewClusters(newGridList, class1, class2));
                else
                  mergeGridList(gridListAdjusted,
                                mergeNewClusters(newGridList, class2, class1));

                return gridListAdjusted;
              }
              // If neighbourGrid is transitional and outside of cluster, assign
              // it to cluster
              else if (characteristicVec2.isTransitional(dm, dl)) {
                characteristicVec2.label = class1;
                cluster1.addGrid(neighbourGrid);
                this->newClusterList.at(class1) = cluster1;
                gridListAdjusted.insert(
                    std::make_pair(neighbourGrid, characteristicVec2));
                return gridListAdjusted;
              }
            }
          }
        }
      }
    }
  }
  return gridListAdjusted;
}

/**
 * Adjusts the clustering of a dense density grid. Implements lines 8 through 18
 * from Figure 4 of Chen and Tu 2007.
 *
 * @param grid the dense density grid being adjusted
 * @param characteristicVec the characteristic vector of dg
 * @param gridClass the cluster to which dg belonged
 *
 * @return a HashMapcontaining density grids for update after this iteration
 */
SESAME::HashMap
SESAME::V9::adjustForDenseGrid(const DensityGrid &grid,
                               CharacteristicVector characteristicVec,
                               int gridClass) {
  // Among all neighbours of dg, find the grid h whose cluster ch has the
  // largest size
  GridCluster gridCluster; // The cluster, ch, of h
  DensityGrid gridChosen(
      grid); // The chosen grid h, whose cluster ch has the largest size
  double ChosenGridSize = -1.0; // The size of gridCluster, the largest cluster
  int hClass = NO_CLASS;        // The class label of h
  int hChosenClass = NO_CLASS;  // The class label of ch

  HashMap newGridList;
  //// SESAME_INFO("adjust For Dense Grid "<<gridClass<<".");
  for (DensityGrid &neighbourGrid :
       grid.getNeighbours()) // The neighbour of g being considered
  {
    if (this->gridList.find(neighbourGrid) != gridList.end()) {
      hClass = this->gridList.find(neighbourGrid)->second.label;
      if (hClass != NO_CLASS) {
        for (auto gridCluster : clusterList) {
          if (gridCluster.clusterLabel == hClass) {
            if (gridCluster.grids.size() > ChosenGridSize) {
              ChosenGridSize = gridCluster.grids.size();
              hChosenClass = hClass;
              gridChosen = DensityGrid(neighbourGrid);
            }
          }
        }
      }
    }
  }

  if (hChosenClass != NO_CLASS && hChosenClass != gridClass) {
    gridCluster = this->clusterList.at(hChosenClass);

    // If h is a dense grid
    if (this->gridList.find(gridChosen)->second.attribute == DENSE) {
      // // SESAME_INFO("h is dense.");
      // If dg is labelled as NO_CLASS
      if (gridClass == NO_CLASS) {
        //  // SESAME_INFO("g was labelled NO_CLASS");
        characteristicVec.label = hChosenClass;
        newGridList.insert(std::make_pair(grid, characteristicVec));
        gridCluster.addGrid(grid);
        this->clusterList.at(hChosenClass) = gridCluster;
      }
      // Else if dg belongs to cluster c and h belongs to c'
      else {
        // // SESAME_INFO("g was labelled "<<gridClass);
        double gSize = this->clusterList.at(gridClass).grids.size();

        if (gSize <= ChosenGridSize)
          mergeClusters(gridClass, hChosenClass);
        else
          mergeClusters(hChosenClass, gridClass);
      }
    }

    // Else if h is a transitional grid
    else if (this->gridList.at(gridChosen).attribute == TRANSITIONAL) {
      //  // SESAME_INFO("h is transitional.");
      // If dg is labelled as no class and if h is an outside grid if dg is
      // added to ch
      if (gridClass == NO_CLASS && !gridCluster.isInside(gridChosen, grid)) {
        characteristicVec.label = hChosenClass;
        newGridList.insert(std::make_pair(grid, characteristicVec));
        gridCluster.addGrid(grid);
        this->clusterList.at(hChosenClass) = gridCluster;
        // // SESAME_INFO(" dg is added to cluster "<<hChosenClass<<".");
      }
      // Else if dg is in cluster c and |c| >= |ch|
      else if (gridClass != NO_CLASS) {
        GridCluster c = this->clusterList.at(gridClass);
        double gSize = c.grids.size();

        if (gSize >= ChosenGridSize) {
          // Move h from cluster ch to cluster c
          gridCluster.removeGrid(gridChosen);
          c.addGrid(gridChosen);
          CharacteristicVector cvhChosen =
              this->gridList.find(gridChosen)->second;
          cvhChosen.label = gridClass;
          newGridList.insert(std::make_pair(gridChosen, cvhChosen));
          // // SESAME_INFO("dgClass is "<<gridClass<<", hChosenClass is
          // "<<hChosenClass<<", gSize is "<<gSize<< " and hChosenSize is
          // "<<ChosenGridSize<<" h is added to cluster "<<gridClass<<".");
          this->clusterList.at(hChosenClass) = gridCluster;
          this->clusterList.at(gridClass) = c;
        }
      }
    }
  }
  // If dgClass is dense and not in a cluster, and none if its neighbours are in
  // a cluster, put it in its own new cluster and search the neighbourhood for
  // transitional or dense grids to add
  else if (gridClass == NO_CLASS) {
    int newClass = (int)this->clusterList.size();
    GridCluster c = GridCluster(newClass);
    c.addGrid(grid);
    // System.out.println("Added "+dg.toString()+" to cluster "+newClass+".");
    this->clusterList.push_back(c);
    characteristicVec.label = newClass;
    if (newGridList.find(grid) != newGridList.end())
      newGridList.find(grid)->second = characteristicVec;
    else
      newGridList.insert(std::make_pair(grid, characteristicVec));
    // Iterate through the neighbourhood until no more transitional neighbours
    // can be added (dense neighbours will add themselves as part of their
    // adjust process)
    for (DensityGrid &dghprime : grid.getNeighbours()) {
      if (this->gridList.find(dghprime) != this->gridList.end() &&
          c.grids.find(dghprime) != c.grids.end()) {
        CharacteristicVector cvhprime = this->gridList.find(dghprime)->second;
        if (cvhprime.attribute == TRANSITIONAL) {
          c.addGrid(dghprime);
          cvhprime.label = newClass;
          newGridList.insert(std::make_pair(dghprime, cvhprime));
        }
      }
    }
    this->clusterList.at(newClass) = c;
  }

  return newGridList;
}

/**
 * Adjusts the clustering of a transitional density grid. Implements lines 20
 * and 21 from Figure 4 of Chen and Tu 2007.
 *
 * @param dg the dense density grid being adjusted
 * @param cv the characteristic vector of dg
 * @param dgClass the cluster to which dg belonged
 *
 * @return a HashMap<DensityGrid, CharacteristicVector> containing density grids
 * for update after this iteration
 */
SESAME::HashMap
SESAME::V9::adjustForTransitionalGrid(const DensityGrid &grid,
                                      CharacteristicVector characteristicVec,
                                      int gridClass) {
  // Among all neighbours of dg, find the grid h whose cluster ch has the
  // largest size and satisfies that dg would be an outside grid if added to it
  GridCluster gridCluster;     // The cluster, ch, of h
  double hChosenSize = 0.0;    // The size of ch, the largest cluster
  DensityGrid neighbourGrid;   // The neighbour of dg being considered
  int hClass = NO_CLASS;       // The class label of h
  int hChosenClass = NO_CLASS; // The class label of ch
  HashMap newGridList;
  //// SESAME_INFO("adjust For Transitional Grid "<<gridClass<<".");
  for (DensityGrid &neighbourGrid : grid.getNeighbours()) {
    auto it = this->gridList.find(neighbourGrid);
    if (it != gridList.end()) {
      hClass = it->second.label;
      ;
      if (hClass != NO_CLASS) {
        for (auto gridCluster : clusterList) {
          if (gridCluster.clusterLabel == hClass) {
            if ((gridCluster.grids.size() > hChosenSize) &&
                !gridCluster.isInside(grid, grid)) {
              hChosenSize = gridCluster.grids.size();
              hChosenClass = hClass;
            }
          }
        }
      }
    }
  }

  if (hChosenClass != NO_CLASS && hChosenClass != gridClass) {
    for (auto gridCluster : clusterList) {
      if (gridCluster.clusterLabel == hChosenClass)
        gridCluster.addGrid(grid);
      if (gridCluster.clusterLabel == gridClass and gridClass != NO_CLASS)
        gridCluster.removeGrid(grid);
    }
    gridCluster = this->clusterList.at(hChosenClass);
    gridCluster.addGrid(grid);
    this->clusterList.at(hChosenClass) = gridCluster;

    characteristicVec.label = hChosenClass;
    newGridList.insert(std::make_pair(grid, characteristicVec));
  }

  return newGridList;
}

SESAME::HashMap SESAME::V9::mergeNewClusters(SESAME::HashMap newGridList,
                                             int smallCluster, int bigCluster) {
  // System.out.println("Merge new clusters "+smallCluster+" and
  // "+bigCluster+"."); Iterate through the density grids in grid_list to find
  // those which are in highClass
  for (HashMap::iterator gridIter = newGridList.begin();
       gridIter != gridList.end(); gridIter++) {
    DensityGrid grid = gridIter->first;
    CharacteristicVector characteristicVec = gridIter->second;

    // Assign density grids in small Cluster to bigCluster
    if (characteristicVec.label == smallCluster) {
      characteristicVec.label = bigCluster;
      newGridList.insert(std::make_pair(grid, characteristicVec));
    }
  }
  // SESAME_INFO("Density grids assigned to cluster "<<bigCluster<<".");

  // Merge the GridCluster objects representing each cluster
  GridCluster bGC = this->newClusterList.at(bigCluster);
  bGC.absorbCluster(this->newClusterList.at(smallCluster));
  this->newClusterList.at(bigCluster) = bGC;
  this->newClusterList.erase(this->newClusterList.begin() + smallCluster);
  // System.out.println("Cluster "+smallClus+" removed from list.");
  newGridList = cleanNewClusters(newGridList);

  return newGridList;
}

/**
 * Reassign all grids belonging in the small cluster to the big cluster
 * Merge the GridCluster objects representing each cluster
 *
 * @param smallCluster - the index of the smaller cluster
 * @param bigCluster - the index of the bigger cluster
 */
void SESAME::V9::mergeClusters(int smallCluster, int bigCluster) {
  // SESAME_INFO("Merge clusters "<<smallCluster<<" and "<<bigCluster<<".");
  // Iterate through the density grids in grid_list to find those which are in
  // highClass
  for (auto gridIter = gridList.begin(); gridIter != gridList.end();
       gridIter++) {
    // Assign density grids in smallCluster to bigCluster
    if (gridIter->second.label == smallCluster) {
      gridIter->second.label = bigCluster;
    }
  }
  // SESAME_INFO("Density grids assigned to cluster "<<bigCluster<<".");
  // Merge the GridCluster objects representing each cluster
  GridCluster bigGridCluster = this->clusterList.at(bigCluster);
  bigGridCluster.absorbCluster(this->clusterList.at(smallCluster));
  this->clusterList.at(bigCluster) = bigGridCluster;
  this->clusterList.erase(clusterList.begin() + smallCluster);
  // SESAME_INFO("Cluster "<<smallCluster<<" removed from list.");
  cleanClusters();
}

void SESAME::V9::mergeGridList(HashMap &thisGridList,
                               const HashMap &otherList) {
  for (auto &gridIter : otherList) {
    thisGridList[gridIter.first] = gridIter.second;
  }
}

SESAME::HashMap SESAME::V9::cleanNewClusters(SESAME::HashMap newGridList) {
  std::vector<GridCluster> toRemove;
  // Check to see if there are any empty clusters
  for (auto &cluster : this->newClusterList) {
    if (cluster.grids.empty())
      toRemove.push_back(cluster);
  }

  // Remove empty clusters
  if (!toRemove.empty()) {
    for (auto &RemoveCluster : toRemove) {
      auto removeCIter = std::find(newClusterList.begin(), newClusterList.end(),
                                   RemoveCluster);
      if (std::find(newClusterList.begin(), newClusterList.end(),
                    RemoveCluster) != newClusterList.end())
        this->newClusterList.erase(removeCIter);
    }
  }
  for (auto &cluster : this->newClusterList) {
    auto clusterIter =
        std::find(newClusterList.begin(), newClusterList.end(), cluster);
    int index = (int)std::distance(newClusterList.begin(), clusterIter);
    cluster.clusterLabel = index;
    unordered_map<DensityGrid, bool, GridKeyHash, EqualGrid> removeGrids;
    for (auto &gridOfCluster : cluster.grids) {
      DensityGrid grid = gridOfCluster.first;
      ;
      if (newGridList.find(grid) != newGridList.end())
        newGridList.find(grid)->second.label = index;
      else
        removeGrids.insert(gridOfCluster);
    }
    for (auto &grid : removeGrids) {
      if (cluster.grids.find(grid.first) != cluster.grids.end())
        cluster.grids.erase(grid.first);
    }
  }
  // SESAME_INFO("Clean finish!");
  return newGridList;
}

// TODO stop right here and confused about get()

/**
 * Iterates through cluster_list to ensure that all empty clusters have been
 * removed and that all cluster IDs match the cluster's index in cluster_list.
 */

void SESAME::V9::cleanClusters() {
  //// SESAME_INFO("Clean Clusters");

  std::vector<GridCluster> toRemove;

  // Check to see if there are any empty clusters
  for (auto &cluster : this->clusterList) {
    if (cluster.grids.empty())
      toRemove.push_back(cluster);
  }
  // Remove empty clusters
  if (!toRemove.empty()) {
    for (auto &RemoveCluster : toRemove) {
      auto removeCIter =
          std::find(clusterList.begin(), clusterList.end(), RemoveCluster);
      if (std::find(clusterList.begin(), clusterList.end(), RemoveCluster) !=
          clusterList.end())
        this->clusterList.erase(removeCIter);
    }
  }
  // Adjust remaining clusters as necessary, index = label = order
  for (auto &cluster : this->clusterList) {
    auto clusterIter =
        std::find(clusterList.begin(), clusterList.end(), cluster);
    int index = (int)std::distance(clusterList.begin(), clusterIter);
    cluster.clusterLabel = index;
    unordered_map<DensityGrid, bool, GridKeyHash, EqualGrid> removeGrids;
    for (auto &gridOfCluster : cluster.grids) {
      DensityGrid grid = gridOfCluster.first;
      if (gridList.find(grid) != gridList.end())
        gridList.find(grid)->second.label = index;
      else
        removeGrids.insert(gridOfCluster);
    }
    for (auto &grid : removeGrids) {
      if (cluster.grids.find(grid.first) != cluster.grids.end())
        cluster.grids.erase(grid.first);
    }
    this->clusterList.at(index) = cluster;
  }
}

/**
 * Implements the procedure described in section 4.2 of Chen and Tu 2007
 *  For each grid g in grid_list
      a. If g is sporadic
          i. If currTime - tg > gap, delete g from grid_list
          ii. Else if (S1 && S2), mark as sporadic
          iii. Else, mark as normal
      b. Else
         i. If (S1 && S2), mark as sporadic
 */
void SESAME::V9::removeSporadic() {
  // SESAME_INFO("REMOVE SPORADIC CALLED");
  // For each grid g in grid_list

  HashMap newGridList;
  std::vector<DensityGrid> removeGridList;
  for (auto &gridIter : this->gridList) {
    const DensityGrid &grid = gridIter.first;
    CharacteristicVector characteristicVec = gridIter.second;
    // If g is sporadic
    if (characteristicVec.isSporadic) {
      // If currTime - tg > gap, delete g from grid_list
      if (currentTimeStamp - characteristicVec.updateTime >= gap) {
        int gridClass = characteristicVec.label;

        if (gridClass != -1) {
          for (auto gridCluster : clusterList) {
            if (gridCluster.clusterLabel == gridClass) {
              gridCluster.removeGrid(grid);
            }
          }
        }
        removeGridList.push_back(grid);
      }
      // Else if (S1 && S2), mark as sporadic - Else mark as normal
      else {
        characteristicVec.isSporadic = checkIfSporadic(characteristicVec);
        newGridList.insert(std::make_pair(grid, characteristicVec));
      }
    }
    // Else if (S1 && S2), mark as sporadic
    else {
      characteristicVec.isSporadic = checkIfSporadic(characteristicVec);
      newGridList.insert(std::make_pair(grid, characteristicVec));
    }
  }
  mergeGridList(gridList, newGridList);

  // SESAME_INFO(" - Removed "<<removeGridList.size()<<" grids from
  // grid_list.");
  for (DensityGrid &sporadicGrid : removeGridList) {
    this->gridList.erase(sporadicGrid);
    for (auto &cluster : this->clusterList) {
      if (cluster.grids.find(sporadicGrid) != cluster.grids.end()) {
        cluster.grids.erase(sporadicGrid);
      }
    }
    for (auto &cluster : this->newClusterList) {
      if (cluster.grids.find(sporadicGrid) != cluster.grids.end()) {
        cluster.grids.erase(sporadicGrid);
        break;
      }
    }
  }
}

/**
 * Determines whether a sparse density grid is sporadic using rules S1 and S2 of
 * Chen and Tu 2007
 *
 * @param characteristicVec - the CharacteristicVector of the density grid being
 * assessed for sporadicity
 */
bool SESAME::V9::checkIfSporadic(CharacteristicVector characteristicVec) {
  // Check S1
  if (characteristicVec.getCurrGridDensity(currentTimeStamp, param.lambda) <
      param.outlier_cap) {
    // Check S2 TODO CHANGE REMOVE TIME FROM 0 TO -1
    if (characteristicVec.removeTime == 0 ||
        (currentTimeStamp -
         ((1 + param.beta) * characteristicVec.removeTime)) >= 0)
      return true;
  }

  return false;
}