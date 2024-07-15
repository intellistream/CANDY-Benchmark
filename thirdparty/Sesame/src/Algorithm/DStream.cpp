//
// Created by 1124a on 2021/10/27.
//

#include "Algorithm/DStream.hpp"
#include "Algorithm/DataStructure/DataStructureFactory.hpp"
#include "Algorithm/WindowModel/WindowFactory.hpp"

#include <cfloat>
#include <cmath>
#include <iostream>

using namespace std;

SESAME::DStream::DStream(param_t &cmd_params) { this->param = cmd_params; }

SESAME::DStream::~DStream() = default;

void SESAME::DStream::Init() {
  sum_timer.Tick();
  ds_timer.Tick();
  dampedWindow = WindowFactory::createDampedWindow(param.lambda, 1);
  gap = 1;
  dm = -1.0;
  dl = -1.0;
  NGrids = 1;
  minVals = std::vector<double>(param.dim, DBL_MAX);
  maxVals = std::vector<double>(param.dim, DBL_MIN);
  Coord = std::vector<int>(param.dim, 0);
  ds_timer.Tock();
}

void SESAME::DStream::RunOnline(PointPtr input) {
  this->currentTimeStamp = input->getIndex();
  ifReCalculate(input);
  GridListUpdate(Coord); // tempCoord
  // 5. If tc == gap, then initial clustering
  // and
  // 6. If tc mod gap == 0, then:
  // Detect and remove sporadic grids from grid_list then adjust clustering
  if (!init && currentTimeStamp >= gap) {
    // SESAME_INFO("Initial clustering! gap is "<<gap);
    initialClustering();
    init = true;
  }
  if (currentTimeStamp != 0 and currentTimeStamp % gap == 0) {
    // SESAME_INFO("adjust clustering!"<<currentTimeStamp);
    removeSporadic();
    adjustClustering();
    //    SESAME_INFO("Cluster number is "<<clusterList.size());
  }
  ds_timer.Tock();

  lat_timer.Add(input->toa);
}

void SESAME::DStream::RunOffline(DataSinkPtr sinkPtr) {
  cout << "num_grids: " << NGrids << endl;
  cout << "gap: " << gap << endl;
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  // SESAME_INFO(" cluster list size "<<clusterList.size());
  int cluID = 0;
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

void SESAME::DStream::ifReCalculate(PointPtr point) {
  bool recalculateN = false;
  for (int i = 0; i < param.dim; i++) {
    auto feature = point->getFeatureItem(i);
    if (feature > maxVals[i]) {
      maxVals[i] = feature;
      recalculateN = true;
    } else if (feature < minVals[i]) {
      minVals[i] = feature;
      recalculateN = true;
    }
    Coord[i] = point->getFeatureItem(i) / param.grid_width;
  }
  if (recalculateN)
    reCalculateParameter();
}
void SESAME::DStream::reCalculateParameter() {
  int curGridNumber = 1;
  for (int i = 0; i < param.dim; i++) {
    int gridNum = floor((maxVals[i] - minVals[i]) / param.grid_width);
    if (gridNum <= 0)
      gridNum = 1;
    curGridNumber = curGridNumber * gridNum;
  }
  double dlBack = param.cl / (curGridNumber * (1.0 - param.lambda));
  double dmBack = param.cm / (curGridNumber * (1.0 - param.lambda));
  if (dlBack < 0)
    return;
  else {
    this->dl = dlBack;
    this->dm = dmBack;
    this->NGrids = curGridNumber;
    // // SESAME_INFO(" dl = " << this->dl << ", dm = " << this->dm);
    // // SESAME_INFO("TOTAL GRIDS ARE "<<this->NGrids);
    // Calculate the value for gap using the method defined in eq 26 of Chen and
    // Tu 2007
    double optionA = param.cl / param.cm;
    double optionB =
        ((double)this->NGrids - param.cm) / ((double)this->NGrids - param.cl);
    auto de = log(param.lambda);
    auto nu = log(max(optionA, optionB));
    auto res = (int)floor(nu / de);
    // Ensure that gap is not zero (i.e. if the procedure to calculate gap
    // rounds down to zero, then set gap to 1 and adjust clustering every
    // instance)
    if (res > 1) {
      gap = res;
    }
  }
}

/* Update the grid list of DStream when data inserting into the grid
 * */
void SESAME::DStream::GridListUpdate(std::vector<int> coordinate) {
  CharacteristicVector characteristicVec;
  DensityGrid grid(coordinate);
  // 3. If (g not in grid_list) insert dg to grid_list
  auto it = this->gridList.find(grid);
  if (it == gridList.end()) { // SESAME_INFO("Insert "<< currentTimeStamp);
    auto it2 = this->deletedGrids.find(grid);
    if (it2 != deletedGrids.end()) {
      characteristicVec = CharacteristicVector(currentTimeStamp, it2->second,
                                               1.0, -1, false, dl, dm);
      this->deletedGrids.erase(grid);
    }

    else
      characteristicVec =
          CharacteristicVector(currentTimeStamp, -1, 1.0, -1, false, dl, dm);
    // this->gridList.insert(std::make_pair(grid, characteristicVec));
    this->gridList.insert(std::make_pair(grid, characteristicVec));
  }
  // 4. Update the characteristic vector of dg
  else {
    // SESAME_INFO("Update Grid");
    ds_timer.Tock();
    win_timer.Tick();
    characteristicVec = it->second;
    characteristicVec.densityWithNew(currentTimeStamp, param.lambda);
    characteristicVec.updateTime = currentTimeStamp;
    it->second = characteristicVec;
    win_timer.Tock();
    ds_timer.Tick();
  }
}

/**
 * Implements the procedure given in Figure 3 of Chen and Tu 2007
 */
void SESAME::DStream::initialClustering() {
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
      // SESAME_INFO(" was dense (class "<<gridClass<<")" <<"Position is
      // "<<this->clusterList.size());
    } else
      characteristicVecOfG.label = NO_CLASS;
    // newGridList.insert(std::make_pair(grid, characteristicVecOfG));
    // newGridList= putHashMap(newGridList,grid, characteristicVecOfG);

    auto it1 = newGridList.find(grid);
    if (it1 != newGridList.end())
      it1->second = characteristicVecOfG;
    else
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
bool SESAME::DStream::adjustLabels() {
  // bool adjust=false;
  // a. For each cluster c

  for (GridCluster &gridCluster : this->clusterList) {
    // // SESAME_INFO("Adjusting from cluster "<<gridCluster.clusterLabel<<",
    // standby..."); b. for each grid, dg, of c
    for (auto gridIter = gridCluster.grids.begin();
         gridIter != gridCluster.grids.end(); gridIter++) {
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
            // ...and if neighbouring grid isn't already in the same cluster as
            // grid...
            if (class1 != class2) {
              // If neighbouring grid is in cluster c', merge c and c' into the
              // larger of the two
              if (class2 != NO_CLASS) {
                // SESAME_INFO(". Cluster number is "<<clusterList.size()<<
                // "Neighbour 1 and 2 are "<<class1<<" "<<class2);
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
                ////TODO CHECK HERE
                characteristicVec2.label = class1;
                gridCluster.addGrid(gridNeighbourhood);
                this->clusterList.at(class1) = gridCluster; // Testing
                // putHashMap(gridList,grid,characteristicVec2);
                auto it1 = gridList.find(grid);
                if (it1 != gridList.end())
                  it1->second = characteristicVec2;
                else
                  gridList.insert(std::make_pair(grid, characteristicVec2));

                return true;
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
void SESAME::DStream::updateGridListDensity() {
  // // SESAME_INFO("grid list size is "<<this->gridList.size());
  for (auto &iter : this->gridList) {
    iter.second.isVisited = false;
    iter.second.UpdateAllDensity(currentTimeStamp, param.lambda, dl, dm);
  }
}

/**
 * Performs the periodic adjustment of clusters every 'gap' timesteps.
 * Implements the procedure given in Figure 4 of Chen and Tu 2007
 *
 * @see moa.clusterers.DStream.DStream#gap
 */
void SESAME::DStream::adjustClustering() {
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
bool SESAME::DStream::inspectChangedGrids() {
  HashMap newGridList;
  auto gridIter = this->gridList.begin();
  int a = 0;
  while (gridIter != gridList.end() &&
         newGridList.empty()) //&& newGridList.empty()
  {
    const DensityGrid &grid = gridIter->first;
    const CharacteristicVector &characteristicVec = gridIter->second;
    int gridClass = characteristicVec.label;
    if (characteristicVec.attChange &&
        !characteristicVec.isVisited) // grid.isVisited
    {                                 // grid.isVisited=true;
      gridIter->second.isVisited = true;
      // SESAME_INFO(a<<"th visit! Whether visited
      // "<<gridIter->second.isVisited);
      newGridList.insert(std::make_pair(grid, characteristicVec));
      auto it1 = gridList.find(grid);
      if (it1 != gridList.end())
        it1->second = characteristicVec;
      else
        gridList.insert(std::make_pair(grid, characteristicVec));
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
    gridIter++;
    a++;
  }
  // SESAME_INFO("Inspect changes in grids "<<gridList.size());
  //  If there are grids in new grid list, update the corresponding grids in
  //  grid_list and clean up the cluster list
  if (!newGridList.empty()) {
    // SESAME_INFO("There are "<<newGridList.size()<<" entries to update from
    // new Grid List to grid_list.");
    mergeGridList(this->gridList, newGridList);
    // SESAME_INFO("Now grid size is  "<<gridList.size());
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
SESAME::HashMap SESAME::DStream::adjustForSparseGrid(
    DensityGrid grid, CharacteristicVector characteristicVec, int gridClass) {
  if (gridClass != -1 && gridClass > clusterList.size()) {
    SESAME_INFO("Current grids size and cluster size is "
                << gridList.size() << " and " << clusterList.size()
                << " adjust For Sparse Grid " << gridClass << ".");
  }

  HashMap newGridList;
  // System.out.print("Density grid "+dg.toString()+" is adjusted as a sparse
  // grid at time
  // "+this.getCurrTime()+". ");
  if (gridClass != NO_CLASS) {
    for (auto gridCluster : clusterList) {
      if (gridCluster.clusterLabel == gridClass) { ////TODO CHECK HERE
        gridCluster.removeGrid(grid);
        characteristicVec.label = NO_CLASS;
        // newGridList.insert(std::make_pair(grid, characteristicVec));
        // newGridList=putHashMap(newGridList,grid, characteristicVec);
        auto it1 = newGridList.find(grid);
        if (it1 != newGridList.end())
          it1->second = characteristicVec;
        else
          newGridList.insert(std::make_pair(grid, characteristicVec));
        if (!gridCluster.grids.empty() && !gridCluster.isConnected())
          mergeGridList(newGridList, reCluster(gridCluster));
        break;
      }
    }
  }
  // else
  // System.out.println("It was not clustered ("+dgClass+").");
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
SESAME::HashMap SESAME::DStream::reCluster(GridCluster gridCluster) {
  HashMap newGridList;
  auto gcIter = gridCluster.grids.begin();
  newClusterList = std::vector<GridCluster>();
  // SESAME_INFO("ReCluster called for cluster "<<gridCluster.clusterLabel);
  //  Assign every dense grid in gc to its own cluster, assign all other grids
  //  to NO_CLASS
  while (gcIter != gridCluster.grids.end()) {
    DensityGrid grid = gcIter->first;
    CharacteristicVector characteristicVecOfGrid =
        this->gridList.find(grid)->second;
    if (characteristicVecOfGrid.attribute == DENSE) {
      int gridClass = (int)newClusterList.size();
      SESAME_INFO("grid class for the new cluster list size! " << gridClass);
      ////TODO CHECK HERE
      characteristicVecOfGrid.label = gridClass;
      GridCluster newCluster(gridClass);
      newCluster.addGrid(grid);
      newClusterList.push_back(newCluster);
    } else
      characteristicVecOfGrid.label = NO_CLASS;
    // newGridList.insert(std::make_pair(grid, characteristicVecOfGrid));
    // newGridList=putHashMap(newGridList,grid, characteristicVecOfGrid);
    auto it1 = newGridList.find(grid);
    if (it1 != newGridList.end())
      it1->second = characteristicVecOfGrid;
    else
      newGridList.insert(std::make_pair(grid, characteristicVecOfGrid));
    gcIter++;
  }

  bool changesMade;
  // While changes can be made...
  do {
    changesMade = false;
    HashMap gridListAdjusted = adjustNewLabels(newGridList);
    if (!gridListAdjusted.empty()) {
      SESAME_INFO("grid list is adjusted for sparse!");
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

SESAME::HashMap SESAME::DStream::adjustNewLabels(SESAME::HashMap newGridList) {
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
                ////TODO CHECK HERE
                characteristicVec2.label = class1;
                cluster1.addGrid(neighbourGrid);
                this->newClusterList.at(class1) = cluster1;
                // Change for detecting repeated grids

                auto it1 = gridListAdjusted.find(neighbourGrid);
                if (it1 != gridListAdjusted.end())
                  it1->second = characteristicVec2;
                else
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
 * @return a HashMap containing density grids for update after this iteration
 */
SESAME::HashMap SESAME::DStream::adjustForDenseGrid(
    DensityGrid grid, CharacteristicVector characteristicVec, int gridClass) {
  if (gridClass != -1 && gridClass > clusterList.size()) {
    SESAME_INFO("Current grids size and cluster size is "
                << gridList.size() << " and " << clusterList.size()
                << " adjust For Dense Grid class is " << gridClass << ".");
  }
  //  SESAME_INFO("Current grids size and cluster size is "
  //               <<  gridList.size()<<" and "<<clusterList.size()<<
  //            " adjust For Dense Grid class is "<<gridClass<<".");
  // Among all neighbours of dg, find the grid h whose cluster ch has the
  // largest size
  GridCluster gridCluster; // The cluster, ch, of h
  DensityGrid gridChosen(
      grid); // The chosen grid h, whose cluster ch has the largest size
  double ChosenGridSize = -1.0; // The size of gridCluster, the largest cluster
  int hClass = NO_CLASS;        // The class label of h
  int hChosenClass = NO_CLASS;  // The class label of ch

  HashMap newGridList;

  for (DensityGrid &neighbourGrid :
       grid.getNeighbours()) // The neighbour of g being considered
  {
    if (this->gridList.find(neighbourGrid) != gridList.end()) {
      hClass = this->gridList.find(neighbourGrid)->second.label;
      if (hClass != NO_CLASS) {
        for (auto gridC : clusterList) {
          if (gridC.clusterLabel == hClass) {
            if (gridC.grids.size() > ChosenGridSize) {
              ChosenGridSize = gridC.grids.size();
              hChosenClass = hClass;
              gridChosen = DensityGrid(neighbourGrid);
            }
            break;
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
        // newGridList.insert(std::make_pair(grid, characteristicVec));
        // newGridList=putHashMap(newGridList,grid, characteristicVec);
        auto it1 = newGridList.find(grid);
        if (it1 != newGridList.end())
          it1->second = characteristicVec;
        else
          newGridList.insert(std::make_pair(grid, characteristicVec));
        gridCluster.addGrid(grid);
        this->clusterList.at(hChosenClass) = gridCluster;
      }
      // Else if dg belongs to cluster c and h belongs to c'
      else {
        // SESAME_INFO("grid was labelled "<<gridClass<<", the total size for
        // cluster is
        // "<<this->clusterList.size());
        for (auto gridCluster : clusterList) {
          if (gridCluster.clusterLabel == gridClass) {
            double gSize = this->clusterList.at(gridClass).grids.size();

            if (gSize <= ChosenGridSize)
              mergeClusters(gridClass, hChosenClass);
            else
              mergeClusters(hChosenClass, gridClass);
            break;
          }
        }
      }
    }

    // Else if h is a transitional grid
    else if (this->gridList.find(gridChosen)->second.attribute ==
             TRANSITIONAL) {
      //  SESAME_INFO("h is transitional.");
      // If dg is labelled as no class and if h is an outside grid if dg is
      // added to ch
      if (gridClass == NO_CLASS && !gridCluster.isInside(gridChosen, grid)) {
        ////TODO CHECK HERE
        characteristicVec.label = hChosenClass;
        // newGridList.insert(std::make_pair(grid, characteristicVec));
        //  newGridList=putHashMap(newGridList,grid, characteristicVec);
        auto it1 = newGridList.find(grid);
        if (it1 != newGridList.end())
          it1->second = characteristicVec;
        else
          newGridList.insert(std::make_pair(grid, characteristicVec));
        gridCluster.addGrid(grid);
        this->clusterList.at(hChosenClass) = gridCluster;
        // SESAME_INFO(" dg is added to cluster "<<hChosenClass<<".");
      }
      // Else if dg is in cluster c and |c| >= |ch|
      else if (gridClass != NO_CLASS) {
        //  SESAME_INFO("Finding this cluster......");
        for (auto c : clusterList) {
          if (c.clusterLabel == gridClass) {
            // GridCluster c = this->clusterList.at(gridClass);
            int gSize = c.grids.size();
            if (gSize >= ChosenGridSize) {
              // Move h from cluster ch to cluster c
              gridCluster.removeGrid(gridChosen);
              c.addGrid(gridChosen);
              CharacteristicVector cvhChosen =
                  this->gridList.find(gridChosen)->second;
              ////TODO CHECK HERE
              cvhChosen.label = gridClass;
              // newGridList.insert(std::make_pair(gridChosen, cvhChosen));
              //  newGridList=putHashMap(newGridList,gridChosen, cvhChosen);

              auto it1 = newGridList.find(gridChosen);
              if (it1 != newGridList.end())
                it1->second = cvhChosen;
              else
                newGridList.insert(std::make_pair(gridChosen, cvhChosen));
              // SESAME_INFO("dgClass is "<<gridClass<<", hChosenClass
              // is"<<hChosenClass<<", gSize is "<<gSize<<
              //     " and hChosenSize is"<<ChosenGridSize<<" h is added to
              //     cluster
              //     "<<gridClass<<".");
              this->clusterList.at(hChosenClass) = gridCluster;
              this->clusterList.at(gridClass) = c;
            }
          }
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
    ////TODO CHECK HERE
    characteristicVec.label = newClass;
    // newGridList= putHashMap(newGridList,grid, characteristicVec);
    auto it1 = newGridList.find(gridChosen);
    if (it1 != newGridList.end())
      it1->second = characteristicVec;
    else
      newGridList.insert(std::make_pair(gridChosen, characteristicVec));
    // Iterate through the neighbourhood until no more transitional neighbours
    // can be added (dense neighbours will add themselves as part of their
    // adjust process)
    for (DensityGrid &dghprime : grid.getNeighbours()) {
      if (this->gridList.find(dghprime) != this->gridList.end() &&
          c.grids.find(dghprime) != c.grids.end()) {
        CharacteristicVector cvhprime = this->gridList.find(dghprime)->second;
        if (cvhprime.attribute == TRANSITIONAL) {
          c.addGrid(dghprime);
          ////TODO CHECK HERE
          cvhprime.label = newClass;
          // newGridList.insert(std::make_pair(dghprime, cvhprime));
          // newGridList=putHashMap(newGridList,dghprime, cvhprime);
          auto it1 = newGridList.find(dghprime);
          if (it1 != newGridList.end())
            it1->second = cvhprime;
          else
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
SESAME::HashMap SESAME::DStream::adjustForTransitionalGrid(
    DensityGrid grid, CharacteristicVector characteristicVec, int gridClass) {
  // Among all neighbours of dg, find the grid h whose cluster ch has the
  // largest size and satisfies that dg would be an outside grid if added to it
  GridCluster gridCluster;     // The cluster, ch, of h
  double hChosenSize = 0.0;    // The size of ch, the largest cluster
  DensityGrid neighbourGrid;   // The neighbour of dg being considered
  int hClass = NO_CLASS;       // The class label of h
  int hChosenClass = NO_CLASS; // The class label of ch
  HashMap newGridList;
  if (gridClass != -1 && gridClass > clusterList.size()) {
    SESAME_INFO("Current grids size and cluster size is "
                << gridList.size() << " and " << clusterList.size()
                << " adjust For Transitional Grid " << gridClass << ".");
  }

  for (DensityGrid &neighbourGrid : grid.getNeighbours()) {
    if (this->gridList.find(neighbourGrid) != gridList.end()) {
      hClass = this->gridList.find(neighbourGrid)->second.label;
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

  // System.out.println(" Chosen neighbour is from cluster "+hChosenClass+",
  // dgClass is
  // "+dgClass+".");

  if (hChosenClass != NO_CLASS && hChosenClass != gridClass) {
    for (auto gridCluster : clusterList) {
      if (gridCluster.clusterLabel == hChosenClass) {
        gridCluster.addGrid(grid);
        this->clusterList.at(hChosenClass) = gridCluster;
      }
    }

    if (gridClass != NO_CLASS) {
      GridCluster cluster = this->clusterList.at(gridClass);
      cluster.removeGrid(grid);
      this->clusterList.at(gridClass) = cluster;
    }
    ////TODO CHECK HERE
    characteristicVec.label = hChosenClass;
    auto it1 = newGridList.find(grid);
    if (it1 != newGridList.end())
      it1->second = characteristicVec;
    else
    // newGridList.insert(std::make_pair(grid, characteristicVec));
    //  newGridList= putHashMap(newGridList,grid, characteristicVec);
    {
      auto it1 = newGridList.find(grid);
      if (it1 != newGridList.end())
        it1->second = characteristicVec;
      else
        newGridList.insert(std::make_pair(grid, characteristicVec));
    }
  }

  return newGridList;
}

SESAME::HashMap SESAME::DStream::mergeNewClusters(SESAME::HashMap newGridList,
                                                  int smallCluster,
                                                  int bigCluster) {
  // System.out.println("Merge new clusters "+smallCluster+" and
  // "+bigCluster+"."); Iterate through the density grids in grid_list to find
  // those which are in highClass
  for (HashMap::iterator gridIter = newGridList.begin();
       gridIter != gridList.end(); gridIter++) {
    DensityGrid grid = gridIter->first;
    CharacteristicVector characteristicVec = gridIter->second;

    // Assign density grids in small Cluster to bigCluster
    if (characteristicVec.label == smallCluster) {
      ////TODO CHECK HERE
      characteristicVec.label = bigCluster;
      auto it1 = newGridList.find(grid);
      if (it1 != newGridList.end())
        it1->second = characteristicVec;
      else
        // newGridList.insert(std::make_pair(grid, characteristicVec));
        //  newGridList=   putHashMap(newGridList,grid, characteristicVec);
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
 * Determines whether a sparse density grid is sporadic using rules S1 and S2 of
 * Chen and Tu 2007
 *
 * @param characteristicVec - the CharacteristicVector of the density grid being
 * assessed for sporadicity
 */
bool SESAME::DStream::checkIfSporadic(CharacteristicVector characteristicVec) {
  // Check S1
  if (characteristicVec.getCurrGridDensity(currentTimeStamp, param.lambda) <
      outlier_density_thresholdFunction(characteristicVec.densityUpdateTime,
                                        param.cl, param.lambda, this->NGrids)) {
    // Check S2 TODO CHANGE REMOVE TIME FROM 0 TO -1
    if (characteristicVec.removeTime == 0 ||
        (currentTimeStamp -
         ((1 + param.beta) * characteristicVec.removeTime)) >= 0)
      return true;
  }

  return false;
}

/**
 * Implements the function pi given in Definition 4.1 of Chen and Tu 2007
 *
 * @param tg - the update time in the density grid's characteristic vector
 * @param cl - user defined parameter which controls the threshold for sparse
 * grids
 * @param lambda - see lambda definition
 * @param NGrids - the number of density grids,
 */
double SESAME::DStream::outlier_density_thresholdFunction(int tg, double cl,
                                                          double lambda,
                                                          int NGrids) {
  return (cl * (1.0 - pow(lambda, ((double)(currentTimeStamp - tg) + 1.0)))) /
         (NGrids * (1.0 - lambda));
}

/**
 * Reassign all grids belonging in the small cluster to the big cluster
 * Merge the GridCluster objects representing each cluster
 *
 * @param smallCluster - the index of the smaller cluster
 * @param bigCluster - the index of the bigger cluster
 */
void SESAME::DStream::mergeClusters(int smallCluster, int bigCluster) {
  //  SESAME_INFO("Merge clusters "<<smallCluster<<" and "<<bigCluster<<".");
  // Iterate through the density grids in grid_list to find those which are in
  // highClass
  for (auto gridIter = gridList.begin(); gridIter != gridList.end();
       gridIter++) {
    DensityGrid grid = gridIter->first;
    CharacteristicVector characteristicVec = gridIter->second;

    // Assign density grids in smallCluster to bigCluster
    if (characteristicVec.label == smallCluster) {
      ////TODO CHECK HERE
      characteristicVec.label = bigCluster;
      gridIter->second = characteristicVec;
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

void SESAME::DStream::mergeGridList(HashMap &thisGridList,
                                    const HashMap &otherList) {
  // if(otherList.size()>0) SESAME_INFO("Merge grid list! size this and other is
  // "<< thisGridList.size()<<" "<<otherList.size());
  for (auto &gridIter : otherList) {
    auto it1 = thisGridList.find(gridIter.first);
    if (it1 != thisGridList.end())
      thisGridList[gridIter.first] = gridIter.second;

    else
      thisGridList.insert(std::make_pair(gridIter.first, gridIter.second));
  }
}

SESAME::HashMap SESAME::DStream::cleanNewClusters(SESAME::HashMap newGridList) {
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
    ////TODO CHECK HERE
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

void SESAME::DStream::cleanClusters() {
  // SESAME_INFO("Clean Clusters");

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
    ////TODO CHECK HERE
    cluster.clusterLabel = index;
    unordered_map<DensityGrid, bool, GridKeyHash, EqualGrid> removeGrids;
    for (auto &gridOfCluster : cluster.grids) {
      DensityGrid grid = gridOfCluster.first;
      if (gridList.find(grid) != gridList.end())
        ////TODO CHECK HERE
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
void SESAME::DStream::removeSporadic() {
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
        // newGridList.insert(std::make_pair(grid, characteristicVec));
        // newGridList= putHashMap(newGridList,grid, characteristicVec);
        auto it1 = newGridList.find(grid);
        if (it1 != newGridList.end())
          it1->second = characteristicVec;
        else
          newGridList.insert(std::make_pair(grid, characteristicVec));
      }
    }
    // Else if (S1 && S2), mark as sporadic
    else {
      characteristicVec.isSporadic = checkIfSporadic(characteristicVec);
      // newGridList.insert(std::make_pair(grid, characteristicVec));
      //  newGridList=  putHashMap(newGridList,grid, characteristicVec);
      auto it1 = newGridList.find(grid);
      if (it1 != newGridList.end())
        it1->second = characteristicVec;
      else
        newGridList.insert(std::make_pair(grid, characteristicVec));
    }
  }
  mergeGridList(gridList, newGridList);

  // SESAME_INFO(" - Removed "<<removeGridList.size()<<" grids from
  // grid_list.");
  for (DensityGrid &sporadicGrid : removeGridList) {
    this->deletedGrids.insert(std::make_pair(sporadicGrid, currentTimeStamp));
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

/*
HashMap SESAME::DStream::putHashMap(HashMap gList, const DensityGrid& g,
CharacteristicVector cv)
{
    auto it1 = gList.find(g);
    if (it1 != gList.end())
        it1->second = cv;
    else
     gList.insert(std::make_pair(g, cv));
    return gList;
}*/