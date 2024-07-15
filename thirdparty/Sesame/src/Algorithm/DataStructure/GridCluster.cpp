//
// Created by 1124a on 2021/10/27.
//

#include <Algorithm/DataStructure/GridCluster.hpp>
#include <Utils/Logger.hpp>

SESAME::GridCluster::GridCluster() {}
SESAME::GridCluster::GridCluster(int label) { this->clusterLabel = label; }
// TODO: if Using this function, be careful when grids are not NULL
SESAME::GridCluster::GridCluster(HashGrids hashMap, int label) {
  HashGrids::iterator iterW;
  for (iterW = hashMap.begin(); iterW != hashMap.end(); iterW++) {
    DensityGrid grid = iterW->first;
    bool inside = iterW->second;
    // this->grids.insert(std::make_pair(grid, inside));
    putHashGrid(this->grids, grid, inside);
  }
  this->clusterLabel = label;
}

/**
 * @param grid the density grid to add to the cluster
 */

void SESAME::GridCluster::addGrid(const DensityGrid &grid) {
  bool inside = isInside(grid);
  auto it1 = grids.find(grid);
  if (it1 != grids.end())
    grids[grid] = inside;
  else
    grids.insert(std::make_pair(grid, inside));
  // Iterate on grids and judge whether they are inside grids or not
  for (auto &grid_iter : this->grids) {
    bool inside2U = grid_iter.second;
    if (!inside2U) {
      DensityGrid dg2U = grid_iter.first;
      grid_iter.second = isInside(dg2U);
    }
  }
}

/**
 * @param dg the density grid to remove from the cluster
 */
void SESAME::GridCluster::removeGrid(const DensityGrid &grid) {
  this->grids.erase(grid);
}

/**
 * @param gridClus the GridCluster to be absorbed into this cluster
 */
void SESAME::GridCluster::absorbCluster(GridCluster gridCluster) {
  bool inside;
  SESAME::HashGrids newCluster;
  // SESAME_INFO("Absorb cluster "<< gridCluster.clusterLabel <<" into cluster
  // "<<this->clusterLabel<<".");

  // Add each density grid from gridCluster into this->grids
  auto grid = gridCluster.grids.begin();
  while (grid != gridCluster.grids.end()) {
    // TODO whether they have same grids?
    // this->grids.insert(std::make_pair(grid->first, false));
    putHashGrid(this->grids, grid->first, false);
    grid++;
  }
  // SESAME_INFO("...density grids added");
  // Determine which density grids in this.grids are 'inside' and which are
  // 'outside'
  auto thisGrid = this->grids.begin(); // mod
  while (thisGrid != this->grids.end()) {
    inside = isInside(thisGrid->first);
    putHashGrid(newCluster, thisGrid->first, inside);
    thisGrid++;
  }
  this->grids = newCluster;
  // SESAME_INFO("...inside/outside determined");
}

/**
 * Inside Grids are defined in Definition 3.5 of Chen and Tu 2007 as:
 * Consider a grid group G and a grid g ∈ G, suppose g =(j1, ··· ,jd), if g has
 * neighboring grids in every dim i =1, ·· · ,d, then g is an inside grid
 * in G.Otherwise g is an outside grid in G.
 *
 * @param grid the density grid to label as being inside or out
 * @return TRUE if g is an inside grid, FALSE otherwise
 */
bool SESAME::GridCluster::isInside(DensityGrid grid) {
  std::vector<DensityGrid> neighbour = grid.getNeighbours();
  for (auto gridNeighbourhood : neighbour) {
    if (this->grids.find(gridNeighbourhood) == this->grids.end()) {
      return false;
    }
  }
  return true;
}

/**
 * Inside Grids are defined in Definition 3.5 of Chen and Tu 2007 as:
 * Consider a grid group G and a grid g ∈ G, suppose g =(j1, ··· ,jd), if g has
 * neighboring grids in every dim i =1, ·· · ,d, then g is an inside grid
 * in G. Otherwise g is an outside grid in G.
 *
 * @param grid the density grid being labelled as inside or outside
 * @param other the density grid being proposed for addition
 * @return TRUE if g would be an inside grid, FALSE otherwise
 */
bool SESAME::GridCluster::isInside(DensityGrid grid, DensityGrid other) {
  std::vector<DensityGrid> neighbour = grid.getNeighbours();
  for (auto gridNeighbourhood : neighbour) {
    if (this->grids.find(gridNeighbourhood) != this->grids.end() &&
        gridNeighbourhood == other) {
      return false;
    }
  }
  return true;
}
/**
 * Tests a grid cluster for connectedness according to Definition 3.4, Grid
 * Group, from Chen and Tu 2007.
 *
 * Selects one density grid in the grid cluster as a starting point and iterates
 * repeatedly through its neighbours until no more density grids in the grid
 * cluster can be visited.
 *
 * @return TRUE if the cluster represent one single grid group; FALSE otherwise.
 */

bool SESAME::GridCluster::isConnected() {
  // TODO A little confused about here

  if (!this->grids.empty()) {
    DensityGrid grid = this->grids.begin()->first;
    putHashGrid(this->visited, grid, this->grids.begin()->second);
    bool changesMade;

    do {
      changesMade = false;
      auto visIter = this->visited.begin();
      HashGrids toAdd;

      while (visIter != this->visited.end() && toAdd.empty()) {
        DensityGrid dg2V = visIter->first;
        std::vector<DensityGrid> neighbour = dg2V.getNeighbours();
        for (auto dg2VNeighbourhood : neighbour) {
          if (this->grids.find(dg2VNeighbourhood) != this->grids.end() &&
              this->visited.find(dg2VNeighbourhood) == this->visited.end())
            putHashGrid(toAdd, dg2VNeighbourhood,
                        this->grids.find(dg2VNeighbourhood)->second);
        }
        visIter++;
      }

      if (!toAdd.empty()) {
        HashGrids::iterator gridToAdd;
        for (gridToAdd = toAdd.begin(); gridToAdd != toAdd.end(); gridToAdd++) {
          putHashGrid(this->visited, gridToAdd->first, gridToAdd->second);
        }
        changesMade = true;
      }
    } while (changesMade);
  }

  if (this->visited.size() == this->grids.size()) {
    // SESAME_INFO("The cluster is still connected. "<<this->visited.size()+" of
    // "<<this->grids.size()<<" reached.");
    return true;
  } else {
    // SESAME_INFO("The cluster is no longer connected.
    // "<<this.visited.size()<<" of
    // "+this.grids.size()+" reached.");
    return false;
  }
}

/**
 * Iterates through the DensityGrids in the cluster and calculates the inclusion
 * probability for each.
 *
 * @return 1.0 if instance matches any of the density grids; 0.0 otherwise.
 */
double SESAME::GridCluster::getInclusionProb(Point point) {
  HashGrids::iterator iterW;
  // Iterate on grids and judge whether they are inside grids or not
  for (iterW = this->grids.begin(); iterW != this->grids.end(); iterW++) {
    DensityGrid grid = iterW->first;
    if (grid.getInclusionProbability(point) == 1.0)
      return 1.0;
  }
  return 0.0;
}

bool SESAME::GridCluster::operator==(GridCluster &other) const {
  bool equal = false;
  if (clusterLabel == other.clusterLabel &&
      grids.size() == other.grids.size() &&
      visited.size() == other.visited.size())
    equal = true;
  return equal;
}

void SESAME::GridCluster::putHashGrid(HashGrids grids1, const DensityGrid &g,
                                      bool inside) {
  auto it1 = grids1.find(g);
  if (it1 != grids1.end())
    it1->second = inside;
  else
    grids1.insert(std::make_pair(g, inside));
}
