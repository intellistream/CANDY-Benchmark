//
// Created by 1124a on 2021/10/27.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_GRIDCLUSTER_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_GRIDCLUSTER_HPP_
#include <Algorithm/DataStructure/DensityGrid.hpp>
#include <algorithm>
#include <functional>
#include <unordered_map>
namespace SESAME {
class GridCluster;
typedef std::unordered_map<DensityGrid, bool, GridKeyHash, EqualGrid> HashGrids;
class GridCluster {
public:
  HashGrids grids;
  HashGrids visited;
  int clusterLabel;
  // Initialize
  GridCluster(int label);
  GridCluster();
  GridCluster(HashGrids hashMap, int label);
  /**
   * @param grid the density grid to add to the cluster
   */
  void addGrid(const DensityGrid &grid);

  /**
   * @param dg the density grid to remove from the cluster
   */
  void removeGrid(const DensityGrid &grid);

  /**
   * @param gridClus the GridCluster to be absorbed into this cluster
   */
  void absorbCluster(GridCluster gridCluster);
  /**
   * Inside Grids are defined in Definition 3.5 of Chen and Tu 2007 as:
   * Consider a grid group G and a grid g ∈ G, suppose g =(j1, ··· ,jd), if g
   * has neighboring grids in every dim i =1, ·· · ,d, then g is an inside grid
   * in G.Otherwise g is an outside grid in G.
   *
   * @param grid the density grid to label as being inside or out
   * @return TRUE if g is an inside grid, FALSE otherwise
   */
  bool isInside(DensityGrid grid);

  /**
   * Inside Grids are defined in Definition 3.5 of Chen and Tu 2007 as:
   * Consider a grid group G and a grid g ∈ G, suppose g =(j1, ··· ,jd), if g
   * has neighboring grids in every dim i =1, ·· · ,d, then g is an inside grid
   * in G. Otherwise g is an outside grid in G.
   *
   * @param grid the density grid being labelled as inside or outside
   * @param other the density grid being proposed for addition
   * @return TRUE if g would be an inside grid, FALSE otherwise
   */
  bool isInside(DensityGrid grid, DensityGrid other);

  /**
   * add a grid into grids, if exists, update value, if not, insert
   */
  void putHashGrid(HashGrids grids1, const DensityGrid &g, bool inside);

  /**
   * Tests a grid cluster for connectedness according to Definition 3.4, Grid
   * Group, from Chen and Tu 2007.
   *
   * Selects one density grid in the grid cluster as a starting point and
   * iterates repeatedly through its neighbours until no more density grids in
   * the grid cluster can be visited.
   *
   * @return TRUE if the cluster represent one single grid group; FALSE
   * otherwise.
   */

  bool isConnected();

  /**
   * Iterates through the DensityGrids in the cluster and calculates the
   * inclusion probability for each.
   *
   * @return 1.0 if instance matches any of the density grids; 0.0 otherwise.
   */
  double getInclusionProb(Point point);
  bool operator==(GridCluster &Other) const;
};

} // namespace SESAME

#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_GRIDCLUSTER_HPP_
