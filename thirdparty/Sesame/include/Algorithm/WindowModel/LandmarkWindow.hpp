// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#ifndef SESAME_SRC_ALGORITHM_WINDOWMODEL_LANDMARKWINDOW_HPP_
#define SESAME_SRC_ALGORITHM_WINDOWMODEL_LANDMARKWINDOW_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/DataStructure/Snapshot.hpp"
#include "Algorithm/DataStructure/TreeNode.hpp"
#include "Algorithm/WindowModel/WindowModel.hpp"
#include "Timer/TimeMeter.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace SESAME {
class LandmarkWindow;
typedef std::shared_ptr<LandmarkWindow> LandmarkWindowPtr;

class LandmarkWindow : WindowModel {
public:
  class CoresetTree;
  typedef std::shared_ptr<CoresetTree> CoresetTreePtr;

  class CoresetTree {
  private:
    param_t param;

  public:
    CoresetTree(const param_t &param) : param(param) {}
    /**
     * initalizes root as a treenode with the union of setA and setB as pointset
     * and centre as centre
     * @param root
     * @param setA
     * @param setB
     * @param n_1
     * @param n_2
     * @param centre
     * @param centreIndex
     */
    void constructRoot(TreeNodePtr root, std::vector<PointPtr> &setA,
                       std::vector<PointPtr> &setB, int n_1, int n_2,
                       PointPtr centre, int centreIndex);

    /**
    Constructs a coreset of size k from the union of setA and setB
    **/
    void unionTreeCoreset(int k, int n_1, int n_2, std::vector<PointPtr> &setA,
                          std::vector<PointPtr> &setB,
                          std::vector<PointPtr> &centres);
    void freeTree(TreeNodePtr root);
    bool treeFinished(TreeNodePtr root);
    bool isLeaf(TreeNodePtr node);

    /**
     * Computes the target function value of the n points of the treenode.
     * Differs from the function "targetFunctionValue" in three things:
     * 1. only the centre of the treenode is used as a centre
     * 2. works on arrays of pointers instead on arrays of points
     * 3. stores the cost in the treenode
     * @param node
     */
    void treeNodeTargetFunctionValue(TreeNodePtr node);

    /**
    selects a leaf node (using the kMeans++ distribution)
    **/
    TreeNodePtr selectNode(TreeNodePtr root);
    PointPtr chooseCentre(TreeNodePtr node);
    double treeNodeCostOfPoint(TreeNodePtr node, PointPtr p);
    double treeNodeSplitCost(TreeNodePtr node, PointPtr CenterA,
                             PointPtr CenterB);
    void split(TreeNodePtr parent, PointPtr newCentre, int newCentreIndex);
    PointPtr determineClosestCentre(PointPtr point, PointPtr centreA,
                                    PointPtr centreB);
  };
  /**
  DataStructure representing a single window
  **/
  struct Window {
    int cursize;
    std::vector<PointPtr> points;
    std::vector<PointPtr> spillover;
  };

  /**
  datastructure for managing all O(log(n)) windows
  **/
  struct WindowManager {
    int numberOfWindow;
    int maxWindowSize;
    std::vector<Window> windows; //    struct Window *windows;
  };

  WindowManager windowManager;
  TimeMeter timerMeter;
  CoresetTreePtr tree;
  /**
   * initialize windows in the window manager.
   * @param dim
   * @param coreset_size
   */
  void initWindow(int num);

  /**
  inserts a single point into the bucketmanager
  **/
  void insertPoint(PointPtr point);

  /**
  It may happen that the manager is not full (since n is not always a power of
  2). In this case we extract the coreset from the manager by computing a
  coreset of all nonempty windows

  Case 1: the last bucket is full
  => n is a power of 2 and we return the contents of the last bucket

  Case2: the last bucket is not full
  => we compute a coreset of all nonempty windows

  this operation should only be called after the streaming process is sourceEnd
  **/
  std::vector<PointPtr> getCoresetFromManager(
      std::vector<PointPtr> &
          coreset); // https://stackoverflow.com/questions/15704565/efficient-way-to-return-a-stdvector-in-c

  /**
  Pyramidal time frame: Taking snapshots of online Micro clusters in Clustream
  algorithms
  **/

  /**
  Data Structure representing all orders of snapshots
   **/
  SESAME::QueueOrderSnapshot orderSnapShots;
  struct PyramidalWindow {
    unsigned int time_interval; // time interval of pyramidal window
    unsigned int currentOrder;  // the biggest order T of snapshots
  };
  PyramidalWindow pyramidalWindow;

  /**
   * @Description: init the pyramidal window, pass the value of startTime and
   * reset shared_ptr Arrayqueue of snapshots
   * @Param: startTime: the start time point of algorithm
   * time_interval: the time interval of online pyramidal window frame
   * @Return: void
   */
  void initPyramidalWindow(unsigned int time_interval);
  /**
   * @Description: this function is pyramidal window for Clustream,
   * which takes and stores the snapshots of micro clusters
   * @Param: microClusters: micro clusters' snapshot need to be stored
   * elapsedTime: the current elapsed time
   * @Return: void
   */
  void pyramidalWindowProcess(int elapsedTime,
                              const MicroClusters &microClusters);
  /**
   * @Description: this function stores snapshots into the pyramidal window data
   * structure
   * @Param: currentOrder: the ith order  snapshots stored into
   * microClusters: micro clusters' snapshot need to be stored
   * elapsedTime: the current elapsed time
   * @Return: void
   */
  void storeSnapshot(unsigned int currentOrder,
                     const MicroClusters &microClusters, int elapsedTime);
  void clearPyramidalWindow();
};

} // namespace SESAME

#endif // SESAME_SRC_ALGORITHM_WINDOWMODEL_LANDMARKWINDOW_HPP_
