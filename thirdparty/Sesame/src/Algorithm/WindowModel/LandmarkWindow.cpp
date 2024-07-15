// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#include <Algorithm/DataStructure/CoresetTree.hpp>
#include <Algorithm/DataStructure/DataStructureFactory.hpp>
#include <Algorithm/WindowModel/LandmarkWindow.hpp>
#include <Utils/Logger.hpp>
#include <iterator>

/**
 * TODO: seems not meaningful? @Wangxin
 * @Description: create a blank window, initial the point in the window.
 * @param: dim
 * @param: windowSize
 */
void SESAME::LandmarkWindow::initWindow(int num) {
  int i;
  for (i = 0; i < this->windowManager.numberOfWindow; i++) {
    Window blankWindow;
    vector<PointPtr> p_list;
    vector<PointPtr> s_list;
    for (int j = 0; j < num; j++) {
      p_list.push_back(GenericFactory::New<Point>());
      s_list.push_back(GenericFactory::New<Point>());
    }
    blankWindow.points = p_list;
    blankWindow.spillover = s_list;
    blankWindow.cursize = 0;
    this->windowManager.windows.push_back(blankWindow);
  }
}

/**
 * TODO: extract the incremental computation progress from the whole function.
 * @Wangxin
 * @Description: insert every data point into the time window. If both of the
 * current original and spillover windows are full, construct a coreset tree
 * using all data and store the computed results tree nodes into the next
 * original window
 * @Param: Point: the single point data
 * @Return: void
 */
void SESAME::LandmarkWindow::insertPoint(PointPtr point) {
  // check if there is enough space in the first window
  int cursize = this->windowManager.windows[0].cursize;
  if (cursize >= this->windowManager.maxWindowSize) {
    //    SESAME_DEBUG("Window 0 is full");
    // start spillover process
    int curWindow = 0;
    int nextWindow = 1;

    // check if the next window is empty
    if (this->windowManager.windows[nextWindow].cursize == 0) {
      timerMeter.dataInsertAccMeasure();
      //      SESAME_DEBUG("Window " << nextWindow << " is not full, move window
      //      " << curWindow << " points to window "
      //                       << nextWindow);

      // if empty, copy the window
      int i;
      for (i = 0; i < this->windowManager.maxWindowSize; i++) {
        this->windowManager.windows[nextWindow].points[i] =
            this->windowManager.windows[curWindow].points[i]->copy();
      }
      // window is now full
      this->windowManager.windows[nextWindow].cursize =
          this->windowManager.maxWindowSize;
      // first window is now set empty
      this->windowManager.windows[curWindow].cursize = 0;
      cursize = 0;
      timerMeter.dataInsertEndMeasure();
    } else {
      timerMeter.dataInsertAccMeasure();

      // if next window is full
      //      SESAME_DEBUG(
      //          "Window " << nextWindow << " is full, move window " <<
      //          curWindow << " to spillover " << nextWindow);
      //      SESAME_DEBUG(
      //          "Window " << nextWindow << " is full");
      // copy the points in the current window to the next spillover and
      // continue
      int i;
      for (i = 0; i < this->windowManager.maxWindowSize; i++) {
        this->windowManager.windows[nextWindow].spillover[i] =
            this->windowManager.windows[curWindow].points[i]->copy();
      }
      this->windowManager.windows[0].cursize = 0;
      cursize = 0;
      curWindow++;
      nextWindow++;
      timerMeter.dataInsertEndMeasure();
      // as long as the next window is full output the coreset to the next
      // spillover, using points in the next window and spillover
      while (this->windowManager.windows[nextWindow].cursize ==
             this->windowManager.maxWindowSize) {
        timerMeter.clusterUpdateAccMeasure();

        //        SESAME_DEBUG("Window " << nextWindow
        //                         << " is full, Continue! construct the coreset
        //                         using points in window and spillover "
        //                         << curWindow << " and store it in the
        //                         spillover " << nextWindow);
        //        SESAME_DEBUG("Window " << nextWindow << " is full");
        this->tree->unionTreeCoreset(
            this->windowManager.maxWindowSize,
            this->windowManager.maxWindowSize,
            this->windowManager.maxWindowSize,
            this->windowManager.windows[curWindow].points,
            this->windowManager.windows[curWindow].spillover,
            this->windowManager.windows[nextWindow].spillover);
        // here we store the m constructed coreset points into the next
        // spillover current window now empty
        this->windowManager.windows[curWindow].cursize = 0;
        curWindow++;
        nextWindow++;
        timerMeter.clusterUpdateEndMeasure();
      }
      timerMeter.clusterUpdateAccMeasure();

      //      SESAME_DEBUG("Window " << nextWindow
      //                       << " is not full, End! construct the coreset
      //                       using points in window and spillover "
      //                       << curWindow << " and store the it in the last
      //                       spillover");
      // if the next window is empty, just do the same operation and store the
      // constructed coreset into the next spillover, now the next window is
      // empty but next spillover is full
      this->tree->unionTreeCoreset(
          this->windowManager.maxWindowSize, this->windowManager.maxWindowSize,
          this->windowManager.maxWindowSize,
          this->windowManager.windows[curWindow].points,
          this->windowManager.windows[curWindow].spillover,
          this->windowManager.windows[nextWindow].points);
      this->windowManager.windows[curWindow].cursize = 0;
      this->windowManager.windows[nextWindow].cursize =
          this->windowManager.maxWindowSize;
      timerMeter.clusterUpdateEndMeasure();
    }
  }
  timerMeter.dataInsertAccMeasure();
  // if the first window is not full, just insert point into it
  this->windowManager.windows[0].points[cursize] =
      point->copy(); //   .copy(point);
  this->windowManager.windows[0].cursize++;
  timerMeter.dataInsertEndMeasure();
}

/**
It may happen that the manager is not full (since n is not always a power of 2).
In this case we extract the coreset from the manager by computing a coreset of
all nonempty windows

Case 1: the last bucket is full
=> n is a power of 2 and we return the contents of the last bucket

Case2: the last bucket is not full
=> we compute a coreset of all nonempty windows

this operation should only be called after the streaming process is sourceEnd
**/
std::vector<SESAME::PointPtr>
SESAME::LandmarkWindow::getCoresetFromManager(std::vector<PointPtr> &coreset) {
  int i = 0;
  if (this->windowManager.windows[this->windowManager.numberOfWindow - 1]
          .cursize == this->windowManager.maxWindowSize) {
    coreset =
        this->windowManager.windows[this->windowManager.numberOfWindow - 1]
            .points;
  } else {
    // find the first nonempty bucket
    for (i = 0; i < this->windowManager.numberOfWindow; i++) {
      if (this->windowManager.windows[i].cursize ==
          this->windowManager.maxWindowSize) {
        coreset = this->windowManager.windows[i].points;
        break;
      }
    }
    // as long as there is a nonempty bucket compute a coreset
    int j;
    for (j = i + 1; j < this->windowManager.numberOfWindow; j++) {
      if (this->windowManager.windows[j].cursize != 0) {
        // output the coreset into the spillover of bucket j
        this->tree->unionTreeCoreset(this->windowManager.maxWindowSize,
                                     this->windowManager.maxWindowSize,
                                     this->windowManager.maxWindowSize,
                                     this->windowManager.windows[j].points,
                                     coreset,
                                     this->windowManager.windows[j].spillover);
        coreset = this->windowManager.windows[j].spillover;
      }
    }
  }
  return coreset;
}

/**
 * @Description: init the pyramidal window, pass the value of startTime and
 * reset shared_ptr Arrayqueue of snapshots
 * time_interval: the time interval of online pyramidal window frame
 * @Return: void
 */
void SESAME::LandmarkWindow::initPyramidalWindow(unsigned int time_interval) {
  QueueSnapshotPtr queueSnapshot;
  queueSnapshot.reserve(time_interval + 1);
  for (int i = 0; i < 50; i++) {
    orderSnapShots.push_back(queueSnapshot);
  }
  this->pyramidalWindow.currentOrder = 0;
}

/**
 * @Description: this function is pyramidal window for Clustream,
 * which takes and stores the snapshots of micro clusters
 * @Param: microClusters: micro clusters' snapshot need to be stored
 * startTime: start time of algorithms
 * @Return: void
 */
// TODO Still need to debug
void SESAME::LandmarkWindow::pyramidalWindowProcess(
    int elapsedTime, const SESAME::MicroClusters &microClusters) {
  int i = -1;
  if (elapsedTime > 0) {
    this->pyramidalWindow.currentOrder =
        (int)(log(elapsedTime) / log(this->pyramidalWindow.time_interval));
    // NOTE: snapshot when elapsed time =0 always add to the front of latest T
    // order
    while (++i >= 0) //++i>=0
    {
      if (this->pyramidalWindow.currentOrder >= i &&
          elapsedTime % (int)(pow(this->pyramidalWindow.time_interval, i)) ==
              0) {
        if (elapsedTime %
                (int)(pow(this->pyramidalWindow.time_interval, i + 1)) !=
            0) {
          storeSnapshot(i, microClusters, elapsedTime);
        }
      } else
        break;
    }
  } else
    storeSnapshot(0, microClusters, 0);
}
/**
 * @Description: this function stores snapshots into the pyramidal window data
 * structure
 * @Param: currentOrder: the ith order  snapshots stored into
 * microClusters: micro clusters' snapshot need to be stored
 * elapsedTime: the current elapsed time
 * @Return: void
 */
void SESAME::LandmarkWindow::storeSnapshot(unsigned int currentOrder,
                                           const MicroClusters &microClusters,
                                           int elapsedTime) {
  //  SESAME_INFO("taking snapshot");

  unsigned int size = orderSnapShots[currentOrder].size();
  SnapshotPtr snapshot;
  snapshot = DataStructureFactory::createSnapshot(
      const_cast<std::vector<MicroClusterPtr> &>(microClusters), elapsedTime);
  // SESAME_INFO("The current order size is "<<size<<" current order is
  // "<<currentOrder<<" elapsed Time "<<elapsedTime);
  if (size == this->pyramidalWindow.time_interval + 1) {
    orderSnapShots[currentOrder].erase(orderSnapShots[currentOrder].begin());
  }
  orderSnapShots[currentOrder].push_back(snapshot);
}
void SESAME::LandmarkWindow::clearPyramidalWindow() {
  std::vector<SESAME::QueueSnapshotPtr>().swap(this->orderSnapShots);
}
