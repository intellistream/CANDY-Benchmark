// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

#ifndef SESAME_SRC_TIMER_CYCLEMETER_HPP_
#define SESAME_SRC_TIMER_CYCLEMETER_HPP_

#include "Algorithm/Algorithm.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime> /* gettimeofday */
#include <fstream>
#include <iostream>
#include <vector>

namespace SESAME {
struct T_TIMER {
#ifndef NO_TIMING
  // the start and end timestamp of the algorithm
  struct timespec start, end, overallPre;
  /**
   * accumulate for data point coming
   * online_increment_timer_pre: start of accumulate online timer ( start of
   * every xx s) online_timer: end of the online increment part
   * */
  timespec online_increment_timer_pre, onlineAccTimer, online_timer;

  /**
   * initialTimerStart: Start of initial part
   *  initialTimer: end of initial part
   * */
  timespec initialTimerStart, initialTimer;

  /**
   *  accumulate for data point coming
   *  dataInsertTimer_pre: start of accumulated data insertion timer ( start of
   *every xx s) dataInsertTimer: end of the online increment part
   **/

  timespec dataInsertTimer_pre, dataInsertTimer; // accumulate.

  /**
   *  accumulate for data point coming
   *  clusterUpdateTimer_pre: start of accumulated cluster update timer ( start
   *of every xx s) clusterUpdateTimer: end of the cluster update part
   **/
  timespec clusterUpdateTimer_pre, clusterUpdateTimer; // accumulate.
  /**
   *  accumulate for data point coming
   *  outlierDetectionTimer_pre: start of accumulated outlier Detection timer (
   *start of every xx s) outlierDetectionTimer: end of the outlier Detection
   *part
   **/

  timespec outlierDetectionTimer_pre, outlierDetectionTimer; // accumulate.

  /**
   *  accumulate for data point coming
   *  pruneTimer_pre: start of accumulated prune timer (when every periodical
   *prune function starts) pruneTimer: end of the prune part
   **/

  timespec pruneTimer_pre, pruneTimer; // accumulate.

  /**
   *  accumulate for data point coming
   *  pruneTimer_pre: start of accumulated snapshot timer
   *  pruneTimer: end of the snapshot part
   **/
  timespec snapshotTimer_pre,
      snapshotTimer; // accumulate (Special for CluStream key design)
  timespec finalCluster_pre,
      finalClusterTimer; // accumulate (Special for online or periodical forming
                         // final cluster)

  /**
   * refinementStart: Start of refinement part
   *  refinementTimer: end of refinement part
   **/

  timespec refinementStart, refinementTimer; // offline refinement

  // Store the overall elapsed time every xx s
  std::vector<long> recordOverall;

  // Store the overall elapsed time every xx s
  std::vector<long> recordOnline;

  // Store the data insertion elapsed time every xx s
  std::vector<long> recordInsert;

  // Store the data Concept drift elapsed time every xx s
  std::vector<long> recordConceptDrift;

  // Store the data Outlier Detection every xx s
  std::vector<long> recordOutlierDetection;

  // Store the data prune elapsed time when it occurs
  std::vector<long> prune;

  std::vector<long> snapshot;

  std::vector<long> finalCluster;

  int pruneCnt = 0;
  int snapshotCnt = 0;
  int periodicalCluCnt = 0;

  /*
  uint64_t overall_timer, online_increment_timer_pre,
  online_increment_timer;//accumulate for data point coming, uint64_t
  online_timer; uint64_t initialTimer = 0;//initialTimer_pre = 0, uint64_t
  dataInsertTimer_pre = 0, dataInsertTimer = 0;//accumulate. uint64_t
  conceptDriftTimer_pre = 0, conceptDriftTimer = 0;//accumulate. uint64_t
  outlierDetectionTimer_pre = 0, outlierDetectionTimer = 0;//accumulate.
  uint64_t pruneTimer_pre = 0, pruneTimer = 0;//accumulate. uint64_t
  snapshotTimer_pre = 0, snapshotTimer = 0;//accumulate(Special for CluStream
  key design) uint64_t refinementTimer = 0;//offline refinement
  std::vector<uint64_t> recordOverall; std::vector<uint64_t> recordInsert;
  std::vector<uint64_t> recordConceptDrift; std::vector<uint64_t>
  recordOutlierDetection; std::vector<int32_t> recordID;
   */
#endif
};

class TimeMeter {
private:
  struct timespec start, stop;
  int interval = 1, intervalCnt = 0;
  T_TIMER timer;
  bool InsertJudge = false;
  // the overall elapsed time of every part
  long overallTime = 0;
  long overallPreTime = 0;
  long onlineTime = 0;
  long dataInsertTime = 0;
  long onlineClusterUpdateTime = 0;
  std::vector<long> recordOverall;

  long outlierDetectionTime = 0; // if possible
  long pruneTime = 0;            // if possible
  long initialTime = 0;          // if possible
  long snapshotTime = 0;         // if possible
  long refinementTime = 0;       // if possible
  long finalClusterTime = 0;     // if possible
  long otherTime = 0;

public:
  void setInterval(int interV);
  void START_MEASURE();
  void END_MEASURE();
  long MeterUSEC(); // return the meter result in micro second unit.

  // static void MEASURE(timespec Time);
  long
  MeterUSEC(timespec startAcc,
            timespec endAcc); // return the meter result in micro second unit.

  // the overall start and end time of every part
  void overallStartMeasure();
  void overallEndMeasure();
  long MeterOverallUSEC();
  // the start  of every xx s
  void overallAccMeasure();
  // start of online part
  // void MeterOverallAccUSEC();

  void onlineAccMeasure();
  void onlineAccEMeasure();
  // end of online part
  void onlineEndMeasure();
  void MeterOnlineAccUSEC();
  long MeterOnlineUSEC();

  // start of initial  part
  void initialMeasure();
  // end of initial part
  void initialEndMeasure();
  long MeterInitialUSEC();

  // start of data Insert  part
  void dataInsertAccMeasure();
  // end of data Insert  part
  void dataInsertEndMeasure();
  void MeterDataInsertAccUSEC();
  long MeterDataInsertUSEC();

  // start of Online cluster update  part
  void clusterUpdateAccMeasure();
  // end of Online cluster update
  void clusterUpdateEndMeasure();
  void MeterClusterUpdateAccUSEC();
  long MeterClusterUpdateUSEC();

  // start of outlier Detection part
  void outlierDetectionAccMeasure();
  // end of outlier Detection  part
  void outlierDetectionEndMeasure();
  void MeterOutlierDetectionAccUSEC();
  long MeterOutlierDetectionUSEC();

  // start of prune part
  void pruneAccMeasure();
  // end of prune  part
  void pruneEndMeasure();
  void MeterPruneAccUSEC();
  long MeterPruneUSEC();

  // start of snapshot  part
  void snapshotAccMeasure();
  // end of snapshot  part
  void snapshotEndMeasure();
  void MeterSnapshotAccUSEC();
  long MeterSnapshotUSEC();

  // Used for incremental or periodical
  // start of forming final clusters
  void finalClusterAccMeasure();
  // end of forming final clusters
  void finalClusterEndMeasure();
  void MeterFinalClusterAccUSEC();
  long MeterFinalClusterUSEC();

  // start of refinement  part
  void refinementStartMeasure();
  // end of refinement  part
  void refinementEndMeasure();
  long MeterRefinementUSEC();

  // Store the result of every xx s
  void AccumulateWithPointTimer(timespec start, timespec end, long elapsedTime,
                                std::vector<long> timerVector);
  long getOnlineEtime();

  void OverallPreUpdate();

  void setOverallTime(long overallT);
  void setOnlineTime(long onlineT);
  void setDataInsertTime(long dataInsertT);
  void setClusterUpdateTime(long clusterUpdateT);
  void setOutlierDetectionTime(long outlierDetectionT);

  void setInitialTime(long initialT);
  // if possible
  void setRefinementTime(long refinementT);
  // if possible
  void setSnapshotTime(long snapshotT);
  void setPruneTime(long pruneT);

  /** print out the execution time statistics of stream clustering algorithms */
  void breakdown_global(bool initial, bool snapshot, bool outlierBuffer,
                        bool refine); // int64_t total_results,
  /** print out the execution time statistics of stream clustering algorithms */
  void printTime(bool initial, bool snapshot, bool outlierBuffer,
                 bool finalCluster); //
  void printCumulative();
  // TODO the code below will be removed later
  /*
  #ifndef BEGIN_MEASURE_INITIALIZE
  #define BEGIN_MEASURE_INITIALIZE(timer) \
  startTimer(&(timer)->initialTimer);
  #endif

  #ifndef END_MEASURE_INITIALIZE
  #define END_MEASURE_INITIALIZE(timer) \
  stopTimer(&(timer)->initialTimer);
  #endif


  #ifndef BEGIN_MEASURE_ONLINE
  #define BEGIN_MEASURE_ONLINE(timer) \
  startTimer(&(timer)->online_timer);
  #endif

  #ifndef END_MEASURE_ONLINE
  #define END_MEASURE_ONLINE(timer) \
  stopTimer(&(timer)->online_timer);
  #endif

  #ifndef BEGIN_MEASURE_ONLINE_ACC
  #define BEGIN_MEASURE_ONLINE_ACC(timer) \
  startTimer(&(timer)->online_timer_pre);
  #endif
  */

#ifndef /*END_MEASURE_ONLINE_ACC*/ NO_TIMING
#define END_MEASURE_ONLINE_ACC(timer)                                          \
  accTimer(&(timer)->online_timer_pre,                                         \
           &(timer)->online_timer); // ONLINE one-pass absorbing data time
#endif

#ifndef BEGIN_MEASURE_INSERT_ACC
#define BEGIN_MEASURE_INSERT_ACC(timer)                                        \
  startTimer(&(timer)->dataInsertTimer_pre);
#endif

#ifndef /*END_MEASURE_INSERT_ACC*/ NO_TIMING
#define END_MEASURE_INSERT_ACC(timer)                                          \
  accTimer(&(timer)->dataInsertTimer_pre,                                      \
           &(timer)->dataInsertTimer); /* data insert time */
#endif

#ifndef BEGIN_MEASURE_CONDRIFT_ACC
#define BEGIN_MEASURE_CONDRIFT_ACC(timer)                                      \
  startTimer(&(timer)->conceptDriftTimer_pre);
#endif

#ifndef /*END_MEASURE_CONDRIFT_ACC*/ NO_TIMING
#define END_MEASURE_CONDRIFT_ACC(timer)                                        \
  accTimer(&(timer)->conceptDriftTimer_pre,                                    \
           &(timer)->conceptDriftTimer); /* Concept drift time */
#endif

#ifndef BEGIN_MEASURE_PRUNE_ACC
#define BEGIN_MEASURE_PRUNE_ACC(timer) startTimer(&(timer)->pruneTimer_pre);
#endif

#ifndef /*END_MEASURE_PRUNE_ACC*/ NO_TIMING
#define END_MEASURE_PRUNE_ACC(timer)                                           \
  accTimer(&(timer)->pruneTimer_pre,                                           \
           &(timer)->pruneTimer); /* Outlier prune time */
#endif

#ifndef BEGIN_MEASURE_SNAPSHOT_ACC
#define BEGIN_MEASURE_SNAPSHOT_ACC(timer)                                      \
  startTimer(&(timer)->snapshotTimer_pre);
#endif

#ifndef /*END_MEASURE_SNAPSHOT_ACC*/ NO_TIMING
#define END_MEASURE_SNAPSHOT_ACC(timer)                                        \
  accTimer(&(timer)->snapshotTimer_pre,                                        \
           &(timer)->snapshotTimer); // Taking Snapshots time
#endif

  /*
  #ifndef BEGIN_MEASURE_REFINEMENT
  #define BEGIN_MEASURE_REFINEMENT(timer) \
  startTimer(&(timer)->refinementTimer);
  #endif

  #ifndef END_MEASURE_REFINEMENT
  #define END_MEASURE_REFINEMENT(timer) \
  stopTimer(&(timer)->refinementTimer);
  #endif


  #ifndef OVERALL_START_MEASURE
  #define OVERALL_START_MEASURE(timer) \
  gettimeofday(&(timer)->start, NULL); \
  startTimer(&(timer)->overall_timer); (\
  tim)er->partition_timer = 0;
  #endif

  #ifndef OVERALL_END_MEASURE
  #define OVERALL_END_MEASURE(timer) \
  stopTimer(&(timer)->overall_timer);  \
  gettimeofday(&(timer)->end, NULL);
  #endif
   */
};

} // namespace SESAME

#endif // SESAME_SRC_TIMER_CYCLEMETER_HPP_
