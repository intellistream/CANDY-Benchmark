//
// Created by tony on 24-10-21.
//
// Created by tony on 06/01/24.
//
#include <iostream>
#include <torch/torch.h>
#include <CANDY.h>
#include <Utils/UtilityFunctions.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

using namespace INTELLI;
static inline CANDY::AbstractIndexPtr indexPtr = nullptr;
static inline std::vector<INTELLI::IntelliTimeStampPtr> timeStamps, tsDel;
static inline timer_t timerid;

bool fileExists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good(); // Returns true if the file is open and in a good state
}
static inline int64_t s_timeOutSeconds = -1;

static inline void earlyTerminateTimerCallBack() {
  INTELLI_ERROR(
      "Force to terminate due to timeout in " + std::to_string(s_timeOutSeconds) + "seconds");
  /*if (indexPtr->isHPCStarted) {
    indexPtr->endHPC();
  }*/
  auto briefOutCfg = newConfigMap();
  double latency95 = 0;
  briefOutCfg->edit("latencyOfConstruction", (int64_t) 0);
  briefOutCfg->edit("throughputOfInsert", (int64_t) 0);
  briefOutCfg->edit("throughputOfDel", (int64_t) 0);
  briefOutCfg->edit("recall", (int64_t) 0);
  briefOutCfg->edit("throughputByElements", (int64_t) 0);
  briefOutCfg->edit("95%latency(Del)", (int64_t) 0);
  briefOutCfg->edit("95%latency(Insert)", latency95);
  briefOutCfg->edit("pendingWrite", latency95);
  briefOutCfg->edit("latencyOfQuery", (int64_t) 0);
  briefOutCfg->edit("normalExit", (int64_t) 0);
  briefOutCfg->toFile("onlineCUD_result.csv");
  std::cout << "brief results\n" << briefOutCfg->toString() << std::endl;
  //UtilityFunctions::saveTimeStampToFile("onlineCUD_timestamps.csv", timeStamps);
  exit(-1);
}
static inline void timerCallback(union sigval v) {
  earlyTerminateTimerCallBack();
}
static inline void setEarlyTerminateTimer(int64_t seconds) {
  struct sigevent sev;
  memset(&sev, 0, sizeof(struct sigevent));
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = timerCallback;
  struct itimerspec its;
  its.it_value.tv_sec = seconds;
  its.it_value.tv_nsec = 0;
  its.it_interval.tv_sec = seconds;
  its.it_interval.tv_nsec = 0;
  timer_create(CLOCK_REALTIME, &sev, &timerid);
  timer_settime(timerid, 0, &its, nullptr);
}
/**
 *
 */
int main(int argc, char **argv) {

  /**
   * @brief 1. load the configs
   */
  INTELLI::ConfigMapPtr inMap = newConfigMap();
  //size_t incrementalBuildTime = 0, incrementalSearchTime = 0;
  if (inMap->fromCArg(argc, argv) == false) {
    if (argc >= 2) {
      std::string fileName = "";
      fileName += argv[1];
      if (inMap->fromFile(fileName)) { std::cout << "load config from file " + fileName << endl; }
    }
  }
  /**
   * @brief 2. create the data and query, and prepare initialTensor
   */
  CANDY::DataLoaderTable dataLoaderTable;
  std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);
  int64_t cutOffTimeSeconds = inMap->tryI64("cutOffTimeSeconds", -1, true);
  int64_t waitPendingWrite = inMap->tryI64("waitPendingWrite", 0, true);
  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  INTELLI_INFO("2.Data loader =" + dataLoaderTag);
  if (dataLoader == nullptr) {
    return -1;
  }
  dataLoader->setConfig(inMap);
  int64_t initialRows = inMap->tryI64("initialRows", 0, true);
  int64_t deleteRows = inMap->tryI64("deleteRows", 0, true);
  /*auto dataTensorAll = dataLoader->getData().nan_to_num(0);
  //auto dataTensorAll = dataLoader->getData();
  auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);
  auto dataTensorStream = dataTensorAll.slice(0, initialRows, dataTensorAll.size(0));*/
  //auto queryTensor = dataLoader->getQuery();
  auto queryTensor = dataLoader->getQuery().nan_to_num(0);
  int64_t streamSize = dataLoader->size() - initialRows;
  INTELLI_INFO(
      "Initial tensor: Dimension =" + std::to_string(dataLoader->getDimension()) + ",#data="
          + std::to_string(dataLoader->size()));
  INTELLI_INFO(
      "Streaming tensor: ,#data="
          + std::to_string(streamSize) + ",#query="
          + std::to_string(queryTensor.size(0)));

  /**
  * @brief 3. create the timestamps
  */
  INTELLI::IntelliTimeStampGenerator timeStampGen, tsGen2;
  inMap->edit("streamingTupleCnt", (int64_t) std::max(streamSize, deleteRows));
  timeStampGen.setConfig(inMap);
  timeStamps = timeStampGen.getTimeStamps();
  tsGen2.setConfig(inMap);
  tsDel = tsGen2.getTimeStamps();
  INTELLI_INFO("3.TimeStampSize =" + std::to_string(timeStamps.size()));
  int64_t batchSize = inMap->tryI64("batchSize", streamSize, true);
  /**
   * @brief 4. creat index
   */
  CANDY::IndexTable indexTable;
  std::string indexTag = inMap->tryString("indexTag", "flat", true);
  indexPtr = indexTable.getIndex(indexTag);
  if (indexPtr == nullptr) {
    return -1;
  }
  indexPtr->setConfig(inMap);
  indexPtr->startHPC();
  indexPtr->isHPCStarted = true;
  /**
   * @brief 5. streaming feed
   */
  uint64_t startRow = 0;
  uint64_t endRow = startRow + batchSize;
  uint64_t tNow = 0;
  uint64_t tEXpectedArrival = tsDel[endRow - 1]->arrivalTime;
  uint64_t tp = 0;
  uint64_t tDone = 0;
  uint64_t aRows = deleteRows;
  if (cutOffTimeSeconds > 0) {
    setEarlyTerminateTimer(cutOffTimeSeconds);
    s_timeOutSeconds = cutOffTimeSeconds;
    INTELLI_WARNING(
        "Allow up to" + std::to_string(cutOffTimeSeconds) + "seconds before termination");
  }
  INTELLI_INFO("3.0 Load initial tensor!");
  auto start = std::chrono::high_resolution_clock::now();
  if (initialRows > 0) {

    /*auto qTemp=queryTensor.clone();
    if(inMap->tryI64("loadQueryDistribution",0, false)==1) {
      INTELLI_WARNING("The distribution of query is also loaded");
      for (; qTemp.size(0) < dataTensorInitial.size(0);) {
        INTELLI::IntelliTensorOP::appendRows(&qTemp, &queryTensor);
      }
    }*/
    auto initialTensor = dataLoader->getDataAt(0, initialRows);
    indexPtr->loadInitialTensor(initialTensor);
  }
  int64_t constructionTime = chronoElapsedTime(start);
  start = std::chrono::high_resolution_clock::now();
  int64_t frozenLevel = inMap->tryI64("frozenLevel", 1, true);
  indexPtr->setFrozenLevel(frozenLevel);
  INTELLI_INFO("3.1  Delete NOW!!!");
  double prossedOld = 0;
  while (startRow < aRows) {
    tNow = chronoElapsedTime(start);
    //index++;
    while (tNow < tEXpectedArrival) {
      tNow = chronoElapsedTime(start);
      // INTELLI_INFO("Wait "+ to_string(tNow)+":"+ to_string(tEXpectedArrival));
    }
    double prossed = endRow;
    prossed = prossed * 100.0 / aRows;

    /**
     * @brief now, the whole batch has arrived, compute
     */
    auto subA = dataLoader->getDataAt((int64_t) startRow, (int64_t) endRow);
    //std::cout<<subA;
    indexPtr->deleteTensor(subA);
    tp = chronoElapsedTime(start);
    /**
     * @brief the new arrived A will be no longer probed, so we can assign the processed time now
     */
    for (size_t i = startRow; i < endRow; i++) {
      tsDel[i]->processedTime = tp;
    }
    /**
     * @brief update the indexes
     */

    startRow += batchSize;
    endRow += batchSize;
    if (endRow >= aRows) {
      endRow = aRows;
    }
    if (prossed - prossedOld >= 10.0) {
      INTELLI_INFO("Done" + to_string(prossed) + "%(" + to_string(startRow) + "/" + to_string(aRows) + ")");
      prossedOld = prossed;
    }
    tEXpectedArrival = tsDel[endRow - 1]->arrivalTime;

  }
  tDone = chronoElapsedTime(start);
  double throughputDel = aRows * 1e6 / tDone;

  INTELLI_INFO("3.2 Insert NOW!!!");
  startRow = 0;
  endRow = startRow + batchSize;
  tNow = 0;
  tEXpectedArrival = timeStamps[endRow - 1]->arrivalTime;
  tp = 0;
  tDone = 0;
  aRows = streamSize;
  start = std::chrono::high_resolution_clock::now();

  indexPtr->setFrozenLevel(frozenLevel);
  prossedOld = 0;
  while (startRow < aRows) {
    tNow = chronoElapsedTime(start);
    //index++;
    while (tNow < tEXpectedArrival) {
      tNow = chronoElapsedTime(start);
    }
    double prossed = endRow;
    prossed = prossed * 100.0 / aRows;

    /**
     * @brief now, the whole batch has arrived, compute
     */
    auto subA = dataLoader->getDataAt((int64_t) startRow + initialRows, (int64_t) endRow + initialRows);
    indexPtr->insertTensor(subA);
    tp = chronoElapsedTime(start);
    /**
     * @brief the new arrived A will be no longer probed, so we can assign the processed time now
     */
    for (size_t i = startRow; i < endRow; i++) {
      timeStamps[i]->processedTime = tp;
    }
    /**
     * @brief update the indexes
     */

    startRow += batchSize;
    endRow += batchSize;
    if (endRow >= aRows) {
      endRow = aRows;
    }
    if (prossed - prossedOld >= 10.0) {
      INTELLI_INFO("Done" + to_string(prossed) + "%(" + to_string(startRow) + "/" + to_string(aRows) + ")");
      prossedOld = prossed;
    }
    tEXpectedArrival = timeStamps[endRow - 1]->arrivalTime;

  }
  tDone = chronoElapsedTime(start);
  int64_t ANNK = inMap->tryI64("ANNK", 5, true);
  int64_t pendingWriteTime = 0;
  if (waitPendingWrite) {
    INTELLI_WARNING("There is pending write, wait first");
    auto startWP = std::chrono::high_resolution_clock::now();
    indexPtr->waitPendingOperations();
    pendingWriteTime = chronoElapsedTime(startWP);
    INTELLI_INFO("Wait " + std::to_string(pendingWriteTime / 1000) + " ms for pending writings");
  }
  INTELLI_INFO("Insert is done, let us validate the results");
  auto startQuery = std::chrono::high_resolution_clock::now();
  auto indexResults = indexPtr->searchTensor(queryTensor, ANNK);
  tNow = chronoElapsedTime(startQuery);
  INTELLI_INFO("Query done in " + to_string(tNow / 1000) + "ms");
  uint64_t queryLatency = tNow;
  indexPtr->endHPC();
  indexPtr->isHPCStarted = false;
  std::string groundTruthPrefix = inMap->tryString("groundTruthPrefix", "onlineCUD_GroundTruth", true);

  std::string probeName = groundTruthPrefix + "/" + std::to_string(indexResults.size() - 1) + ".rbt";
  double recall = 0.0;
  int64_t groundTruthRedo = inMap->tryI64("groundTruthRedo", 1, true);

  if (fileExists(probeName) && (groundTruthRedo == 0)) {
    INTELLI_INFO("Ground truth exists, so I load it");
    auto gdResults = UtilityFunctions::tensorListFromFile(groundTruthPrefix, indexResults.size());
    INTELLI_INFO("Ground truth is loaded");
    recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
  } else {
    INTELLI_INFO("Ground truth does not exist, so I'll create it");
    auto gdMap = newConfigMap();
    gdMap->loadFrom(*inMap);
    gdMap->edit("faissIndexTag", "flat");
    CANDY::IndexTable indexTable2;
    auto gdIndex = indexTable2.getIndex("flat");
    gdIndex->setConfig(gdMap);
    if (initialRows > 0) {
      auto initialTensor = dataLoader->getDataAt(0, initialRows);
      gdIndex->loadInitialTensor(initialTensor);
    }

    startRow = 0;
    endRow = startRow + batchSize;
    aRows = deleteRows;
    while (startRow < aRows) {

      /**
       * @brief now, the whole batch has arrived, compute
       */
      auto subA = dataLoader->getDataAt((int64_t) startRow , (int64_t) endRow );
      gdIndex->deleteTensor(subA);

      /**
       * @brief update the indexes
       */
      startRow += batchSize;
      endRow += batchSize;
      if (endRow >= aRows) {
        endRow = aRows;
      }
    }

    /* auto subADel = dataLoader->getDataAt((int64_t) 0, (int64_t) deleteRows);
     gdIndex->deleteTensor(subADel);*/
    auto subAInsert = dataLoader->getDataAt((int64_t) initialRows, (int64_t) (initialRows + streamSize));
    gdIndex->insertTensor(subAInsert);

    auto gdResults = gdIndex->searchTensor(queryTensor, ANNK);
    INTELLI_INFO("Ground truth is done");
    recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
    UtilityFunctions::tensorListToFile(gdResults, groundTruthPrefix);
    //std::cout<<"index results"<<indexResults[0]<<std::endl;
    //std::cout<<"ground truth results"<<gdResults[0]<<std::endl;
  }

  double throughput = aRows * 1e6 / tDone;
  // double throughputByElements = throughput * dataTensorStream.size(1);
  double latency95 = UtilityFunctions::getLatencyPercentage(0.95, timeStamps);
  double latency95Del = UtilityFunctions::getLatencyPercentage(0.95, tsDel);
  auto briefOutCfg = newConfigMap();
  briefOutCfg->edit("latencyOfConstruction", constructionTime);
  briefOutCfg->edit("throughputOfInsert", throughput);
  briefOutCfg->edit("throughputOfDel", throughputDel);
  briefOutCfg->edit("recall", recall);
  briefOutCfg->edit("95%latency(Insert)", latency95);
  briefOutCfg->edit("95%latency(Del)", latency95Del);
  briefOutCfg->edit("pendingWrite", pendingWriteTime);
  briefOutCfg->edit("latencyOfQuery", queryLatency);
  briefOutCfg->edit("normalExit", (int64_t) 1);
  briefOutCfg->toFile("onlineCUD_result.csv");
  auto indexStatistics = indexPtr->getIndexStatistics();
  std::cout << "brief results\n" << briefOutCfg->toString() << std::endl;
  std::cout << "index statistics\n" << indexStatistics->toString() << std::endl;
  indexStatistics->toFile("onlineCUD_indexStatistics.csv");
  UtilityFunctions::saveTimeStampToFile("onlineCUD_timestamps.csv", timeStamps);
  return 0;
}