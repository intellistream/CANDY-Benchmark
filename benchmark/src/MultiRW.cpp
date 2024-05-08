/*! \file OnlineInsert.cpp*/
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
static inline std::vector<INTELLI::IntelliTimeStampPtr> timeStamps;
static inline timer_t timerid;

bool fileExists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good(); // Returns true if the file is open and in a good state
}
static inline int64_t s_timeOutSeconds = -1;
static inline int64_t s_numberOfRWSeq = -1, s_curSeq = 0;
static inline std::vector<int64_t> insert95Vec, pendingWriteVec, latQueryVec;
static inline std::vector<double> recallVec, throughputVec;
static inline void generateNormalResultCsv(INTELLI::ConfigMapPtr cfg, std::string fname) {
  int64_t throughputSum = 0, insert95Sum = 0, pendingWriteSum = 0, latQuerySum = 0;
  double recallSum = 0;
  int64_t seqNum = throughputVec.size();
  for (int64_t i = 0; i < seqNum; i++) {
    cfg->edit("throughput_" + to_string(i), (double) throughputVec[i]);
    throughputSum += throughputVec[i];
    cfg->edit("recall_" + to_string(i), (double) recallVec[i]);
    recallSum += recallVec[i];
    cfg->edit("95%latency(Insert)_" + to_string(i), (int64_t) insert95Vec[i]);
    insert95Sum += insert95Vec[i];
    cfg->edit("pendingWrite_" + to_string(i), (int64_t) pendingWriteVec[i]);
    pendingWriteSum += pendingWriteVec[i];
    cfg->edit("latencyOfQuery_" + to_string(i), (int64_t) latQueryVec[i]);
    latQuerySum += latQueryVec[i];
    cfg->edit("normalExit_" + to_string(i), (int64_t) 1);
  }
  cfg->edit("throughput", (double) throughputSum / seqNum);
  cfg->edit("recall", (double) recallSum / seqNum);
  cfg->edit("95%latency(Insert)", (int64_t) insert95Sum / seqNum);
  cfg->edit("pendingWrite", (int64_t) pendingWriteSum / seqNum);
  cfg->edit("latencyOfQuery", (int64_t) latQuerySum / seqNum);
  cfg->edit("normalExit", (int64_t) 1);
  cfg->toFile(fname);
  std::cout << "brief results\n" << cfg->toString() << std::endl;
}
static inline void generateEarlyAbortResultCsv(INTELLI::ConfigMapPtr cfg, std::string fname) {

  int64_t seqNum = throughputVec.size();
  int64_t i = 0;
  for (; i < s_curSeq; i++) {
    cfg->edit("throughput_" + to_string(i), (double) throughputVec[i]);

    cfg->edit("recall_" + to_string(i), (double) recallVec[i]);

    cfg->edit("95%latency(Insert)_" + to_string(i), (int64_t) insert95Vec[i]);

    cfg->edit("pendingWrite_" + to_string(i), (int64_t) pendingWriteVec[i]);

    cfg->edit("latencyOfQuery_" + to_string(i), (int64_t) latQueryVec[i]);

    cfg->edit("normalExit_" + to_string(i), (int64_t) 1);
  }
  for (; i < seqNum; i++) {
    cfg->edit("throughput_" + to_string(i), (double) 0);

    cfg->edit("recall_" + to_string(i), (double) 0);

    cfg->edit("95%latency(Insert)_" + to_string(i), (int64_t) 0);

    cfg->edit("pendingWrite_" + to_string(i), (int64_t) 0);

    cfg->edit("latencyOfQuery_" + to_string(i), (int64_t) 0);

    cfg->edit("normalExit_" + to_string(i), (int64_t) 0);
  }
  cfg->edit("throughput", (int64_t) 0);
  cfg->edit("recall", (double) 0);
  cfg->edit("95%latency(Insert)", (int64_t) 0);
  cfg->edit("pendingWrite", (int64_t) 0);
  cfg->edit("latencyOfQuery", (int64_t) 0);
  cfg->toFile(fname);
  std::cout << "brief results\n" << cfg->toString() << std::endl;
}
static inline void earlyTerminateTimerCallBack() {
  INTELLI_ERROR(
      "Force to terminate due to timeout in " + std::to_string(s_timeOutSeconds) + "seconds");
  if (indexPtr->isHPCStarted) {
    indexPtr->endHPC();
  }
  auto briefOutCfg = newConfigMap();
  UtilityFunctions::saveTimeStampToFile("multiRW_timestamps.csv", timeStamps);
  generateEarlyAbortResultCsv(briefOutCfg, "multiRW_result.csv");
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
  s_numberOfRWSeq = inMap->tryI64("numberOfRWSeq", 1, true);
  throughputVec = std::vector<double>(s_numberOfRWSeq, 0);
  insert95Vec = std::vector<int64_t>(s_numberOfRWSeq, 0);
  pendingWriteVec = std::vector<int64_t>(s_numberOfRWSeq, 0);
  latQueryVec = std::vector<int64_t>(s_numberOfRWSeq, 0);
  recallVec = std::vector<double>(s_numberOfRWSeq, 0);
  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  INTELLI_INFO("2.Data loader =" + dataLoaderTag);
  if (dataLoader == nullptr) {
    return -1;
  }
  dataLoader->setConfig(inMap);
  int64_t initialRows = inMap->tryI64("initialRows", 0, true);
  auto dataTensorAll = dataLoader->getData().nan_to_num(0);
  //auto dataTensorAll = dataLoader->getData();
  auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);
  auto dataTensorStreamAll = dataTensorAll.slice(0, initialRows, dataTensorAll.size(0));
  //auto queryTensor = dataLoader->getQuery();
  auto queryTensorAll = dataLoader->getQuery().nan_to_num(0);
  int64_t currSeq = 0;
  int64_t dbSeqRows = dataTensorStreamAll.size(0) / s_numberOfRWSeq;
  int64_t querySeqRows = queryTensorAll.size(0) / s_numberOfRWSeq;
  int64_t ANNK = inMap->tryI64("ANNK", 5, true);
  int64_t pendingWriteTime = 0;

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
  auto gdMap = newConfigMap();
  gdMap->loadFrom(*inMap);
  gdMap->edit("faissIndexTag", "flat");
  CANDY::IndexTable indexTable2;
  auto gdIndex = indexTable2.getIndex("faiss");
  gdIndex->setConfig(gdMap);
  if (initialRows > 0) {
    gdIndex->loadInitialTensor(dataTensorInitial);
  }
  if (cutOffTimeSeconds > 0) {
    setEarlyTerminateTimer(cutOffTimeSeconds);
    s_timeOutSeconds = cutOffTimeSeconds;
    INTELLI_WARNING(
        "Allow up to" + std::to_string(cutOffTimeSeconds) + "seconds before termination");
  }
  INTELLI_INFO("3.0 Load initial tensor!");
  INTELLI_INFO(
      "Initial tensor: Dimension =" + std::to_string(dataTensorInitial.size(1)) + ",#data="
          + std::to_string(dataTensorInitial.size(0)));
  if (initialRows > 0) {
    indexPtr->loadInitialTensor(dataTensorInitial);
  }
  int64_t frozenLevel = inMap->tryI64("frozenLevel", 1, true);
  indexPtr->setFrozenLevel(frozenLevel);
  for (; currSeq < s_numberOfRWSeq; currSeq++) {
    INTELLI_INFO("RW seq" + std::to_string(currSeq) + "/" + std::to_string(s_numberOfRWSeq));
    int64_t dbTensorStartRow = dbSeqRows * currSeq;
    int64_t dbTensorEndRow =
        (dbSeqRows * (currSeq + 1) > dataTensorStreamAll.size(0)) ? dataTensorStreamAll.size(0) : dbSeqRows
            * (currSeq + 1);
    s_curSeq = currSeq;
    auto dataTensorStream = dataTensorStreamAll.slice(0, dbTensorStartRow, dbTensorEndRow);

    int64_t qTensorStartRow = querySeqRows * currSeq;
    int64_t qTensorEndRow =
        (querySeqRows * (currSeq + 1) > queryTensorAll.size(0)) ? queryTensorAll.size(0) : querySeqRows * (currSeq + 1);
    auto queryTensor = queryTensorAll.slice(0, qTensorStartRow, qTensorEndRow);
    int64_t batchSize = inMap->tryI64("batchSize", dataTensorStream.size(0), true);

    INTELLI_INFO(
        "Streaming tensor: Dimension =" + std::to_string(dataTensorStream.size(1)) + ",#data="
            + std::to_string(dataTensorStream.size(0)) + ",#query="
            + std::to_string(queryTensor.size(0)));

    /**
    * @brief 3. create the timestamps
    */
    INTELLI::IntelliTimeStampGenerator timeStampGen;
    inMap->edit("streamingTupleCnt", (int64_t) dataTensorStream.size(0));
    timeStampGen.setConfig(inMap);
    timeStamps = timeStampGen.getTimeStamps();
    INTELLI_INFO("3.TimeStampSize =" + std::to_string(timeStamps.size()));

    /**
     * @brief 5. streaming feed
     */
    uint64_t startRow = 0;
    uint64_t endRow = startRow + batchSize;
    uint64_t tNow = 0;
    uint64_t tEXpectedArrival = timeStamps[endRow - 1]->arrivalTime;
    uint64_t tp = 0;
    uint64_t tDone = 0;
    uint64_t aRows = dataTensorStream.size(0);

    auto start = std::chrono::high_resolution_clock::now();

    INTELLI_INFO("3.1 STREAMING NOW!!!");
    double prossedOld = 0;
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
      auto subA = dataTensorStream.slice(0, startRow, endRow);
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

    std::string groundTruthPrefix = inMap->tryString("groundTruthPrefix", "onlineInsert_GroundTruth", true);

    std::string probeName = groundTruthPrefix + "/" + std::to_string(indexResults.size() - 1) + ".rbt";
    double recall = 0.0;

    {
      INTELLI_INFO("Ground truth does not exist, so I'll create it");

      gdIndex->insertTensor(dataTensorStream);

      auto gdResults = gdIndex->searchTensor(queryTensor, ANNK);
      INTELLI_INFO("Ground truth is done");
      recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
      UtilityFunctions::tensorListToFile(gdResults, groundTruthPrefix);
      //std::cout<<"index results"<<indexResults[0]<<std::endl;
      //std::cout<<"ground truth results"<<gdResults[0]<<std::endl;
    }

    double throughput = aRows * 1e6 / tDone;
    throughputVec[currSeq] = throughput;
    pendingWriteVec[currSeq] = pendingWriteTime;
    insert95Vec[currSeq] = UtilityFunctions::getLatencyPercentage(0.95, timeStamps);
    recallVec[currSeq] = recall;
    latQueryVec[currSeq] = queryLatency;
    UtilityFunctions::saveTimeStampToFile("multiRW_timestamps" + to_string(currSeq) + ".csv", timeStamps);
  }

  indexPtr->endHPC();
  indexPtr->isHPCStarted = false;
  auto briefOutCfg = newConfigMap();
  generateNormalResultCsv(briefOutCfg, "multiRW_result.csv");
  return 0;
}