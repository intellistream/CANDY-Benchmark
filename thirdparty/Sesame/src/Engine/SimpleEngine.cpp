// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#include "Engine/SimpleEngine.hpp"
#include "Utils/Logger.hpp"
#include "Utils/SPSCQueue.hpp"

#include <boost/timer/progress_display.hpp>

#ifdef GPERF
#include <gperftools/profiler.h>
#endif

#include <iostream>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

SESAME::SimpleEngine::SimpleEngine(DataSourcePtr sourcePtr, DataSinkPtr sinkPtr,
                                   AlgorithmPtr algoPtr)
    : threadID(0) {
  this->sourcePtr = std::move(sourcePtr);
  this->sinkPtr = std::move(sinkPtr);
  this->algoPtr = std::move(algoPtr);
  threadPtr = std::make_shared<SingleThread>();
}

void SESAME::SimpleEngine::run() {
  barrierPtr = UtilityFunctions::createBarrier(3);
  this->sourcePtr->setBarrier(barrierPtr);
  this->sinkPtr->setBarrier(barrierPtr);

  // start source thread
  this->sourcePtr->start(assignID());

  // start engine thread(s) for algorithm.
  this->start(sourcePtr, sinkPtr, algoPtr, assignID());

  // start sink thread
  this->sinkPtr->start(assignID());
}
bool SESAME::SimpleEngine::start(DataSourcePtr sourcePtr, DataSinkPtr sinkPtr,
                                 AlgorithmPtr algoPtr, int id) {
  auto fun = [this, sourcePtr, sinkPtr, algoPtr]() {
    runningRoutine(sourcePtr, sinkPtr, algoPtr);
  };
  threadPtr->construct(fun, id);
  SESAME_DEBUG("Engine spawn thread=" << threadPtr->getID());
  return true;
}
/**
 * TODO: this is not generic enough to capture different algorithms.
 * @param sourcePtr
 * @param sinkPtr
 * @param algoPtr
 */
void SESAME::SimpleEngine::runningRoutine(DataSourcePtr sourcePtr,
                                          DataSinkPtr sinkPtr,
                                          AlgorithmPtr algoPtr) {
  barrierPtr->arrive_and_wait(); // wait for source and sink.
  SESAME_INFO("Algorithm start to process data");
  overallMeter.START_MEASURE();
  overallMeter.overallStartMeasure();
  // We set observing interval for cumulative time: every 100 tuples
  overallMeter.setInterval(100);
  // initialization

  algoPtr->Init();

  boost::timer::progress_display show_progress(
      algoPtr->param.num_points, std::cerr, "Online Clustering:\n");

#ifdef GPERF
  std::string prof = "/tmp/" + algoPtr->param.Workload() + "." +
                     algoPtr->param.Name() + ".prof";
  ProfilerStart(prof.c_str());
#endif

  // run online clustering
  while (!sourcePtr->sourceEnded()) { // continuously processing infinite
                                      // incoming data streams.
    if (!sourcePtr->empty()) {
      auto item = sourcePtr->get();
      overallMeter.onlineAccMeasure();
      algoPtr->RunOnline(item->copy());
      algoPtr->Count();
      ++show_progress;
      overallMeter.onlineAccEMeasure();
    }
  }
  while (!sourcePtr->empty()) { // process the remaining data streams after
                                // source stops.
    auto item = sourcePtr->get();
    overallMeter.onlineAccMeasure();
    algoPtr->RunOnline(item->copy());
    algoPtr->Count();
    ++show_progress;
    overallMeter.onlineAccEMeasure();
  }
  overallMeter.onlineEndMeasure();

  SESAME_INFO("ready to offline clustering");

  // run offline clustering
  overallMeter.refinementStartMeasure();
  algoPtr->RunOffline(sinkPtr);
  SESAME_INFO("Engine sourceEnd process data");
  overallMeter.refinementEndMeasure();

#ifdef GPERF
  ProfilerStop();
#endif

  overallMeter.overallEndMeasure();
  overallMeter.END_MEASURE();
  // TODO Add break down output

  sinkPtr->Ended(); // Let sink knows that there won't be any more data coming.
  SESAME_INFO("Engine sourceEnd emit data");
  barrierPtr->arrive_and_wait(); // wait for source and sink.
  SESAME_DEBUG("Engine sourceEnd wait for source and sink.");
}

bool SESAME::SimpleEngine::stop() {
  if (threadPtr) {
    SESAME_DEBUG("Engine::stop try to join threads=" << threadPtr->getID());
    threadPtr->join();
    threadPtr.reset();
  } else {
    SESAME_DEBUG("Engine " << threadPtr->getID() << ": Thread is not joinable");
    return false;
  }
  return true;
}
int SESAME::SimpleEngine::assignID() { return threadID++; }
void SESAME::SimpleEngine::printTime() {
  SESAME_INFO("Engine takes " << overallMeter.MeterUSEC()
                              << " useconds to finish.");
  std::cout << "Engine takes " << overallMeter.MeterUSEC()
            << " useconds to finish." << std::endl;
  std::cout << "Online Time: " << overallMeter.MeterOnlineUSEC() << "\n"
            << "Refinement Time: " << overallMeter.MeterRefinementUSEC() << "\n"
            << "Overall Time: " << overallMeter.MeterOverallUSEC() << "\n"
            << std::endl;
}
