// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#ifndef SESAME_SRC_SOURCES_DATASOURCE_HPP_
#define SESAME_SRC_SOURCES_DATASOURCE_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Engine/SingleThread.hpp"
#include "Timer/TimeMeter.hpp"
#include "Utils/SPSCQueue.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <boost/lockfree/spsc_queue.hpp>

#include <atomic>
#include <cstring>
#include <list>
#include <queue>
#include <string>
#include <vector>

namespace SESAME {
class DataSource;
typedef std::shared_ptr<DataSource> DataSourcePtr;

class DataSource {
private:
  std::vector<PointPtr> input;
  std::shared_ptr<boost::lockfree::spsc_queue<PointPtr>> inputQueue;
  SingleThreadPtr threadPtr;
  BarrierPtr barrierPtr;
  TimeMeter overallMeter;
  std::atomic_bool sourceEnd;
  param_t param;

public:
  void load();
  bool empty();
  PointPtr get();
  std::vector<PointPtr> getInputs();
  DataSource(const param_t &);
  ~DataSource();
  void runningRoutine();
  bool start(int i);
  bool stop();
  void setBarrier(BarrierPtr barrierPtr);
  void printTime();
  bool sourceEnded();
  int size() { return inputQueue->read_available(); }
  void push(const PointPtr &p);
};
} // namespace SESAME
#endif // SESAME_SRC_SOURCES_DATASOURCE_HPP_
