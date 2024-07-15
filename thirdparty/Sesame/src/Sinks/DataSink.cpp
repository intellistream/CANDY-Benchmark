// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#include "Sinks/DataSink.hpp"

#include <chrono>
#include <fstream>
#include <thread>

using namespace std::chrono_literals;
using namespace std;
using namespace SESAME;

DataSink::DataSink(const param_t &param) : param(param) {
  outputQueue = std::make_shared<boost::lockfree::spsc_queue<PointPtr>>(
      (size_t)param.num_points);
  threadPtr = std::make_shared<SingleThread>();
  sourceEnd = false;
}

void DataSink::runningRoutine() {
  barrierPtr->arrive_and_wait();
  SESAME_INFO("DataSink start to grab data");
  while (!sourceEnd) {
    while (!outputQueue->empty()) {
      output.push_back(outputQueue->front());
      outputQueue->pop();
    }
    std::this_thread::sleep_for(1ms);
  }
  while (!outputQueue->empty()) {
    output.push_back(outputQueue->front());
    outputQueue->pop();
  }
  finished = true;
  auto out = param.Name() + "." + param.Workload() + ".out";
  ofstream outfile = ofstream(out, ios::out | ios::trunc);
  for (auto &i : output) {
    outfile << i->index << "," << i->clu_id << "," << i->outlier;
    for (auto f : i->feature)
      outfile << "," << f;
    outfile << endl;
  }
  barrierPtr->arrive_and_wait();
  SESAME_INFO("DataSink sourceEnd grab data, in total:" << output.size());
}

bool DataSink::start(int id) {
  auto fun = [this]() { runningRoutine(); };
  threadPtr->construct(fun, id);
  SESAME_INFO("DataSink spawn thread=" << threadPtr->getID());
  return true;
}

bool DataSink::stop() {
  if (threadPtr) {
    SESAME_INFO("DataSink::stop try to join threads=" << threadPtr->getID());
    threadPtr->join();
    // threadPtr.reset();
    return true;
  }
  return false;
}

void DataSink::put(PointPtr resultPtr) { outputQueue->push(resultPtr); }

std::vector<PointPtr> &DataSink::getResults() { return output; }

void DataSink::setBarrier(BarrierPtr barrierPtr) {
  this->barrierPtr = barrierPtr;
}

DataSink::~DataSink() {
  Ended();
  stop();
}

void DataSink::Ended() { this->sourceEnd = true; }

bool DataSink::isFinished() { return this->finished.load(); }
