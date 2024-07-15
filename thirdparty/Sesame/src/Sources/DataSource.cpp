// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/07/2021.
//

#include "Sources/DataSource.hpp"
#include "Algorithm/DataStructure/DataStructureFactory.hpp"
#include "Utils/Logger.hpp"
#include "Utils/UtilityFunctions.hpp"

#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono;

/**
 * Create input data points.
 * TODO: Remove the hard-coded part.
 * @param point_number
 * @param dim
 * @param input
 * @return
 */
void SESAME::DataSource::load() {
  // Pass input file as a string to DataSource.
  std::vector<std::string> data;
  ifstream infile;
  infile.open(param.input_file);
  if (infile.is_open() == 0) {
    std::cerr << "input file not found" << std::endl;
    exit(1);
  }
  SESAME_INFO("Read from the file...");

  // insert the data once per line into the string vector, every string element
  // represents a data line
  for (int i = 0; i < param.num_points; i++) {
    data.emplace_back();
    getline(infile, data[i]);
  }
  infile.close();

  // convert the string format into point vector, every string represents a
  // point
  SESAME_INFO("Finished loading input data");
  // The step used to generate random timestamps
  const int timeStep = 100000;
  for (int i = 0; i < param.num_points; i++) {
    int timeStamp = timeStep * i + rand() % timeStep;
    PointPtr point = GenericFactory::New<Point>(param.dim, i);
    point->timestamp = timeStamp;
    auto charData = data[i].data();
    // use c_str() to convert string to char * but it's just a temp pointer we
    // have to use strcpy to store it
    const char *sep = " ";
    char *feature =
        strtok(charData, sep);      // TODO: why this?? Read token from charData
    feature = strtok(nullptr, sep); // Skip the first token (index number)
    int index = 0;
    while (feature != nullptr) {
      if (index == param.dim) {
        point->setClusteringCenter(atoi(feature));
        if (point->getClusteringCenter() ==
            -1) { // If cluster id == -1, then it is an noise / outlier
          point->setOutlier(true);
        } else {
          point->setOutlier(false);
        }
      } else {
        point->setFeatureItem(strtod(feature, nullptr), index);
        index++;
      }
      feature = strtok(nullptr, sep);
    }
    this->input.push_back(point);
  }
}

SESAME::DataSource::DataSource(const param_t &param) : param(param) {
  inputQueue = GenericFactory::New<boost::lockfree::spsc_queue<PointPtr>>(
      (size_t)param.num_points);
  threadPtr = std::make_shared<SingleThread>();
  sourceEnd = false;
}

void SESAME::DataSource::push(const PointPtr &p) {
  bool flag = false;
  do {
    flag = inputQueue->push(p);
  } while (!flag);
}

// TODO: we can control the source speed here: done
void SESAME::DataSource::runningRoutine() {
  barrierPtr->arrive_and_wait();
  overallMeter.START_MEASURE();
  // Initialize timer at time 0
  auto start = high_resolution_clock::now();
  SESAME_INFO("DataSource start to emit data");
  if (param.arr_rate) {
    const int wait_ns = 1e9 / param.arr_rate;
    for (PointPtr &p : this->input) {
      p->toa = std::chrono::high_resolution_clock::now();
      push(p->copy());
      while (std::chrono::high_resolution_clock::now() - (p->toa) <
             std::chrono::nanoseconds(wait_ns))
        ;
    }
  } else if (param.fast_source) {
    for (PointPtr &p : this->input) {
      p->toa = std::chrono::high_resolution_clock::now();
      push(p->copy());
    }
  } else {
    for (PointPtr &p : this->input) {
      auto now = high_resolution_clock::now();
      if (p->timestamp > duration_cast<nanoseconds>(now - start).count()) {
        std::this_thread::sleep_for(nanoseconds(
            p->timestamp - duration_cast<nanoseconds>(now - start).count()));
      }
      // Wait until (currentTime - startTime) >= timestamp to push point
      p->toa = std::chrono::high_resolution_clock::now();
      push(p->copy());
    }
  }
  SESAME_DEBUG("sourceEnd set to true");
  sourceEnd = true; // Let engine knows that there won't be any more data
                    // coming.
  barrierPtr->arrive_and_wait();
  overallMeter.END_MEASURE();
  SESAME_DEBUG("DataSource sourceEnd emit data");
  // printTime();
}

bool SESAME::DataSource::start(int id) {
  auto fun = [this]() { runningRoutine(); };
  threadPtr->construct(fun, id);
  SESAME_DEBUG("DataSource spawn thread=" << threadPtr->getID());
  return true;
}

bool SESAME::DataSource::stop() {
  if (threadPtr) {
    SESAME_DEBUG("DataSource::stop try to join threads=" << threadPtr->getID());
    threadPtr->join();
    threadPtr.reset();
  } else {
    SESAME_DEBUG("DataSource: Thread is not joinable");
    return false;
  }
  return true;
}

bool SESAME::DataSource::empty() { return inputQueue->empty(); }

SESAME::PointPtr SESAME::DataSource::get() {
#ifndef NDEBUG
  // assert(inputQueue->size());
#endif
  auto rt = inputQueue->front();
  inputQueue->pop();
  return rt;
}

void SESAME::DataSource::setBarrier(SESAME::BarrierPtr barrierPtr) {
  this->barrierPtr = barrierPtr;
}
SESAME::DataSource::~DataSource() { stop(); }

vector<SESAME::PointPtr> SESAME::DataSource::getInputs() { return input; }

void SESAME::DataSource::printTime() {
  SESAME_INFO("DataSource takes " << overallMeter.MeterUSEC()
                                  << " useconds to finish.");
}

bool SESAME::DataSource::sourceEnded() { return sourceEnd; }
