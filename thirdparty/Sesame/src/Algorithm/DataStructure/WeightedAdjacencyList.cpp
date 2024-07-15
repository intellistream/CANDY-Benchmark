//
// Created by 1124a on 2021/8/30.
//
#include <Algorithm/DataStructure/WeightedAdjacencyList.hpp>

SESAME::AdjustedWeight::AdjustedWeight(double weight, int pointTime,
                                       timespec pointTime0) {
  this->weight = weight;
  this->updateTime = pointTime;
  this->updateTime0 = pointTime0;
}
void SESAME::AdjustedWeight::add(timespec startTime, double decayValue) {
  long elapsedTime =
      ((updateTime0).tv_sec * 1000000L + (updateTime0).tv_nsec / 1000L) -
      ((startTime).tv_sec * 1000000L + (startTime).tv_nsec / 1000L);
  if (elapsedTime == 0) {
    weight++;
  } else {
    weight *= decayValue + 1;
    this->updateTime0 = startTime;
  }
}
void SESAME::AdjustedWeight::add(int startTime, double decayValue) {
  if (startTime == this->updateTime) {
    weight++;
  } else {
    weight = weight * decayValue + 1;
    this->updateTime = startTime;
  }
}

double SESAME::AdjustedWeight::getCurrentWeight(double decayFactor) {
  return weight * decayFactor;
}
