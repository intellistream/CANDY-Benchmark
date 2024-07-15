//
// Created by 1124a on 2021/8/19.
//
#include <Algorithm/WindowModel/DampedWindow.hpp>
#include <iostream>
SESAME::DampedWindow::DampedWindow(double base, double lambda) {
  this->base = base;
  this->lambda = lambda;
}

double SESAME::DampedWindow::decayFunction(timespec startTime,
                                           timespec currentTimestamp) const {
  long elapsedTime =
      (((currentTimestamp).tv_sec * 1000000L +
        (currentTimestamp).tv_nsec / 1000L) -
       ((startTime).tv_sec * 1000000L + (startTime).tv_nsec / 1000L));
  return pow(this->base, -1 * this->lambda * elapsedTime);
}
double SESAME::DampedWindow::decayFunction(int startTime,
                                           int currentTimestamp) const {
  int elapsedTime = currentTimestamp - startTime;
  return pow(this->base, -1 * this->lambda * elapsedTime);
}