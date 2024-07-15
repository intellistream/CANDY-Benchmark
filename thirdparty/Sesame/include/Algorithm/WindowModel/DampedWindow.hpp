//
// Created by 1124a on 2021/8/19.
//

#ifndef SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_DAMPEDWINDOW_HPP_
#define SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_DAMPEDWINDOW_HPP_

#include "Algorithm/WindowModel/WindowModel.hpp"
#include "Timer/TimeMeter.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
namespace SESAME {

class DampedWindow;
typedef std::shared_ptr<DampedWindow> DampedWindowPtr;
class DampedWindow : WindowModel {
public:
  double base;
  double lambda;
  DampedWindow(double base, double lambda);
  double decayFunction(timespec startTime, timespec currentTimestamp) const;
  double decayFunction(int startTime, int currentTimestamp) const;
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_DAMPEDWINDOW_HPP_
