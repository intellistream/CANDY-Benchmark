// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 27/07/2021.
//

#ifndef SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_WINDOWFACTORY_HPP_
#define SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_WINDOWFACTORY_HPP_

#include <Algorithm/WindowModel/DampedWindow.hpp>
#include <Algorithm/WindowModel/LandmarkWindow.hpp>
#include <memory>
namespace SESAME {
class WindowFactory {
public:
  static std::shared_ptr<LandmarkWindow> createLandmarkWindow();
  static std::shared_ptr<DampedWindow> createDampedWindow(double base,
                                                          double lambda);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_WINDOWMODEL_WINDOWFACTORY_HPP_
