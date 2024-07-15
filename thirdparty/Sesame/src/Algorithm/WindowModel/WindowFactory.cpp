// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 27/07/2021.
//

#include <Algorithm/WindowModel/WindowFactory.hpp>
std::shared_ptr<SESAME::LandmarkWindow>
SESAME::WindowFactory::createLandmarkWindow() {
  return std::make_shared<LandmarkWindow>();
}
std::shared_ptr<SESAME::DampedWindow>
SESAME::WindowFactory::createDampedWindow(double base, double lambda) {
  return std::make_shared<DampedWindow>(base, lambda);
}
