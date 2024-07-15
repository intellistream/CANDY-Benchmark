// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 15/8/2021.
//

#ifndef SESAME_DATASOURCEFACTORY_H
#define SESAME_DATASOURCEFACTORY_H

#include <Sources/DataSource.hpp>

namespace SESAME {
class DataSourceFactory {
public:
  static SESAME::DataSourcePtr create();
};
} // namespace SESAME

#endif // SESAME_DATASOURCEFACTORY_H
