// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 17/8/21.
//

#ifndef SESAME_SRC_SINKS_DATASINKFACTORY_HPP_
#define SESAME_SRC_SINKS_DATASINKFACTORY_HPP_
#include <Sinks/DataSink.hpp>
namespace SESAME {
class DataSinkFactory {
public:
  static SESAME::DataSinkPtr create();
};
} // namespace SESAME

#endif // SESAME_SRC_SINKS_DATASINKFACTORY_HPP_
