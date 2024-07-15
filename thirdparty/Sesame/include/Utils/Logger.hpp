// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 19/07/2021.
//

#ifndef SESAME_INCLUDE_UTILS_LOGGER_HPP_
#define SESAME_INCLUDE_UTILS_LOGGER_HPP_
// TRACE < DEBUG < INFO < WARN < ERROR < FATAL
#include <iostream>

enum DebugLevel { LOG_NONE, LOG_WARNING, LOG_DEBUG, LOG_INFO, LOG_TRACE };

#define LEVEL_TRACE 6
#define LEVEL_DEBUG 5
#define LEVEL_INFO 4
#define LEVEL_WARN 3
#define LEVEL_ERROR 2
#define LEVEL_FATAL 1

#define SESAME_TRACE(TEXT) std::cerr << TEXT << std::endl;
#ifndef NDEBUG
#define SESAME_DEBUG(TEXT) std::cerr << TEXT << std::endl;
#define SESAME_INFO(TEXT) std::cerr << TEXT << std::endl;
#else
#define SESAME_DEBUG(TEXT) ;
#define SESAME_INFO(TEXT) ;
#endif
#define SESAME_TRACE(TEXT) std::cerr << TEXT << std::endl;
#define SESAME_WARNING(TEXT) std::cerr << TEXT << std::endl;
#define SESAME_ERROR(TEXT) std::cerr << TEXT << std::endl;
#define SESAME_FATAL_ERROR(TEXT) std::cerr << TEXT << std::endl;

#endif // SESAME_INCLUDE_UTILS_LOGGER_HPP_
