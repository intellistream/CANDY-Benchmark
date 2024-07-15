#ifndef SESAME_INCLUDE_UTILS_TYPES_HPP_
#define SESAME_INCLUDE_UTILS_TYPES_HPP_

#include <chrono>
#include <cstdint>

namespace SESAME {
using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

using fp32 = float;
using fp64 = double;

using clock_t = std::chrono::_V2::system_clock::time_point;
using feature_t = fp64;

} // namespace SESAME

#endif