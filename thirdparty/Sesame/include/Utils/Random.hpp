#ifndef SESAME_INCLUDE_UTILS_RANDOM_HPP_
#define SESAME_INCLUDE_UTILS_RANDOM_HPP_

#include <random>

class Random {
private:
  std::mt19937 r;

public:
  Random(int seed = 0) : r(seed) {}
  int random_uniform(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(r);
  }
  double random_uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(r);
  }
  bool bernoulli(double p) {
    std::bernoulli_distribution dist(p);
    return dist(r);
  }
};

#endif