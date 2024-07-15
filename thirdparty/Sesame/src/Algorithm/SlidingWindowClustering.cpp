#include "Algorithm/SlidingWindowClustering.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"

#include <cassert>

namespace SESAME {
SlidingWindowClustering::SlidingWindowClustering(param_t &cmd_params)
    : r(cmd_params.seed) {
  this->param = cmd_params;
}

SlidingWindowClustering::~SlidingWindowClustering() {}

void SlidingWindowClustering::Init() { sum_timer.Tick(); }

void k_means_plus_plus(Random *r,
                       const std::vector<std::pair<PointPtr, double>> &instance,
                       int32_t k, std::vector<int32_t> *centers, double *cost) {
  centers->clear();

  std::vector<double> min_distance_to_centers(
      instance.size(), std::numeric_limits<double>::max());

  if (k >= instance.size()) {
    for (int32_t i = 0; i < instance.size(); i++) {
      centers->push_back(i);
    }
  } else {
    // add u.a.r. center.
    auto index = r->random_uniform(0, (int)instance.size() - 1);
    centers->push_back(index);
    while (centers->size() < k) {
      double sum_pow_min_distances = 0.0;
      vector<double> min_dist_powers;
      min_dist_powers.reserve(instance.size());

      for (int pos = 0; pos < instance.size(); ++pos) {
        double min_distance = std::min(
            min_distance_to_centers.at(pos),
            instance[pos].first->L2Dist(instance.at(centers->back()).first));
        min_distance_to_centers[pos] = min_distance;
        sum_pow_min_distances +=
            std::pow(min_distance, 2) * instance.at(pos).second;
        min_dist_powers.push_back(std::pow(min_distance, 2) *
                                  instance.at(pos).second);
      }

      double random_place = r->random_uniform(0.0, sum_pow_min_distances);
      for (int32_t i = 0; i < instance.size(); i++) {
        if (random_place <= min_dist_powers[i]) {
          centers->push_back(i);
          break;
        }
        random_place -= min_dist_powers[i];
      }
    }
  }

  *cost = 0;
  for (int pos = 0; pos < instance.size(); ++pos) {
    double min_distance =
        std::min(min_distance_to_centers.at(pos),
                 instance[pos].first->L2Dist(instance[centers->back()].first));
    *cost += std::pow(min_distance, 2) * instance.at(pos).second;
  }
}

// Given a series of instances of the problem, runs the k-means++ algorithm on
// each instance and outputs the vector of costs of the solutions found. Any
// other algorithm could be used instead of k-means++.
vector<double> cost_samples(Random *r, const vector<PointPtr> &samples,
                            int window_size, int num_samples, int k) {
  vector<double> costs;
  costs.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    std::vector<std::pair<PointPtr, double>> instance;
    for (int j = 0; j < window_size; j++) {
      instance.push_back(std::make_pair(samples[i * window_size + j], 1.0));
    }
    std::vector<int32_t> ingnored_centers;
    double cost;
    k_means_plus_plus(r, instance, k, &ingnored_centers, &cost);
    if (cost > 0) {
      costs.push_back(cost);
    }
  }
  return costs;
}

// Given a vector of costs, guesses a range for the optimum value. This is done
// by using a heuristic based on the min/max value and the standard deviation.
// Outputs the min and max bounds as a pair.
std::pair<double, double> guess_bounds(const vector<double> &costs) {
  auto min_max_it = std::minmax_element(costs.begin(), costs.end());
  double mean = std::accumulate(std::begin(costs), std::end(costs), double{0}) /
                costs.size();
  std::vector<double> costs_sq;
  costs_sq.resize(costs.size());
  for (const auto &cost : costs) {
    costs_sq.push_back(cost * cost);
  }
  double mean_sq =
      std::accumulate(std::begin(costs_sq), std::end(costs_sq), double{0}) /
      costs.size();
  double stddev = std::sqrt(mean_sq - mean * mean);

  double lowerbound = std::max(*min_max_it.first / 3, mean - 3.0 * stddev);
  double upperbound = std::max(mean + 3.0 * stddev, *min_max_it.second * 3);

  return std::make_pair(lowerbound, upperbound);
}

std::pair<double, double>
guess_optimum_range_bounds(Random *r, const vector<PointPtr> &samples,
                           int window_size, int num_samples, int k) {
  auto costs = cost_samples(r, samples, window_size, num_samples, k);
  return guess_bounds(costs);
}

void SlidingWindowClustering::RunOnline(PointPtr input) {
  ++count;
  if (!has_sampled) {
    win_timer.Tick();
    if (samples.size() < param.num_samples * param.sliding)
      samples.push_back(input);
    if (samples.size() >= param.num_samples * param.sliding) {
      const auto &[lower_bound, upper_bound] = guess_optimum_range_bounds(
          &r, samples, param.sliding, param.num_samples, param.num_clusters);
      framework = GenericFactory::New<FrameworkAlg<KMeansSummary>>(
          &r, param.sliding, param.num_clusters, param.delta_grid, lower_bound,
          upper_bound);
      for (auto p : samples) {
        framework->process_point(p);
      }
      has_sampled = true;
    }
    win_timer.Tock();
  } else {
    ds_timer.Tick();
    framework->process_point(input);
    ds_timer.Tock();
  }
  lat_timer.Add(input->toa);
}

void SlidingWindowClustering::RunOffline(DataSinkPtr sinkPtr) {
  on_timer.Add(sum_timer.start);
  ref_timer.Tick();
  std::vector<PointPtr> online_centers;
  double cost_estimate = 0;
  framework->solution(&online_centers, &cost_estimate);
  for (int i = 0; i < online_centers.size(); i++) {
    online_centers[i]->clu_id = i;
    sinkPtr->put(online_centers[i]);
  }
  ref_timer.Tock();
  sum_timer.Tock();
}

} // namespace SESAME