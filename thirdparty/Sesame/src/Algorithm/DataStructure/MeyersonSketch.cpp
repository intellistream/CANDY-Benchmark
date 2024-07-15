#include "Algorithm/DataStructure/MeyersonSketch.hpp"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace SESAME;

MeyersonSketch::MeyersonSketch(const param_t &param)
    : param(param), r(param.seed) {
  max_sketch_size_ = pow(2, 2 * 2 + 1) * 4. * param.k *
                     (1. + log(param.sliding * 3)) * (1.0 + 1. / 0.5);
}

void MeyersonSketch::Init() {}

MeyersonSketch::NodePtr MeyersonSketch::Insert(PointPtr input) {
  if (!has_sampled) {
    if (samples.size() < param.num_samples * param.sliding)
      samples.push_back(input);
    if (samples.size() >= param.num_samples * param.sliding) {
      const auto &[lower_bound, upper_bound] = guess_optimum_range_bounds(
          &r, samples, param.sliding, param.num_samples, param.num_clusters);
      distance_denominator_ =
          (upper_bound) / (param.k * (1. + log(param.sliding * 3)));
      NodePtr res;
      for (auto p : samples) {
        res = Process(p);
      }
      has_sampled = true;
      return res;
    }
    return nullptr;
  } else {
    return Process(input);
  }
}

MeyersonSketch::NodePtr MeyersonSketch::Process(PointPtr point) {
  if (centers.empty()) {
    return CreateCenter(point);
  }
  auto [node, dist] = CalcClosestNode(centers, point);
  bool open_new = r.bernoulli(min(1.0, pow(dist, 2) / distance_denominator_));
  if (open_new) {
    return CreateCenter(point);
  } else {
    node->Update(point);
    return node;
  }
}

MeyersonSketch::NodePtr MeyersonSketch::CreateCenter(PointPtr input) {
  if (centers.size() >= max_sketch_size_) {
    return nullptr;
  }
  auto node = std::make_shared<Node>(input);
  centers.push_back(node);
  return node;
}

MeyersonSketch::NodePtr MeyersonSketch::Insert(NodePtr node) {
  centers.push_back(node);
  return node;
}

void MeyersonSketch::Remove(NodePtr node) {
  auto it = std::find(centers.begin(), centers.end(), node);
  if (it != centers.end()) {
    centers.erase(it);
  }
}

std::vector<MeyersonSketch::NodePtr> &MeyersonSketch::clusters() {
  return centers;
}

void MeyersonSketch::ForEach(
    std::function<void(MeyersonSketch::NodePtr)> func) {
  for (auto node : centers) {
    func(node);
  }
}
