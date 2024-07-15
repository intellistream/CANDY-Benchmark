#ifndef SESMAE_INCLUDE_ALGORITHM_DATASTRUCTURE_MEYERSONSKETCH_HPP_
#define SESMAE_INCLUDE_ALGORITHM_DATASTRUCTURE_MEYERSONSKETCH_HPP_

#include "Algorithm/DataStructure/FeatureVector.hpp"
#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"
#include "Utils/Random.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace SESAME {
class MeyersonSketch : public std::enable_shared_from_this<MeyersonSketch> {
public:
  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  using SktchPtr = std::shared_ptr<MeyersonSketch>;

private:
  const SesameParam &param;
  std::vector<PointPtr> samples;
  bool has_sampled = false;
  Random r;
  NodePtr Process(PointPtr);
  NodePtr CreateCenter(PointPtr);
  std::vector<NodePtr> centers;
  int max_sketch_size_;
  double distance_denominator_;

public:
  MeyersonSketch(const SesameParam &param);
  void Init();
  NodePtr Insert(PointPtr input);
  NodePtr Insert(NodePtr node);
  void Remove(NodePtr node);
  std::vector<NodePtr> &clusters();
  void ForEach(std::function<void(NodePtr)> func);

public:
  struct Node : std::enable_shared_from_this<Node> {
    size_t timestamp = 0;
    int index = 0;
    const int dim;
    ClusteringFeatures cf;
    double costs_sum_dist = 0.0, costs_sum_sq_dist = 0.0;
    Node(SktchPtr s, PointPtr p)
        : dim(p->getDimension()), cf(p->getDimension()) {
      Update(p);
    }
    Node(PointPtr p) : dim(p->getDimension()), cf(p->getDimension()) {
      Update(p);
    }
    PointPtr Centroid() {
      auto c = GenericFactory::New<Point>(dim);
      for (int i = 0; i < dim; ++i)
        c->setFeatureItem(cf.ls[i] / cf.num, i);
      return c;
    }
    void Update(PointPtr point) {
      cf.num += point->sgn;
      double d = point->L2Dist(Centroid());
      costs_sum_dist += d * point->sgn;
      costs_sum_sq_dist += d * d * point->sgn;
      for (int i = 0; i < dim; ++i) {
        auto val = point->getFeatureItem(i);
        cf.ls[i] += val * point->sgn;
        cf.ss[i] += (val * val) * point->sgn;
      }
    }
    void Scale(double scale) {
      costs_sum_dist *= scale;
      costs_sum_sq_dist *= scale * scale;
      for (int i = 0; i < dim; ++i) {
        cf.ls[i] *= scale;
        cf.ss[i] *= scale * scale;
      }
    }
  };
};

std::pair<double, double>
guess_optimum_range_bounds(Random *r, const std::vector<PointPtr> &samples,
                           int window_size, int num_samples, int k);

} // namespace SESAME

#endif