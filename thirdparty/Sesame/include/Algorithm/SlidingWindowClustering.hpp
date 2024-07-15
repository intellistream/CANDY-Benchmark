#ifndef SLIDING_WINDOW_CLUSTERING_HPP_
#define SLIDING_WINDOW_CLUSTERING_HPP_

#include "Algorithm/Algorithm.hpp"
#include "Algorithm/Param.hpp"
#include "Utils/Random.hpp"

#include <list>
#include <optional>
#include <vector>

namespace SESAME {
void k_means_plus_plus(Random *r,
                       const std::vector<std::pair<PointPtr, double>> &instance,
                       int32_t k, std::vector<int32_t> *centers, double *cost);

// Class that handles the time stamps of a sliding window. The timestamps start
// from 0.
class SlidingWindowHandler {
public:
  SlidingWindowHandler(const int window_size)
      : curr_time_(-1), begin_window_(0), window_size_(window_size) {}

  // Increase time counter.
  void next() {
    curr_time_++;
    if (curr_time_ >= window_size_) {
      begin_window_++;
    }
    assert(curr_time_ - begin_window_ + 1 <= window_size_);
  }

  // Current time.
  inline int64_t curr_time() const { return curr_time_; }
  // Beginning of the window.
  inline int64_t begin_window() const { return begin_window_; }

private:
  int64_t curr_time_;
  int64_t begin_window_;
  const int32_t window_size_;
};

// This class implements an estimate of the total weight inserted after a
// certain time in a sliding window. The class assumes that the times of
// insertion are inserted in increasing order. This can be used to keep track of
// the number of items inserted in a bucket after a certain tiem or to keep
// track of the total cost of the bucket.
class ApproxTimeCountKeeper {
public:
  // Epsilon is the approximation error allowed (i.e., we allow 1+epsilon).
  explicit ApproxTimeCountKeeper(double epsilon) : epsilon_(epsilon) {
    assert(epsilon_ > 0);
  }
  ~ApproxTimeCountKeeper() {}

  // Increase the count by 'how_much' at time 'time'.
  void increase_total(const int64_t time, const double how_much) {
    for (auto &time_count : time_counts_) {
      assert(time_count.first < time);
      time_count.second += how_much;
    }

    time_counts_.push_back(std::make_pair(time, how_much));

    auto it = time_counts_.begin();
    // Compact indices if they are not too different.
    while (true) {
      auto it2 = it;
      // Move twice ahead if possible.
      if (it2 == time_counts_.end()) {
        break;
      }
      it2++;
      if (it2 == time_counts_.end()) {
        break;
      }
      auto it_middle = it2;
      it2++;
      if (it2 == time_counts_.end()) {
        break;
      }
      // Check if need to remove middle
      if (it->second <= (1.0 + epsilon_) * it2->second) {
        it_middle = time_counts_.erase(it_middle);
        it = it_middle;
      } else {
        ++it;
      }
    }
  }

  // Returns a 1+epsilon estimate of the total weight added to the counter after
  // begin_time.
  double total_after_time(const int64_t begin_time) const {
    double count = time_counts_.front().second;
    for (const auto &time_count : time_counts_) {
      if (time_count.first <= begin_time) {
        count = time_count.second;
      }
    }
    return count;
  }

  // Returns the count at 0.
  double count_at_0() const { return time_counts_.front().second; }

private:
  std::list<std::pair<int64_t, double>> time_counts_;
  // Approximation error.
  double epsilon_;
};

// This is the class that has to be implemented for a sliding window summary
// algorithm allowing composable sketches.
class SummaryAlg {
public:
  // window is the window length
  // k is the number of centers
  // lambda is the threshold used by the summary
  // gen is a random be generator
  SummaryAlg(Random *r, int64_t window, int32_t k, double lambda)
      : r(r), window_(window), k_(k), lambda_(lambda), is_empty_(true),
        first_element_time_(-1) {}
  virtual ~SummaryAlg() {}

  // Processes one point.
  void process_point(PointPtr point) {
    if (is_empty_) {
      is_empty_ = false;
    }
    process_point_impl(point);
  }

  // Resets the summary.
  void reset() {
    is_empty_ = true;
    first_element_time_ = -1;
    reset_impl();
  }

  Random *r;

  // Returns the time of the first element in the stream,
  int64_t first_element_time() const { return first_element_time_; }

  // Returns whether the stream is empty,
  bool is_empty() const { return is_empty_; }

  // These functions and the ones below must to be implemented by the derived
  // class.
  // Returns the solution and the cost.
  virtual void solution(vector<PointPtr> *sol, double *value) const = 0;

  // Output the solution resulting from a composion this summary (as summary
  // B -- right summary) with summary_A on the left.
  virtual void solution_left_composition(const SummaryAlg &summary_A,
                                         int64_t window_begin,
                                         vector<PointPtr> *sol,
                                         double *value) const = 0;

  // Output the result of composing this summary (as summary A) with  and empty
  // summary_B on the right.
  virtual void solution_right_composition_with_empty(int64_t window_begin,
                                                     vector<PointPtr> *sol,
                                                     double *value) const = 0;

protected:
  // Actual implementation of prcess_point
  virtual void process_point_impl(PointPtr point) = 0;
  // Actual implementation of reset. Notice that the class must be responsible
  // for keeping updated the number of items stored when the reset function is
  // called.
  virtual void reset_impl() = 0;

private:
  int32_t window_;
  int32_t k_;
  double lambda_;
  bool is_empty_;
  int64_t first_element_time_;
};

// Implementation of the KMeans summary algorithm for sliding window streams
// using the Meyerson sketch.
class KMeansSummary : public SummaryAlg {
public:
  KMeansSummary(Random *r, int64_t window, int32_t k,
                double optimum_upperbound_guess)
      : SummaryAlg(r, window, k, optimum_upperbound_guess) {
    max_sketch_size_ = std::pow(2, 2 * 2 + 1) * 4. * k *
                       (1. + std::log(window * 3)) * (1.0 + 1. / 0.5);
    distance_denominator_ =
        (optimum_upperbound_guess) / (k * (1. + std::log(window * 3)));
    // sketch_number_ = std::log(3 * 1. / error_prob);
    k_ = k;
    reset_impl();
  }

  virtual ~KMeansSummary() {}

  void solution(vector<PointPtr> *centers_solution,
                double *cost_solution) const override {
    centers_solution->clear();
    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<PointPtr> centers;
      bool ok = sketches_.at(i).solution(/*after_time=*/0, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }
    sketches_.at(best_pos).solution(/* after_time=*/0, centers_solution,
                                    cost_solution);
  }

  void solution_left_composition(const SummaryAlg &summary_A,
                                 int64_t window_begin, vector<PointPtr> *sol,
                                 double *value) const override {
    sol->clear();
    const KMeansSummary &summary_A_type =
        dynamic_cast<const KMeansSummary &>(summary_A);

    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    assert(summary_A_type.sketches_.size() == sketches_.size());

    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<PointPtr> centers;
      bool ok = sketches_.at(i).compose_left_solution(
          summary_A_type.sketches_.at(i), window_begin, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }

    sketches_.at(best_pos).compose_left_solution(
        summary_A_type.sketches_.at(best_pos), window_begin, sol, value);
  }

  void
  solution_right_composition_with_empty(int64_t window_begin,
                                        vector<PointPtr> *centers_solution,
                                        double *cost_solution) const override {
    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<PointPtr> centers;
      bool ok = sketches_.at(i).solution(window_begin, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }
    sketches_.at(best_pos).solution(window_begin, centers_solution,
                                    cost_solution);
  }

  // This class implements the well-known meyerson sketch for k-means with all
  // the bookkeping needed for the sliding window algorithm. This class
  // implements a single sketch.
  class MeyersonSketch {
  public:
    // max_num_centers is the maximum number of centers allowed in the sketch.
    // denominator_prob is the denominator in the probability of selecting a
    // point as center. epsilon_multiplicities is the epsilon factor used in the
    // estimate of the multiplicites of the centers. k is the target number of
    // centers. gen is a random source.
    MeyersonSketch(Random *r, const double max_num_centers,
                   const double denominator_prob,
                   const double epsilon_multiplicities, const int32_t k)
        : r(r), max_num_centers_(max_num_centers),
          denominator_prob_(denominator_prob),
          epsilon_multiplicities_(epsilon_multiplicities), k_(k),
          failed_sketch_(false) {}
    // Notice how the number of items stored is updated by the destructor.
    virtual ~MeyersonSketch() {}

    // Copy operator. Handles the number of items stored.
    MeyersonSketch(MeyersonSketch const &other) {
      r = other.r;
      max_num_centers_ = other.max_num_centers_;
      denominator_prob_ = other.denominator_prob_;
      epsilon_multiplicities_ = other.epsilon_multiplicities_;
      k_ = other.k_;
      // gen_ = other.gen_;
      failed_sketch_ = other.failed_sketch_;
      centers_ = other.centers_;
      // ITEMS_STORED += centers_.size();
      multiplicities_ = other.multiplicities_;
      costs_sum_dist_ = other.costs_sum_dist_;
      costs_sum_sq_dist_ = other.costs_sum_sq_dist_;
    }

    MeyersonSketch &operator=(MeyersonSketch const &other) {
      r = other.r;
      max_num_centers_ = other.max_num_centers_;
      denominator_prob_ = other.denominator_prob_;
      epsilon_multiplicities_ = other.epsilon_multiplicities_;
      k_ = other.k_;
      // gen_ = other.gen_;
      failed_sketch_ = other.failed_sketch_;
      // The previous number of centers is freed.
      // ITEMS_STORED -= centers_.size();
      centers_ = other.centers_;
      // ITEMS_STORED += centers_.size();
      multiplicities_ = other.multiplicities_;
      costs_sum_dist_ = other.costs_sum_dist_;
      costs_sum_sq_dist_ = other.costs_sum_sq_dist_;
      return *this;
    }

    // Add a point
    void add_point(const PointPtr &point) {
      if (failed_sketch_) {
        return;
      }
      if (centers_.empty()) {
        create_center(point);
        return;
      }
      double min_distance = std::numeric_limits<double>::max();
      int64_t best_center;
      for (int i = 0; i < centers_.size(); i++) {
        double d = point->L2Dist(centers_[i]);
        if (d < min_distance) {
          best_center = i;
          min_distance = d;
        }
      }
      bool open_new = r->bernoulli(
          std::min(1.0, std::pow(min_distance, 2) / denominator_prob_));
      if (open_new) {
        create_center(point);
      } else {
        add_point_to_center(point, best_center, min_distance);
      }
    }

    // Returns the estimate of the multiplicities of the centers assigned after
    // a certain time.
    void weighted_centers(const int after_time, std::vector<PointPtr> *centers,
                          std::vector<double> *weights) {
      centers->clear();
      weights->clear();
      assert(multiplicities_.size() == centers_.size());
      centers->reserve(multiplicities_.size());
      weights->reserve(multiplicities_.size());
      for (int i = 0; i < multiplicities_.size(); i++) {
        centers->push_back(centers_[i]);
        weights->push_back(multiplicities_[i].total_after_time(after_time));
      }
    }

    // Computes a solution of the state of the Meyerson Sketch after time
    // "after_time".
    bool solution(const int after_time, std::vector<PointPtr> *centers,
                  double *cost) const {
      centers->clear();
      *cost = 0.0;

      if (failed_sketch_) {
        centers = nullptr;
        *cost = std::numeric_limits<double>::max();
        return false;
      }

      if (after_time == 0 && precomputed_cost.has_value()) {
        // No need for recomputation.
        assert(precomputed_solution.has_value());
        *centers = precomputed_solution.value();
        *cost = precomputed_cost.value();
        return true;
      }

      // Create weighted instance.
      std::vector<std::pair<PointPtr, double>> instance;
      assert(multiplicities_.size() == centers_.size());
      assert(multiplicities_.size() == costs_sum_dist_.size());
      assert(multiplicities_.size() == costs_sum_sq_dist_.size());
      instance.reserve(multiplicities_.size());
      for (int i = 0; i < multiplicities_.size(); i++) {
        instance.push_back(std::make_pair(
            centers_[i], multiplicities_[i].total_after_time(after_time)));
      }
      double cost_instance = 0;

      // Solve k-means.
      std::vector<int32_t> pos_centers;
      k_means_plus_plus(r, instance, k_, &pos_centers, &cost_instance);
      for (const auto &pos : pos_centers) {
        assert(pos < instance.size());
        centers->push_back(instance.at(pos).first);
      }
      // Adding the cost of the sketch itset (sum of squared distances)
      for (int i = 0; i < costs_sum_sq_dist_.size(); i++) {
        cost_instance += costs_sum_sq_dist_.at(i).total_after_time(after_time);
      }
      // Adding 2*dist_center_to_sketch(sum dist to sketch center)
      for (int i = 0; i < centers_.size(); i++) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &center : *centers) {
          min_distance = std::min(min_distance, centers_[i]->L2Dist(center));
        }
        cost_instance += 2.0 * min_distance *
                         costs_sum_dist_.at(i).total_after_time(after_time);
      }
      *cost = cost_instance;

      // Update the precomputed solution.
      if (after_time == 0) {
        precomputed_solution.reset();
        precomputed_solution.emplace(*centers);
        precomputed_cost = *cost;
        last_precomputed_0_multiplicities_.clear();
        for (const auto &mult : multiplicities_) {
          last_precomputed_0_multiplicities_.push_back(mult.count_at_0());
        }
        last_precomputed_0_costs_sum_dist_.clear();
        for (const auto &mult : costs_sum_dist_) {
          last_precomputed_0_costs_sum_dist_.push_back(mult.count_at_0());
        }
        last_precomputed_0_costs_sum_sq_dist_.clear();
        for (const auto &mult : costs_sum_sq_dist_) {
          last_precomputed_0_costs_sum_sq_dist_.push_back(mult.count_at_0());
        }
      }

      return true;
    }

    // Compose this sketch as summary_B  with summary_A to the left.
    bool compose_left_solution(const MeyersonSketch &summary_A,
                               int64_t window_begin,
                               std::vector<PointPtr> *centers,
                               double *cost) const {
      centers->clear();
      if (failed_sketch_ || summary_A.failed_sketch_) {
        centers = nullptr;
        *cost = std::numeric_limits<double>::max();
        return false;
      }

      std::vector<std::pair<PointPtr, double>> instance;

      // Add all centers in B
      assert(summary_A.multiplicities_.size() == summary_A.centers_.size());
      assert(summary_A.multiplicities_.size() ==
             summary_A.costs_sum_dist_.size());
      assert(summary_A.multiplicities_.size() ==
             summary_A.costs_sum_sq_dist_.size());
      instance.reserve(summary_A.multiplicities_.size() +
                       multiplicities_.size());
      for (int i = 0; i < summary_A.multiplicities_.size(); i++) {
        instance.push_back(std::make_pair(
            summary_A.centers_.at(i),
            summary_A.multiplicities_.at(i).total_after_time(window_begin)));
      }

      assert(multiplicities_.size() == centers_.size());
      assert(multiplicities_.size() == costs_sum_dist_.size());
      assert(multiplicities_.size() == costs_sum_sq_dist_.size());
      instance.reserve(multiplicities_.size());
      for (int i = 0; i < multiplicities_.size(); i++) {
        instance.push_back(std::make_pair(
            centers_.at(i), multiplicities_.at(i).total_after_time(0)));
      }
      double cost_instance = 0;

      std::vector<int32_t> pos_centers;
      k_means_plus_plus(r, instance, k_, &pos_centers, &cost_instance);
      for (const auto &pos : pos_centers) {
        assert(pos < instance.size());
        centers->push_back(instance.at(pos).first);
      }
      // Adding the cost of the sketch itself
      for (int i = 0; i < summary_A.costs_sum_sq_dist_.size(); i++) {
        cost_instance +=
            summary_A.costs_sum_sq_dist_.at(i).total_after_time(window_begin);
      }
      for (int i = 0; i < costs_sum_sq_dist_.size(); i++) {
        cost_instance += costs_sum_sq_dist_.at(i).total_after_time(0);
      }
      // Adding the cost of 2*d(sum dist)
      for (int i = 0; i < centers_.size(); i++) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &center : *centers) {
          min_distance = std::min(min_distance, centers_[i]->L2Dist(center));
        }
        cost_instance +=
            2.0 * min_distance * costs_sum_dist_.at(i).total_after_time(0);
      }
      for (int i = 0; i < summary_A.centers_.size(); i++) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &center : *centers) {
          min_distance =
              std::min(min_distance, summary_A.centers_[i]->L2Dist(center));
        }
        cost_instance +=
            2.0 * min_distance *
            summary_A.costs_sum_dist_.at(i).total_after_time(window_begin);
      }
      *cost = cost_instance;
      return true;
    }

  private:
    // Initialize a new center
    void create_center(const PointPtr &point) {
      if (failed_sketch_ || centers_.size() == max_num_centers_) {
        failed_sketch_ = true;
        return;
      }
      // Invalidate precomputed solution.
      precomputed_cost.reset();

      // ++ITEMS_STORED;
      centers_.push_back(point);
      multiplicities_.push_back(ApproxTimeCountKeeper(epsilon_multiplicities_));
      multiplicities_.back().increase_total(point->index, 1);
      costs_sum_dist_.push_back(ApproxTimeCountKeeper(epsilon_multiplicities_));
      costs_sum_dist_.back().increase_total(point->index, 0.0);
      costs_sum_sq_dist_.push_back(
          ApproxTimeCountKeeper(epsilon_multiplicities_));
      costs_sum_sq_dist_.back().increase_total(point->index, 0.0);
    }

    // Assign a point to a center.
    void add_point_to_center(const PointPtr &point,
                             const int32_t center_position,
                             const double distance) {
      multiplicities_[center_position].increase_total(point->index, 1);
      costs_sum_dist_[center_position].increase_total(point->index, distance);
      costs_sum_sq_dist_[center_position].increase_total(
          point->index, ::std::pow(distance, 2));

      if (precomputed_cost.has_value()) {
        if (multiplicities_.at(center_position).count_at_0() >
                last_precomputed_0_multiplicities_.at(center_position) *
                    (1.0 + kErrorMarginPrecomputedSolution) ||
            costs_sum_dist_.at(center_position).count_at_0() >
                last_precomputed_0_costs_sum_dist_.at(center_position) *
                    (1.0 + kErrorMarginPrecomputedSolution) ||
            costs_sum_sq_dist_.at(center_position).count_at_0() >
                last_precomputed_0_costs_sum_sq_dist_.at(center_position) *
                    (1.0 + kErrorMarginPrecomputedSolution)) {
          // Invalidate precomputed solution.
          precomputed_cost.reset();
        }
      }
    }

    double max_num_centers_;
    // This is the factor used in p = min(dist / denominator_prob_, 1) for
    // inserting a point.
    double denominator_prob_;
    double epsilon_multiplicities_;
    int32_t k_;
    bool failed_sketch_;
    vector<PointPtr> centers_;
    vector<ApproxTimeCountKeeper> multiplicities_;
    vector<ApproxTimeCountKeeper> costs_sum_dist_;
    vector<ApproxTimeCountKeeper> costs_sum_sq_dist_;
    Random *r;

    // Used to avoid recomputation if the solution has not significantly
    // changed.
    mutable std::optional<double> precomputed_cost;
    mutable std::optional<std::vector<PointPtr>> precomputed_solution;
    mutable vector<double> last_precomputed_0_multiplicities_;
    mutable vector<double> last_precomputed_0_costs_sum_dist_;
    mutable vector<double> last_precomputed_0_costs_sum_sq_dist_;

    // Margin of error in the precomputed solution.
    static constexpr double kErrorMarginPrecomputedSolution = 0.05;
  };

protected:
  void process_point_impl(PointPtr point) override {
    for (auto &sketch : sketches_) {
      sketch.add_point(point);
    }
  }

  void reset_impl() override {
    sketches_.clear();
    sketches_.reserve(kSketchNumber);
    for (int i = 0; i < kSketchNumber; i++) {
      MeyersonSketch sketch(r, max_sketch_size_, distance_denominator_,
                            kEpsilonMult, k_);
      sketches_.push_back(sketch);
    }
  }

private:
  int32_t max_sketch_size_;
  double distance_denominator_;
  int32_t k_;

  std::vector<MeyersonSketch> sketches_;

  // Epsilon factor in the estimation of the multiplicities.
  static constexpr double kEpsilonMult = 0.01;
  // Number of sketches used. As stated in the paper, we use a single sketch as
  // in practice is sufficient to get good results.
  static constexpr int32_t kSketchNumber = 1;
};

// This class implements the procedure to update the pair of summaries
// associated with a threshold lambda.
template <typename SummaryAlg> class OneThresholdsPairSummaryAlg {
public:
  // window is the window size.
  // k is the number of clusters
  // lambda is the threshold for the.
  OneThresholdsPairSummaryAlg(Random *r, int64_t window, int32_t k,
                              double lambda)
      : r(r), window_(window), k_(k), lambda_(lambda),
        summary_A_(r, window_, k_, lambda_),
        summary_B_(r, window_, k_, lambda_) {
    assert(lambda >= 0);
  }

  virtual ~OneThresholdsPairSummaryAlg() {}

  void process_point(PointPtr point) {
    // Add the point to summary B.
    SummaryAlg copy_B = summary_B_;
    summary_B_.process_point(point);
    vector<PointPtr> unused_centers_solution;
    double value;
    summary_B_.solution(&unused_centers_solution, &value);
    // If the value of summary B is < lambda then it is kept in B. Otherwise the
    // new element is the start of a new summary B and summary A is assigned the
    // old value of B.
    if (value >= lambda_) {
      summary_A_ = copy_B;
      summary_B_.reset();
      summary_B_.process_point(point);
      first_element_B_ = point;
    }
  }

  const SummaryAlg *get_summary_A() const { return &summary_A_; }

  const SummaryAlg *get_summary_B() const { return &summary_B_; }

private:
  const int32_t window_;
  const int32_t k_;
  const double lambda_;
  SummaryAlg summary_A_;
  SummaryAlg summary_B_;
  PointPtr first_element_B_;
  Random *r;
};

// This runs the algorithmic framework for our sliding window algorithm.
// The parameters are:
//  delta which is the one in (1+delta) used for the grid of costs
//  window which is the window size
//  begin grid which is used for the begin of the grid
//  end grid which is the end of the grid.
template <typename SummaryAlg> class FrameworkAlg {
public:
  FrameworkAlg(Random *r, int64_t window, int32_t k, double delta,
               double begin_grid, double end_grid)
      : r(r), window_(window), k_(k), delta_(delta), begin_grid_(begin_grid),
        end_grid_(end_grid), window_handler_(window) {
    assert(delta > 0);
    assert(begin_grid < end_grid);
    assert(window_ > 1);

    // Initializes the summary thresholds.
    double lambda = begin_grid_;
    while (lambda <= end_grid_ * (1.0 + delta)) {
      threshold_algs_.push_back(
          OneThresholdsPairSummaryAlg<SummaryAlg>(r, window_, k_, lambda));
      lambda *= (1 + delta);
    }
  }
  // Disllow copy and assign.
  FrameworkAlg(FrameworkAlg const &) = delete;
  void operator=(FrameworkAlg const &x) = delete;

  // Processes a point.
  void process_point(PointPtr point) {
    window_handler_.next();
    for (auto &threshold_alg : threshold_algs_) {
      threshold_alg.process_point(point);
    }
  }

  // Outputs the solution and its cost.
  void solution(vector<PointPtr> *solution, double *value) {
    // Contrary to the simplified version in the main body of the paper, the
    // algorithm considers any valid solution (i.e. computer over the entire
    // window) and outputs the one with the lowest estimate of cost, this only
    // improves the results.
    // Check if there is B_lambda that is = active_window.
    double min_cost = std::numeric_limits<double>::max();
    vector<PointPtr> best_solution;

    for (const auto &threshold_alg : threshold_algs_) {
      if (!threshold_alg.get_summary_B()->is_empty() &&
          threshold_alg.get_summary_B()->first_element_time() ==
              window_handler_.begin_window()) {
        threshold_alg.get_summary_B()->solution(solution, value);
        if (*value < min_cost) {
          min_cost = *value;
          best_solution = *solution;
        }
      } else if (!threshold_alg.get_summary_B()->is_empty() &&
                 threshold_alg.get_summary_B()->first_element_time() <
                     window_handler_.begin_window()) {
        threshold_alg.get_summary_B()->solution_right_composition_with_empty(
            window_handler_.begin_window(), solution, value);
        if (*value < min_cost) {
          min_cost = *value;
          best_solution = *solution;
        }
      }
    }

    // This is a backup in case the algorithmic guarantess of the summaries
    // fails (or if the lower or upper bound are wrong) which is possible in
    // practice. It will output the solution of the sketch (subset of the window
    // that has longest possible history).
    vector<PointPtr> oldest_sketch_solution;
    double oldest_sketch_cost;
    int64_t oldest_begin_first = std::numeric_limits<int64_t>::max();

    for (const auto &threshold_alg : threshold_algs_) {
      // A is not empty and it is not a subset of W for the smallest lambda.
      if (!threshold_alg.get_summary_A()->is_empty() &&
          threshold_alg.get_summary_A()->first_element_time() <
              window_handler_.begin_window()) {
        // Either W is a strict subset of B.
        if (threshold_alg.get_summary_B()->first_element_time() <
            window_handler_.begin_window()) {
          threshold_alg.get_summary_B()->solution_right_composition_with_empty(
              window_handler_.begin_window(), solution, value);
          if (*value < min_cost) {
            min_cost = *value;
            best_solution = *solution;
          }
        } else {
          threshold_alg.get_summary_B()->solution_left_composition(
              *threshold_alg.get_summary_A(), window_handler_.begin_window(),
              solution, value);
          if (*value < min_cost) {
            min_cost = *value;
            best_solution = *solution;
          }
        }
      } else { // backup for the case with no guarantees.
        if (!threshold_alg.get_summary_A()->is_empty() &&
            threshold_alg.get_summary_A()->first_element_time() <
                oldest_begin_first) {
          oldest_begin_first =
              threshold_alg.get_summary_A()->first_element_time();
          threshold_alg.get_summary_B()->solution_left_composition(
              *threshold_alg.get_summary_A(), 0, &oldest_sketch_solution,
              &oldest_sketch_cost);
        }
      }
    }
    *solution = best_solution;
    *value = min_cost;
    if (min_cost < std::numeric_limits<double>::max()) {
      return;
    }
    // This point is reached in case of failure of the sketches to give approx.
    // guarantees or if the upper-lower bounds of the cost are wrong. In this
    // case
    *solution = oldest_sketch_solution;
    *value = oldest_sketch_cost;
    assert(oldest_begin_first < std::numeric_limits<int64_t>::max());
  }

private:
  Random *r;
  const int64_t window_;
  const int32_t k_;
  const double delta_;
  const double begin_grid_;
  const double end_grid_;
  SlidingWindowHandler window_handler_;
  std::vector<OneThresholdsPairSummaryAlg<SummaryAlg>> threshold_algs_;
};

class SlidingWindowClustering : public Algorithm {
private:
  Random r;
  std::vector<PointPtr> samples;
  bool has_sampled = false;
  std::shared_ptr<FrameworkAlg<KMeansSummary>> framework;
  int count = 0;

public:
  SlidingWindowClustering(param_t &cmd_params);

  ~SlidingWindowClustering();

  void Init() override;

  void RunOnline(PointPtr input) override;

  void RunOffline(DataSinkPtr sinkPtr) override;
};

} // namespace SESAME

#endif
