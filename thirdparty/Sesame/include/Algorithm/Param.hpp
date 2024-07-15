#ifndef PARAM_HPP_
#define PARAM_HPP_

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace SESAME {
enum AlgoType {
  BirchType = 0,
  StreamKMeansType = 1,
  CluStreamType = 2,
  DenStreamType = 3,
  DBStreamType = 4,
  EDMStreamType = 5,
  DStreamType = 6,
  SLKMeansType = 7,
  BenneType = 8,
  G1Stream = 21,
  G2Stream,
  G3Stream,
  G4Stream,
  G5Stream,
  G6Stream,
  G7Stream,
  G8Stream,
  G9Stream,
  G10Stream,
  G11Stream,
  G12Stream,
  G13Stream,
  G14Stream,
  G15Stream,
  G16Stream
};

extern char const *algo_names[64];
extern char const *benne_suffix[4];

enum BenneObj {
  balance = 0,
  accuracy = 1,
  efficiency = 2,
  accuracy_no_migration = 3
};

struct BenneThreshold {
  int queue_size = 10000;    // queue size for auto detection
  int dim = 30;              // above is high dimension
  double variance = 100;     // above is high concept drift
  int outliers_num = 200;    // above is many outliers
  double outliers_dist = 50; // above is outlier
};

struct param_t {
  int num_points = 500, dim = 2, num_clusters = 2;
  int arr_rate = 0;
  bool time_decay = false;
  size_t coreset_size = 100;
  int seed = 1;
  bool fast_source = false;
  bool store = true;

  std::string input_file = "datasets/CoverType.txt", output_file = "sesame.out";
  AlgoType algo;

  int num_last_arr = 60,
      time_window = 6; // also used in timer outlier detection
  size_t time_interval = 100;
  int num_online_clusters = 10;

  int buf_size = 500, offline_time_window = 100;

  // CF Tree
  int max_in_nodes = 3, max_leaf_nodes = 3;
  double distance_threshold = 3550;

  // used in DBSCAN
  unsigned int min_points = 10;
  double epsilon = 50;

  // used in DenStream(unique)
  double base = 2, lambda = 1, mu = 7;
  double beta = 0.0021; // Also used in DStream, but different meaning

  // EDMStream
  double delta = 10;
  int num_cache = 100, opt = 2;

  // used in DBStream
  double radius = 0.1, min_weight, alpha = 0.998;
  size_t clean_interval = 2500; // also used in timer outlier detection

  // used in DStream
  double cm = 5.0, cl = 0.8;
  double grid_width = 12.0;

  // used in design aspect
  bool run_offline = true; // determine whether to run the offline refinement
  bool run_eval = true;
  bool run_cmm = false, run_pur = true, run_nmi = false;
  //    bool run_group = true;
  int landmark =
      1000; // this is the index of landmark point[start from 0](determine
            // to process the algorithm from which algorithm)
  int sliding = 10; // since we test the count-based sliding window, this is
                    // the count number
  double outlier_distance_threshold = 1000; // the max distance of the incoming
                                            // point to its nearest clusters
  double outlier_density_threshold = 100;   // the density value of the point to
                                            // be treated as an outlier
  double neighbor_distance = 200; // the distance value of the point to judge
                                  // neighborhoods
  int outlier_cap = 5;            // transfer outlier cluster and true cluster
  bool kmeanspp = true; // whether use kmeans++ to initialize the centroids
  int k = 2;            // number of k in kmeans / kmeanspp

  double delta_grid =
      0.2; // The delta parameter used int the grid for guessing the optimum.
  int num_samples = 100; // The number of samples used in the grid for guessing
                         // the optimum.

  size_t num_res = 0;

  BenneObj obj = (BenneObj)0;
  BenneThreshold benne_threshold;

  void Print() {
    std::cout << "algo_id: " << algo << std::endl;
    std::cout << "algo: \"" << Name() << "\"" << std::endl;
    std::cout << "workload: \"" << Workload() << "\"" << std::endl;
    std::cout << "num_points: " << num_points << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "num_clusters: " << num_clusters << std::endl;
    std::cout << "arr_rate: " << arr_rate << std::endl;
    std::cout << "max_in_nodes: " << max_in_nodes << std::endl;
    std::cout << "max_leaf_nodes: " << max_leaf_nodes << std::endl;
    std::cout << "distance_threshold: " << distance_threshold << std::endl;
    std::cout << "seed: " << seed << std::endl;
    std::cout << "coreset_size: " << coreset_size << std::endl;
    std::cout << "radius: " << radius << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "buf_size: " << buf_size << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "lambda: " << lambda << std::endl;
    std::cout << "clean_interval: " << clean_interval << std::endl;
    std::cout << "min_weight: " << min_weight << std::endl;
    std::cout << "base: " << base << std::endl;
    std::cout << "cm: " << cm << std::endl;
    std::cout << "cl: " << cl << std::endl;
    std::cout << "grid_width: " << grid_width << std::endl;
    std::cout << "min_points: " << min_points << std::endl;
    std::cout << "epsilon: " << epsilon << std::endl;
    std::cout << "mu: " << mu << std::endl;
    std::cout << "num_last_arr: " << num_last_arr << std::endl;
    std::cout << "time_window: " << time_window << std::endl;
    std::cout << "num_online_clusters: " << num_online_clusters << std::endl;
    std::cout << "delta_grid: " << delta_grid << std::endl;
    std::cout << "num_samples: " << num_samples << std::endl;
    std::cout << "landmark: " << landmark << std::endl;
    std::cout << "sliding: " << sliding << std::endl;
    std::cout << "outlier_distance_threshold: " << outlier_distance_threshold
              << std::endl;
    std::cout << "outlier_cap: " << outlier_cap << std::endl;
    std::cout << "outlier_density_threshold: " << outlier_density_threshold
              << std::endl;
    std::cout << "neighbor_distance: " << neighbor_distance << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "run_offline: " << run_offline << std::endl;
    std::cout << "obj: " << obj << std::endl;
    std::cout << "queue_size_threshold: " << benne_threshold.queue_size
              << std::endl;
    std::cout << "dim_threshold: " << benne_threshold.dim << std::endl;
    std::cout << "variance_threshold: " << benne_threshold.variance
              << std::endl;
    std::cout << "outliers_num_threshold: " << benne_threshold.outliers_num
              << std::endl;
    std::cout << "outliers_dist_threshold: " << benne_threshold.outliers_dist
              << std::endl;
  }
  std::string Workload() { return std::filesystem::path(input_file).stem(); }
  std::string Name() {
    if (algo == BenneType) {
      return std::string(algo_names[algo]) + benne_suffix[obj];
    }
    return algo_names[algo];
  }
};

using SesameParam = param_t;

} // namespace SESAME

#endif // PARAM_HPP_