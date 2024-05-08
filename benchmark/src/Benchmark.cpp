/*! \file Benchmark.h*/

/**
 * @brief This is the main entry point of the entire program.
 * We use this as the entry point for benchmarking.
 */
#include <Utils/UtilityFunctions.h>
#include <include/papi_config.h>
#include <Utils/ThreadPerf.hpp>
#include <Utils/Meters/MeterTable.h>
#include <CANDY/OnlinePQIndex/IVFTensorEncodingList.h>
#include <CANDY.h>
#include <map>
using namespace std;
using namespace INTELLI;
using namespace DIVERSE_METER;

// Distance function type (function pointer)
using DistanceFunction = torch::Tensor (*)(const torch::Tensor &, const torch::Tensor &);
torch::Tensor euclideanDistance(const torch::Tensor &a, const torch::Tensor &b) {
  // Assuming 'a' has shape (N, D) and 'b' has shape (K, D)
  torch::Tensor expandedA = a.unsqueeze(1).expand({a.size(0), b.size(0), a.size(1)});
  torch::Tensor expandedB = b.unsqueeze(0).expand({a.size(0), b.size(0), b.size(1)});

  return (expandedA - expandedB).pow(2).sum(2).sqrt();
}
// Function to perform K-Means clustering
torch::Tensor kMeansClustering(torch::Tensor &trainSet,
                               int64_t k,
                               int64_t maxIterations,
                               DistanceFunction distanceFunc = euclideanDistance,
                               bool usingCuda = true) {
  // Initialize centroids randomly
  torch::manual_seed(42);
  // Probability distribution
  int64_t n = trainSet.size(0);
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform

  // Sample k indices from range 0 to n for given probability distribution
  torch::Tensor indices = torch::multinomial(probs, k, true);
  // return indices;
  auto centroids = trainSet.index_select(0, indices);
  auto data = trainSet;
  if (usingCuda) {
    centroids = centroids.to(torch::kCUDA);
    data = data.to(torch::kCUDA);
  }
  for (int64_t iteration = 0; iteration < maxIterations; ++iteration) {
    // Assign each data point to the nearest centroid
    torch::Tensor distances = distanceFunc(data, centroids);
    torch::Tensor labels = std::get<1>(torch::min(distances, /*dim=*/1));

    // Update centroids based on the mean of points in each cluster
    for (int64_t cluster = 0; cluster < k; ++cluster) {
      auto rowMask = labels.eq(cluster);
      torch::Tensor clusterPoints = data.index({rowMask.nonzero().squeeze()});
      if (clusterPoints.size(0) > 0) {
        centroids[cluster] = clusterPoints.mean(0);
      }
    }
  }
  if (usingCuda) {
    return centroids.to(torch::kCPU);
  }
  return centroids;
}
int64_t classifySingleRow(torch::Tensor &rowTensor,
                          torch::Tensor &myCentroids,
                          DistanceFunction distanceFunc = euclideanDistance) {
  torch::Tensor distances = distanceFunc(rowTensor, myCentroids);
  std::cout << distances << std::endl;
// Find the index of the centroid with the minimum distance
  int64_t clusterIndex = std::get<1>(torch::min(distances, /*dim=*/1))[0].item<int>();

  return clusterIndex;
}

int main(int argc, char **argv) {
  string configName, outPrefix = "";
  if (argc >= 2) {
    configName += argv[1];
  } else {
    configName = "config.csv";
  }
  // Example data (2D points)
  torch::manual_seed(999);
  /*torch::Tensor data = torch::randn({10, 7});

  // Number of clusters (k)
  int k = 3;

  // Maximum number of iterations
  int maxIterations = 100;
  auto start2 = std::chrono::high_resolution_clock::now();
  // Perform K-Means clustering
  torch::Tensor centroids2 = kMeansClustering(data, k, maxIterations,euclideanDistance, false);
  auto cpuTime = chronoElapsedTime(start2);
  //return 0;
  // runSingleThreadTest(configName);

  // Display the final centroids
  auto row=data.slice(0,5,6);
  std::cout << "Final Centroids (CPU):\n"  <<"time"<<cpuTime<< std::endl;
  int64_t  idx=classifySingleRow(row,centroids2);
  std::cout << row  <<"belongs to cluster "<<idx<<"centoid"<<centroids2.slice(0,idx,idx+1)<< std::endl;*/

  CANDY::IVFListCell cell0;
  auto t0 = torch::rand({1, 4});
  auto t1 = torch::rand({1, 4});
  auto t2 = torch::rand({1, 4});
  auto t3 = torch::rand({1, 4});
  auto encode = INTELLI::IntelliTensorOP::tensorToFlatBin(&t0);
  cell0.setEncode(encode);
  auto tag = cell0.getEncode();
  if (tag == encode) {
    std::cout << "encode successed" << std::endl;
  }
  cell0.insertTensor(t0);
  cell0.insertTensor(t1);
  cell0.insertTensor(t2);
  std::cout << "the inserted tensors are\n" << cell0.getAllTensors() << std::endl;
  cell0.deleteTensor(t1);
  std::cout << "after delete 1, the inserted tensors are\n" << cell0.getAllTensors() << std::endl;
  CANDY::IVFTensorEncodingList ivf0;
  std::cout << "/**Go to ivf list test***/\n" << std::endl;
  ivf0.init(2, encode.size());
  ivf0.insertTensorWithEncode(t0, encode, 1);
  auto encode2 = INTELLI::IntelliTensorOP::tensorToFlatBin(&t1);
  auto encode3 = encode;
  encode3[0] = encode3[0] + 1;
  ivf0.insertTensorWithEncode(t1, encode2, 1);

  ivf0.insertTensorWithEncode(t2, encode2, 0);
  ivf0.insertTensorWithEncode(t3, encode2, 0);
  std::cout << "get 4 tensors from ivf\n" << ivf0.getMinimumNumOfTensors(t0, encode, 1, 4) << std::endl;
  std::cout << "get 2 tensors from ivf\n" << ivf0.getMinimumNumOfTensors(t0, encode, 1, 2) << std::endl;
  std::cout << "get 1 tensors from ivf\n" << ivf0.getMinimumNumOfTensors(t0, encode, 1, 1) << std::endl;

  return 0;
}
