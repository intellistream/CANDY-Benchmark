#include <Utils/ComputeGT/StepwiseGT.hpp>
#include <Utils/ComputeGT/StepwiseRecall.hpp>

#include <iostream>
#include <string>
#include <mkl.h>   
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void COMPUTE_GT::computeL2sq(float* pointsL2sq, const float* matrix, const int64_t numPoints, const uint64_t dim) {
#pragma omp parallel for schedule(static, 65536)
  for (int64_t i = 0; i < numPoints; ++i) {
    pointsL2sq[i] = cblas_sdot((int64_t)dim, matrix + i * dim, 1, matrix + i * dim, 1);
  }
}

void COMPUTE_GT::distSqToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                                  const float* pointsL2sq, size_t nqueries, const float* queries,
                                  const float* queriesL2sq, float* onesVec) {
  bool onesVecAlloc = false;
  if (onesVec == nullptr) {
    onesVec = new float[std::max(npoints, nqueries)]();
    std::fill_n(onesVec, std::max(npoints, nqueries), 1.0f);
    onesVecAlloc = true;
  }
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, 
                points, dim, queries, dim, (float)0.0, distMatrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, 
                pointsL2sq, npoints, onesVec, nqueries, (float)1.0, distMatrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, 
                onesVec, npoints, queriesL2sq, nqueries, (float)1.0, distMatrix, npoints);

  if (onesVecAlloc)
      delete[] onesVec;
}

void COMPUTE_GT::innerProdToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                                    size_t nqueries, const float* queries, float* onesVec) {
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, -1.0f,
              points, dim, queries, dim, 0.0f, distMatrix, npoints);
}

void COMPUTE_GT::exactKnn(const size_t dim, const size_t k, size_t* closestPoints, float* distClosestPoints,
                            size_t npoints, float* pointsIn, size_t nqueries, 
                            float* queriesIn, COMPUTE_GT::Metric metric) {
  float* pointsL2sq = new float[npoints];
  float* queriesL2sq = new float[nqueries];
  computeL2sq(pointsL2sq, pointsIn, npoints, dim);
  computeL2sq(queriesL2sq, queriesIn, nqueries, dim);

  float* distMatrix = new float[npoints * nqueries];

  if (metric == COMPUTE_GT::Metric::L2) {
    distSqToPoints(dim, distMatrix, npoints, pointsIn, pointsL2sq, nqueries, queriesIn, queriesL2sq);
  } else {
    innerProdToPoints(dim, distMatrix, npoints, pointsIn, nqueries, queriesIn);
  }

#pragma omp parallel for schedule(dynamic, 16)
  for (size_t q = 0; q < nqueries; ++q) {
    std::priority_queue<std::pair<float, size_t>> pq;
    for (size_t p = 0; p < npoints; ++p) {
      float dist = distMatrix[p + q * npoints];
      if (pq.size() < k) {
        pq.emplace(dist, p);
      } else if (dist < pq.top().first) {
        pq.pop();
        pq.emplace(dist, p);
      }
    }
    for (int i = k - 1; i >= 0; --i) {
      closestPoints[q * k + i] = pq.top().second;
      distClosestPoints[q * k + i] = pq.top().first;
      pq.pop();
    }
  }

  delete[] distMatrix;
  delete[] pointsL2sq;
  delete[] queriesL2sq;
}

void COMPUTE_GT::saveGTVectorsAsFile(const std::string& filename, int step, float* queryVectors, float* gtVectors,
                                      size_t npts, size_t ndims) {
  std::ofstream writer(filename, std::ios::binary | std::ios::app);
  if (!writer) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // write step, npts and ndims
  uint64_t step64 = static_cast<uint64_t>(step);
  writer.write(reinterpret_cast<const char*>(&step64), sizeof(step64));
  writer.write(reinterpret_cast<const char*>(&npts), sizeof(npts));
  writer.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));

  // write query vectors
  writer.write(reinterpret_cast<const char*>(queryVectors), npts * ndims * sizeof(float));

  // write gt vectors
  writer.write(reinterpret_cast<const char*>(gtVectors), npts * ndims * sizeof(float));

  writer.close();
}

void COMPUTE_GT::calcStepwiseGT(const std::string& baseFile, const std::string& queryFile,
                                  const std::string& gtFile, size_t k, 
                                  const std::string& distFn, size_t batchSize) {
  COMPUTE_GT::Metric metric;
  if (distFn == "l2") {
    metric = COMPUTE_GT::Metric::L2;
  } else if (distFn == "mips") {
    metric = COMPUTE_GT::Metric::INNER_PRODUCT;
  } else if (distFn == "cosine") {
    metric = COMPUTE_GT::Metric::COSINE;
  } else {
    std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
    return;
  }

  float* baseData = nullptr;
  size_t npoints, dim;
  loadBinAsFloat<float>(baseFile.c_str(), baseData, npoints, dim, 0);

  float* queryData = nullptr;
  size_t nqueries;
  loadBinAsFloat<float>(queryFile.c_str(), queryData, nqueries, dim, 0);

  size_t currentPoints = 0;
  size_t step = 0;

  while (currentPoints < npoints) {
    size_t insertCount = std::min(batchSize, npoints - currentPoints);
    float* batchVectors = baseData + currentPoints * dim;
    currentPoints += insertCount;
    step++;

    float* gtVectors = new float[nqueries * dim];
    size_t* closestPoints = new size_t[nqueries * k];
    float* distClosestPoints = new float[nqueries * k];

    exactKnn(dim, k, closestPoints, distClosestPoints, currentPoints, baseData, nqueries, queryData, metric);

    for (size_t i = 0; i < nqueries; i++) {
      size_t gtIdx = closestPoints[i * k];  
      std::memcpy(gtVectors + i * dim, baseData + gtIdx * dim, dim * sizeof(float));
    }

    saveGTVectorsAsFile(gtFile, step, batchVectors, gtVectors, insertCount, dim);

    delete[] gtVectors;
    delete[] closestPoints;
    delete[] distClosestPoints;

    std::cout << "Step " << step << " completed. Inserted " << insertCount 
              << " vectors. Total: " << currentPoints << std::endl;
  }

  delete[] baseData;
  delete[] queryData;
}

int main(int argc, char** argv) {
  std::string dataType, distFn, baseFile, queryFile, gtFile, recallFile;
  uint64_t K, batchSize;

  try {
    po::options_description desc{"Arguments"};

    desc.add_options()
      ("data_type", po::value<std::string>(&dataType)->required(), "Data type <int8/uint8/float>")
      ("dist_fn", po::value<std::string>(&distFn)->required(), "Distance function <l2/mips/cosine>")
      ("base_file", po::value<std::string>(&baseFile)->required(), "Base vectors binary file")
      ("query_file", po::value<std::string>(&queryFile)->required(), "Query vectors binary file")
      ("gt_file", po::value<std::string>(&gtFile)->required(), "Output GT file")
      ("recall_file", po::value<std::string>(&recallFile)->required(), "Output recall file")
      ("K", po::value<uint64_t>(&K)->required(), "Number of neighbors to compute")
      ("batch_size", po::value<uint64_t>(&batchSize)->default_value(100), "Batch size for incremental computation");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << std::endl;
    return -1;
  }

  COMPUTE_GT::Metric metric;
  if (distFn == "l2") {
    metric = COMPUTE_GT::Metric::L2;
  } else if (distFn == "mips") {
    metric = COMPUTE_GT::Metric::INNER_PRODUCT;
  } else if (distFn == "cosine") {
    metric = COMPUTE_GT::Metric::COSINE;
  } else {
    std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
    return -1;
  }

  COMPUTE_GT::calcStepwiseGT(baseFile, queryFile, gtFile, K, distFn, batchSize);

  std::cout << "Stepwise GT computation completed." << std::endl;

  COMPUTE_GT::calcStepwiseRecall(queryFile, gtFile, recallFile);
  std::cout << "Stepwise Recall computation completed. Results saved to " << recallFile << std::endl;

  return 0;
}

