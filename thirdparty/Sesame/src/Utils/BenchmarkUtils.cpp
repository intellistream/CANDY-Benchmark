#include "Utils/BenchmarkUtils.hpp"
#include "Algorithm/AlgorithmFactory.hpp"
#include "Algorithm/DataStructure/GenericFactory.hpp"
#include "Engine/SimpleEngine.hpp"
#include "Sinks/DataSink.hpp"
#include "Sources/DataSource.hpp"

using namespace std;

namespace SESAME {

std::pair<AccuracyRes, PerfRes> RunBenchmark(param_t &param) {
  // Create Spout.
  DataSourcePtr sourcePtr = GenericFactory::New<DataSource>(param);

  // Directly load data from file. TODO: configure it to load from external
  // sensors, e.g., HTTP.
  sourcePtr->load();

  // Create Sink.
  DataSinkPtr sinkPtr = GenericFactory::New<DataSink>(param);

  // Create Algorithm.
  AlgorithmPtr algoPtr = AlgorithmFactory::create(param);

  param.Print();

  SimpleEngine engine(sourcePtr, sinkPtr,
                      algoPtr); // TODO: create multithread engine in future.
  engine.run();
  while (!sinkPtr->isFinished())
    usleep(100);
  // wait for sink to stop.

  PerfRes perf = algoPtr->GetPerf();

  std::vector<PointPtr> inputs = sourcePtr->getInputs();
  std::vector<PointPtr> results = sinkPtr->getResults();
  std::vector<PointPtr> predicts;
  // the output clusterID start from 0
  if (param.run_eval) {
    UtilityFunctions::groupByCenters(inputs, results, predicts, param.dim);
  }

  param.num_res = results.size();
  std::cerr << "results.size=" << results.size() << std::endl;

  // Store results.
  if (param.store) {
    algoPtr->Store(param.output_file, param.dim, sinkPtr->getResults());
  }

  AccuracyRes acc;
  acc.num_res = param.num_res;
  acc.Evaluate(param, inputs, predicts);

  engine.stop();
  perf.Print();
  acc.Print();

  return make_pair(acc, perf);
}

} // namespace SESAME
