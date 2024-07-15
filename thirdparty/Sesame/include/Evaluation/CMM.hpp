//
// Created by tuidan on 2021/10/11.
//

#ifndef SESAME_INCLUDE_EVALUATION_CMM_HPP_
#define SESAME_INCLUDE_EVALUATION_CMM_HPP_

#include "Algorithm/DataStructure/Point.hpp"
#include "Algorithm/Param.hpp"

#include <atomic>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace SESAME {

enum ClusterType { Cluster, GTCluster };
enum ClusterMapType { vote, similarity };

class CMMPoint;
class CMMCluster;
class CMM;

typedef std::shared_ptr<CMMPoint> CMMPointPtr;
typedef std::shared_ptr<CMMCluster> CMMClusterPtr;

class CMMPoint {
public:
  int id;
  long startTime;
  int dim;
  std::vector<double> vec;
  double weight;
  int truth;
  double conCL = 0;
  double con = 0;

  CMMPoint(PointPtr p)
      : id(p->index), startTime(p->timestamp), dim(p->dim), vec(p->feature),
        weight(p->weight), truth(p->clu_id) {}

  CMMPoint(int id, long startTime, long time, std::vector<double> &vec,
           double a, double lambda, int truth);

  double getDisTo(CMMPointPtr &p);

  double knnDis(int k, CMMCluster &c);
};

class CMMCluster {
public:
  std::vector<CMMPointPtr> points;
  int groundTruth;
  std::vector<int> rho;
  double knhDis{};
  ClusterType type;
  CMMCluster();

  void add(CMMPointPtr &p);

  // void getDistribution(std::unordered_map<std::string, int> &map);

  void getConn();
};

class CMMDriver {
public:
  double a;
  double lambda;
  int dim;
  int groundTruth;

  std::vector<CMMPointPtr> points;
  std::unordered_map<int, CMMClusterPtr> CL;
  std::vector<CMMClusterPtr> CLlist;
  std::unordered_map<int, CMMClusterPtr> C;
  std::unordered_map<std::string, std::string> CToCL;
  std::vector<CMMClusterPtr> Clist;
  std::vector<CMMPointPtr> faultSet;
  std::unordered_set<CMMClusterPtr> faultClu;

  CMMDriver(int dim, double a, double lambda);
  void load(const std::vector<PointPtr> &inputs,
            const std::vector<PointPtr> &predicts, int dim, double weight);
  void voteMap();
  double computeWeight(double deltaTime);
  // int getDelta(CMMClusterPtr ci, CMMClusterPtr cljo);
  void getFaultSet();
  double compCMM();
  void compCon();
};

class CMM {
private:
  param_t param;
  /**
   * number of non-noise points that will create an error due to the underlying
   * clustering model (e.g. point being covered by two clusters representing
   * different classes)
   */
  int noiseErrorByModel = 0;
  /**
   * number of noise points that will create an error due to the underlying
   * clustering model (e.g. noise point being covered by a cluster)
   */
  int pointErrorByModel = 0;
  /**
   * experimental (default: disabled)
   * use exponential connectivity function to model different behavior:
   * closer points will have a stronger connection compared to the linear
   * function. Use ConnRefXValue and ConnX to better parameterize lambda, which
   * controls the decay of the connectivity
   */
  bool useExpConnectivity = false;
  double lambdaConnRefXValue = 0.01;
  double lambdaConnX = 4;
  double lambdaConn;
  /**
   * the threshold which defines when ground truth clusters will be merged.
   * set to 1 to disable merging
   */
  double tauConnection = 0.5;

  /**
   * defines how many nearest neighbors will be used
   */
  int knnNeighbourhood = 2;

  /** average knn distance of all points in the cluster*/
  double knnMeanAvg = 0;

  /** average deviation of knn distance of all points*/
  double knnDevAvg = 0;

  struct Cluster {
    std::unordered_set<int> points;
    std::vector<int> vpoints;
    std::atomic<double> knnMeanAvg = 0.0, knnDevAvg = 0.0;
    void Insert(int i) {
      points.insert(i);
      vpoints.push_back(i);
    }
    void CalcKnn(int k, const std::vector<PointPtr> &inputs);
  };
  std::unordered_map<int, Cluster> clusters;
  std::map<int, int> matchMap;
  double cmm;

public:
  CMM(param_t param) : param(param) {}
  double Evaluate(const std::vector<PointPtr> &inputs,
                  const std::vector<PointPtr> &predicts);
  void AnalyseGT(const std::vector<PointPtr> &inputs,
                 const std::vector<PointPtr> &predicts, bool enableClassMerge);
  double CalcConn(int, const std::vector<PointPtr> &);
  double CalcConn(int, int, const std::vector<PointPtr> &);
  void CalcMatch(const std::vector<PointPtr> &inputs,
                 const std::vector<PointPtr> &predicts);
  void CalcError(const std::vector<PointPtr> &inputs,
                 const std::vector<PointPtr> &predicts);
  double MisplacedError(int, int, int, const std::vector<PointPtr> &);
  double NoiseError(int, int, int, const std::vector<PointPtr> &);
  double MissedError(int, int, int, const std::vector<PointPtr> &);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_EVALUATION_CMM_HPP_
