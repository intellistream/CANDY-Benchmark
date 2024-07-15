//
// Created by tuidan on 2021/9/9.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CACHE_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CACHE_HPP_
#include <Algorithm/DataStructure/DPNode.hpp>
#include <Algorithm/DataStructure/DPTree.hpp>
#include <Algorithm/DataStructure/OutlierReservoir.hpp>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

namespace SESAME {

// define the share point of the class object
class Cache;
typedef std::shared_ptr<Cache> CachePtr;

class Cache {
private:
  int num;
  int size;
  double a;
  double lamd;
  double r;
  std::vector<DPNodePtr> buffer;
  std::vector<DPNodePtr> clus;
  int pnum;

public:
  Cache();
  ~Cache();
  int GetNum();
  void SetNum(int num);
  int GetSize();
  void SetSize(int size);
  double GetA();
  void SetA(double a);
  double GetLamd();
  void SetLamd(double lamd);
  double GetR();
  void SetR(double r);
  std::vector<DPNodePtr> &GetBuffer();
  void SetBuffer(std::vector<DPNodePtr> &buffer);
  std::vector<DPNodePtr> &GetClus();
  void SetClus(std::vector<DPNodePtr> &clus);
  int GetPnum();
  void SetPnum(int pnum);
  Cache(int num, double a, double lamd, double r);
  DPNodePtr add(PointPtr &p, double startTime);
  bool isFull();
  void compDeltaRho(double time);
  void getDPTree(double minRho, double minDelta, DPTreePtr &dpTree,
                 OutPtr &outs, std::unordered_set<ClusterPtr> &clusters);
  // void outputBuffer(String outpath, double minRho)
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CACHE_HPP_
