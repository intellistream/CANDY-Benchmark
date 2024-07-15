//
// Created by tuidan on 2021/9/9.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_OUTLIERRESERVOIR_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_OUTLIERRESERVOIR_HPP_
#include <Algorithm/DataStructure/DPNode.hpp>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

namespace SESAME {

class OutlierReservoir;
typedef std::shared_ptr<OutlierReservoir> OutPtr;
class OutlierReservoir {
private:
  double r;
  double timeGap;
  long lastDelTime;

  double a;
  double lamd;

  std::unordered_set<DPNodePtr> outliers;

public:
  [[nodiscard]] double GetR();
  void SetR(double r);
  [[nodiscard]] double GetTimeGap();
  void SetTimeGap(double time_gap);
  [[nodiscard]] long GetLastDelTime();
  void SetLastDelTime(long last_del_time);
  [[nodiscard]] double GetA();
  void SetA(double a);
  [[nodiscard]] double GetLamd();
  void SetLamd(double lamd);
  [[nodiscard]] std::unordered_set<SESAME::DPNodePtr> &getOutliers();
  void setOutliers(std::unordered_set<SESAME::DPNodePtr> &outliers);
  OutlierReservoir();
  ~OutlierReservoir();
  OutlierReservoir(double r, double a, double lamd);

  void setTimeGap(double timeGap);

  void insert(SESAME::DPNodePtr &c);

  SESAME::DPNodePtr insert(SESAME::PointPtr &p, double time);

  void remove(SESAME::DPNodePtr &nn);
};

} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_OUTLIERRESERVOIR_HPP_
