//
// Created by 1124a on 2021/10/27.
//

#ifndef SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CHARACTERISTICVECTOR_HPP_
#define SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CHARACTERISTICVECTOR_HPP_
#include <Algorithm/WindowModel/DampedWindow.hpp>
#include <cmath>
#include <ctime>
namespace SESAME {
enum Status { NO_CLASS = -1, SPARSE, TRANSITIONAL, DENSE };
class CharacteristicVector {
public:
  /**
   * t_g: The last time when g is updated
   */
  int updateTime;

  /**
   * tm : last time when g is removed from grid_list as a sporadic grid (if
   * ever).
   */
  int removeTime;

  /**
   * D: the grid density at the last update
   */
  double gridDensity;

  /**
   * label: the cluster label of the grid
   */
  int label;

  /**
   * status: status = {SPORADIC, NORMAL}
   */
  bool isSporadic;

  /**
   * attribute: attribute = {SPARSE, TRANSITIONAL, DENSE}
   */
  int attribute;

  /**
   * time stamp at which the grid's density was last updated (including initial
   * and adjust clustering)
   */
  int densityUpdateTime;

  /**
   * Flag marking whether there was a change in the attribute field
   * the last time the grid density was updated.
   */
  bool attChange;
  bool isVisited = false;

  CharacteristicVector();
  CharacteristicVector(int updateTime, int removeTime, double Density,
                       int label, bool isSporadic, double dl, double dm);
  double getCurrGridDensity(int NowTime, double lambda);
  double getCurrGridDensity();
  bool isSparse(double dl);
  bool isDense(double dm);
  bool isTransitional(double dm, double dl);
  /**
   * Implements the density update function given in
   * eq 5 (Proposition 3.1) of Chen and Tu 2007.
   *
   * @param currTime the data stream's current internal time
   * @param decayFactor the value of lambda
   */
  void densityWithNew(int NowTime, double decayFactor);
  void densityWithNew(int NowTime);
  /**
   * Implements the update the density of all grids step given at line 2 of
   * both Fig 3 and Fig 4 of Chen and Tu 2007.
   *
   * @param currTime the data stream's current internal time
   * @param decayFactor the value of lambda
   * @param dl the threshold for sparse grids
   * @param dm the threshold for dense grids
   */
  void UpdateAllDensity(int NowTime, double decayFactor, double dl, double dm);
  void UpdateAllDensity(int NowTime, double dl, double dm);
  void ChangeAttribute(double dl, double dm);
};
} // namespace SESAME
#endif // SESAME_INCLUDE_ALGORITHM_DATASTRUCTURE_CHARACTERISTICVECTOR_HPP_
