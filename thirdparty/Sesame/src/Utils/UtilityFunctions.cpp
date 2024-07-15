// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

#include "Utils/UtilityFunctions.hpp"

#include <cfloat>
#include <cmath>
#include <random>

#include <omp.h>

using namespace std;

namespace SESAME {
static unsigned long mt[N]; /* the array for the state vector  */
static int mti;             /* mti==N+1 means mt[N] is not initialized */

long UtilityFunctions::genrand_int31() { return long(genrand_int32() >> 1); }
unsigned long UtilityFunctions::genrand_int32() {
  unsigned long y;
  static unsigned long mag01[2] = {0x0UL, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mti >= N) { /* generate N words at one time */
    int kk;

    if (mti == N + 1)       /* if init_genrand() has not been called, */
      init_genrand(5489UL); /* a default initial seed is used */

    for (kk = 0; kk < N - M; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < N - 1; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    mti = 0;
  }

  y = mt[mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}
UtilityFunctions::UtilityFunctions() {
  mti = N + 1; /* mti==N+1 means mt[N] is not initialized */
}

/**
 * initializes mt[N] with a seed
 * @param s
 */
void UtilityFunctions::init_genrand(unsigned long s) {
  /* initializes mt[N] with a seed */
  mt[0] = s & 0xffffffffUL;
  for (mti = 1; mti < N; mti++) {
    mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
}

/**
 *  generates a random number on (0,1)-real-interval
 * @return
 */
double UtilityFunctions::genrand_real3() {
  return (((double)genrand_int32()) + 0.5) * (1.0 / 4294967296.0);
  /* divided by 2^32 */
}
shared_ptr<barrier<>> UtilityFunctions::createBarrier(int count) {
  return make_shared<barrier<>>(count);
}

void UtilityFunctions::groupByCenters(const vector<PointPtr> &input,
                                      const vector<PointPtr> &centers,
                                      vector<PointPtr> &output, int dim) {
  auto n = input.size();
  for (int i = 0; i < n; i++)
    output.push_back(input[i]->copy());
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    auto min = DBL_MAX;
    for (int j = 0; j < centers.size(); j++) {
      double dis = output[i]->L2Dist(centers[j]);
      if (min > dis) {
        output[i]->setClusteringCenter(centers[j]->getClusteringCenter());
        min = dis;
      }
    }
    //    if(output[i]->getClusteringCenter() == -1){
    //      output[i]->setOutlier(true);
    //    } else {
    //      output[i]->setOutlier(false);
    //    }
  }
}

#if defined(_WIN32)
#include <Shlwapi.h>
#include <io.h>
#include <windows.h>

#define access _access_s
#endif

#ifdef __APPLE__
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <libgen.h>
#include <limits.h>
#include <unistd.h>

#if defined(__sun)
#define PROC_SELF_EXE "/proc/self/path/a.out"
#else
#define PROC_SELF_EXE "/proc/self/exe"
#endif

#endif

#if defined(_WIN32)

string getExecutablePath() {
  char rawPathName[MAX_PATH];
  GetModuleFileNameA(NULL, rawPathName, MAX_PATH);
  return string(rawPathName);
}

string getExecutableDir() {
  string executablePath = getExecutablePath();
  char *exePath = new char[executablePath.length()];
  strcpy(exePath, executablePath.c_str());
  PathRemoveFileSpecA(exePath);
  string directory = string(exePath);
  delete[] exePath;
  return directory;
}

string mergePaths(string pathA, string pathB) {
  char combined[MAX_PATH];
  PathCombineA(combined, pathA.c_str(), pathB.c_str());
  string mergedPath(combined);
  return mergedPath;
}

#endif

#ifdef __linux__

string getExecutablePath() {
  char rawPathName[PATH_MAX];
  realpath(PROC_SELF_EXE, rawPathName);
  return string(rawPathName);
}

string getExecutableDir() {
  string executablePath = getExecutablePath();
  char *executablePathStr = new char[executablePath.length() + 1];
  strcpy(executablePathStr, executablePath.c_str());
  char *executableDir = dirname(executablePathStr);
  delete[] executablePathStr;
  return string(executableDir);
}

string mergePaths(string pathA, string pathB) { return pathA + "/" + pathB; }

#endif

#ifdef __APPLE__
string getExecutablePath() {
  char rawPathName[PATH_MAX];
  char realPathName[PATH_MAX];
  uint32_t rawPathSize = (uint32_t)sizeof(rawPathName);

  if (!_NSGetExecutablePath(rawPathName, &rawPathSize)) {
    realpath(rawPathName, realPathName);
  }
  return string(realPathName);
}

string getExecutableDir() {
  string executablePath = getExecutablePath();
  char *executablePathStr = new char[executablePath.length() + 1];
  strcpy(executablePathStr, executablePath.c_str());
  char *executableDir = dirname(executablePathStr);
  delete[] executablePathStr;
  return string(executableDir);
}

string mergePaths(string pathA, string pathB) { return pathA + "/" + pathB; }
#endif

bool checkIfFileExists(const string &filePath) {
  return access(filePath.c_str(), 0) == 0;
}

} // namespace SESAME