// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intelliStream) :

//
// Created by Shuhao Zhang on 26/07/2021.
//

#include "Algorithm/AlgorithmFactory.hpp"

#include "Algorithm/Benne.hpp"
#include "Algorithm/Birch.hpp"
#include "Algorithm/CluStream.hpp"
#include "Algorithm/DBStream.hpp"
#include "Algorithm/DStream.hpp"
#include "Algorithm/DataStructure/CoresetTree.hpp"
#include "Algorithm/DataStructure/MeyersonSketch.hpp"
#include "Algorithm/DenStream.hpp"
#include "Algorithm/DesignAspect/Generic.hpp"
#include "Algorithm/DesignAspect/V10.hpp"
#include "Algorithm/DesignAspect/V16.hpp"
#include "Algorithm/DesignAspect/V9.hpp"
#include "Algorithm/EDMStream.hpp"
#include "Algorithm/OutlierDetection/OutlierDetection.hpp"
#include "Algorithm/SlidingWindowClustering.hpp"
#include "Algorithm/StreamKM.hpp"

namespace SESAME {
AlgorithmPtr AlgorithmFactory::create(param_t &cmd_params) {
  switch (cmd_params.algo) {
  case (BenneType): {
    return std::make_shared<Benne>(cmd_params);
  }
  case (StreamKMeansType): {
    return std::make_shared<StreamKM>(cmd_params);
  }
  case (CluStreamType): {
    return std::make_shared<CluStream>(cmd_params);
  }
  case (DenStreamType): {
    return std::make_shared<DenStream>(cmd_params);
  }
  case (EDMStreamType): {
    return std::make_shared<EDMStream>(cmd_params);
  }
  case (DBStreamType): {
    return std::make_shared<DBStream>(cmd_params);
  }
  case (DStreamType): {
    return std::make_shared<DStream>(cmd_params);
  }
  case (SLKMeansType): {
    return std::make_shared<SlidingWindowClustering>(cmd_params);
  }
  case (G9Stream): {
    return std::make_shared<V9>(cmd_params);
  }
  case (G1Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<false, false>, KMeans>>(cmd_params);
  }
  case (G2Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<false, false>, DBSCAN>>(cmd_params);
  }
  case (G3Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (G4Stream): {
    return std::make_shared<
        StreamClustering<Sliding, ClusteringFeaturesList,
                         DistanceDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (G5Stream): {
    return std::make_shared<
        StreamClustering<Damped, ClusteringFeaturesList,
                         DistanceDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (BirchType): {
    return std::make_shared<StreamClustering<Landmark, ClusteringFeaturesTree,
                                             NoDetection, NoRefinement>>(
        cmd_params);
  }
  case (G6Stream): {
    return std::make_shared<StreamClustering<Landmark, ClusteringFeaturesList,
                                             NoDetection, NoRefinement>>(
        cmd_params);
  }
  case (G7Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesTree,
                         DensityDetection<true, false>, NoRefinement>>(
        cmd_params);
  }
  case (G8Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesTree,
                         DistanceDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (G10Stream): {
    return std::make_shared<V10>(cmd_params);
  }
  case (G11Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<false, true>, NoRefinement>>(
        cmd_params);
  }
  case (G12Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<true, false>, NoRefinement>>(
        cmd_params);
  }
  case (G13Stream): {
    return std::make_shared<
        StreamClustering<Landmark, ClusteringFeaturesList,
                         DistanceDetection<true, true>, NoRefinement>>(
        cmd_params);
  }
  case (G14Stream): {
    return std::make_shared<
        StreamClustering<Landmark, MeyersonSketch,
                         OutlierDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (G15Stream): {
    return std::make_shared<StreamClustering<
        Landmark, CoresetTree, OutlierDetection<false, false>, NoRefinement>>(
        cmd_params);
  }
  case (G16Stream): {
    return std::make_shared<V16>(cmd_params);
  }
  default:
    throw std::invalid_argument("Unsupported algorithm");
  }
}

} // namespace SESAME
