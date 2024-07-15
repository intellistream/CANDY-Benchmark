// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by tuidan on 2021/7/22.
//

#include "Algorithm/Algorithm.hpp"

namespace SESAME {
char const *algo_names[64] = {[BirchType] = "Birch",
                              [StreamKMeansType] = "StreamKMeans",
                              [CluStreamType] = "CluStream",
                              [DenStreamType] = "DenStream",
                              [DBStreamType] = "DBStream",
                              [EDMStreamType] = "EDMStream",
                              [DStreamType] = "DStream",
                              [SLKMeansType] = "SLKMeans",
                              [BenneType] = "Benne",
                              [9] = "ERROR9",
                              [10] = "ERROR10",
                              [11] = "ERROR11",
                              [12] = "ERROR12",
                              [13] = "ERROR13",
                              [14] = "ERROR14",
                              [15] = "ERROR15",
                              [16] = "ERROR16",
                              [17] = "ERROR17",
                              [18] = "ERROR18",
                              [19] = "ERROR19",
                              [20] = "ERROR20",
                              [G1Stream] = "G1",
                              [G2Stream] = "G2",
                              [G3Stream] = "G3",
                              [G4Stream] = "G4",
                              [G5Stream] = "G5",
                              [G6Stream] = "G6",
                              [G7Stream] = "G7",
                              [G8Stream] = "G8",
                              [G9Stream] = "G9",
                              [G10Stream] = "G10",
                              [G11Stream] = "G11",
                              [G12Stream] = "G12",
                              [G13Stream] = "G13",
                              [G14Stream] = "G14",
                              [G15Stream] = "G15",
                              [G16Stream] = "G16"};

char const *benne_suffix[4] = {[balance] = "Bal",
                               [accuracy] = "Acc",
                               [efficiency] = "Eff",
                               [accuracy_no_migration] = "AccNoMig"};

void Algorithm::Store(std::string output_file, int dim,
                      std::vector<PointPtr> result) {
  int numberOfCenters = (int)result.size();
  FILE *out = fopen(output_file.c_str(), "w");
  for (int i = 0; i < numberOfCenters; i++) {
    int l;
    fprintf(out, "%f ", result[i]->getWeight());
    for (l = 0; l < dim; l++) {
      fprintf(out, "%f ",
              result[i]->getFeatureItem(l) / result[i]->getWeight());
    }
    fprintf(out, "\n");
  }
  fclose(out);
}

} // namespace SESAME
