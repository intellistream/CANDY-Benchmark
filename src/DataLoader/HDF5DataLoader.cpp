//
// Created by tony on 10/05/23.
//

#include <DataLoader/HDF5DataLoader.h>
#include <hdf5.h>
bool CANDY::HDF5DataLoader::generateData(std::string fname) {
  std::string attr = "dataset";
  auto dataTensor = CANDY::HDF5DataLoader::tensorFromHDF5(fname, attr);
  if (dataTensor.size(0) == 0) {
    return false;
  }
  if (dataTensor.size(1) != vecDim) {
    INTELLI_ERROR(
        "conflict dimension in" + fname);
    return false;
  }
  A = INTELLI::IntelliTensorOP::rowSampling(dataTensor, vecVolume);
  if (normalizeTensor) {
    A = INTELLI::IntelliTensorOP::l2Normalize(A);
  }
  return true;
}
bool CANDY::HDF5DataLoader::generateQuery(std::string fname) {
  if (!useSeparateQuery) {
    B = INTELLI::IntelliTensorOP::rowSampling(A, querySize);
    B = (1 - queryNoiseFraction) * B + queryNoiseFraction * torch::rand({querySize, vecDim});
    return true;
  } else {
    std::string attr = "query";
    auto queryTensor = CANDY::HDF5DataLoader::tensorFromHDF5(fname, attr);
    if (queryTensor.size(0) == 0) {
      return false;
    }
    if (queryTensor.size(1) != vecDim) {
      INTELLI_ERROR(
          "conflict dimension in" + fname);
      return false;
    }
    if (normalizeTensor) {
      queryTensor = INTELLI::IntelliTensorOP::l2Normalize(queryTensor);
    }
    B = INTELLI::IntelliTensorOP::rowSampling(queryTensor, querySize);
    return true;
  }
  return false;
}
torch::Tensor CANDY::HDF5DataLoader::tensorFromHDF5(std::string fname, std::string attr) {
  torch::Tensor ru;
  herr_t status;
  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file_id < 0) {
    INTELLI_ERROR(
        "invalid hdf5 file " + fname);

    return ru;
  }
  hid_t dataset_id;
#if H5Dopen_vers == 2
  dataset_id = H5Dopen2(file_id, attr.c_str(), H5P_DEFAULT);
#else
  dataset_id = H5Dopen(file_id, attr.c_str());
#endif
  if (dataset_id < 0) {
    INTELLI_ERROR(
        "invalid hdf5 attribute" + attr);
    return ru;
  }
  hid_t space_id = H5Dget_space(dataset_id);

  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(space_id, dims_out, NULL);
  std::vector<float> dataVec(dims_out[0] * dims_out[1]);
  float *dataset = reinterpret_cast<float *>(dataVec.data());
  //dataset = flann::Matrix<T>(new T[dims_out[0]*dims_out[1]], dims_out[0], dims_out[1]);

  status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset);
  if (status < 0) {
    INTELLI_ERROR(
        "invalid reading " + attr + "at " + fname);
    return ru;
  }

  H5Sclose(space_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  torch::TensorOptions options(torch::kFloat32);
  ru = torch::from_blob(dataset, {(int64_t) dims_out[0], (int64_t) dims_out[1]}, options).clone();
  return ru;
}
bool CANDY::HDF5DataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 512, true);
  vecVolume = cfg->tryI64("vecVolume", 10000, true);
  querySize = cfg->tryI64("querySize", 10, true);
  seed = cfg->tryI64("seed", 7758258, true);
  queryNoiseFraction = cfg->tryDouble("queryNoiseFraction", 0, true);
  normalizeTensor = cfg->tryI64("normalizeTensor", 1, true);
  auto dataPath = cfg->tryString("dataPath", "datasets/hdf5/sun/sun.hdf5", true);
  useSeparateQuery = cfg->tryI64("useSeparateQuery", 1, true);
  if (queryNoiseFraction < 0) {
    queryNoiseFraction = 0;
  }
  if (queryNoiseFraction > 1) {
    queryNoiseFraction = 1;
  }
  if (querySize > vecVolume) {
    INTELLI_ERROR("invalid size of query");
    return false;
  }
  torch::manual_seed(seed);
  if (generateData(dataPath) == false) {
    return false;
  }
  if (generateQuery(dataPath) == false) {
    return false;
  }
  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + "x" + to_string(A.size(1)) + "]" + ", query size "
          + to_string(B.size(0)));
  if (useSeparateQuery) {
    INTELLI_INFO("Query is loaded from separate attribute");
  } else {
    INTELLI_INFO("Query is sampled from data attribute");
  }
  return true;
}

torch::Tensor CANDY::HDF5DataLoader::getData() {
  return A;
}

torch::Tensor CANDY::HDF5DataLoader::getQuery() {
  return B;
}
