//
// Created by tony on 10/05/23.
//

#include <DataLoader/FVECSDataLoader.h>

bool CANDY::FVECSDataLoader::generateData(std::string fname) {
  auto dataTensor = CANDY::FVECSDataLoader::tensorFromFVECS(fname);
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
bool CANDY::FVECSDataLoader::generateQuery(std::string fname) {
  if (!useSeparateQuery) {
    B = INTELLI::IntelliTensorOP::rowSampling(A, querySize);
    B = (1 - queryNoiseFraction) * B + queryNoiseFraction * torch::rand({querySize, vecDim});
    return true;
  } else {
    auto queryTensor = CANDY::FVECSDataLoader::tensorFromFVECS(fname);
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
torch::Tensor CANDY::FVECSDataLoader::tensorFromFVECS(std::string fname) {
  torch::Tensor ru;
  unsigned num, dim;
  std::ifstream in(fname, std::ios::binary);    //以二进制的方式打开文件
  if (!in.is_open()) {
    INTELLI_ERROR(
        "Double check your data path: " + fname);
    return ru;
  }
  in.read((char *) &dim, 4);    //读取向量维度
  in.seekg(0, std::ios::end);    //光标定位到文件末尾
  std::ios::pos_type ss = in.tellg();    //获取文件大小（多少字节）
  size_t fsize = (size_t) ss;
  num = (unsigned) (fsize / (dim + 1) / 4);    //数据的个数
  std::vector<float> dataVec((size_t) num * (size_t) dim);
  float *data = reinterpret_cast<float *>(dataVec.data());
  in.seekg(0, std::ios::beg);    //光标定位到起始处
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);    //光标向右移动4个字节
    in.read((char *) (data + i * dim), dim * 4);    //读取数据到一维数据data中
  }
  in.close();

  torch::TensorOptions options(torch::kFloat32);
  ru = torch::from_blob(data, {(int64_t) num, (int64_t) dim}, options).clone();
  return ru;
}
bool CANDY::FVECSDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 128, true);
  vecVolume = cfg->tryI64("vecVolume", 10000, true);
  querySize = cfg->tryI64("querySize", 10, true);
  seed = cfg->tryI64("seed", 7758258, true);
  queryNoiseFraction = cfg->tryDouble("queryNoiseFraction", 0, true);
  normalizeTensor = cfg->tryDouble("normalizeTensor", 1, true);
  auto queryPath = cfg->tryString("queryPath", "datasets/fvecs/sift10K/siftsmall_query.fvecs", true);
  auto dataPath = cfg->tryString("dataPath", "datasets/fvecs/sift10K/siftsmall_base.fvecs", true);
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
  if (generateQuery(queryPath) == false) {
    return false;
  }
  INTELLI_INFO(
      "Generating [" + to_string(A.size(0)) + "x" + to_string(A.size(1)) + "]" + ", query size "
          + to_string(B.size(0)));
  if (useSeparateQuery) {
    INTELLI_INFO("Query is loaded from separate file");
  } else {
    INTELLI_INFO("Query is sampled from data file");
  }
  return true;
}

torch::Tensor CANDY::FVECSDataLoader::getData() {
  return A;
}

torch::Tensor CANDY::FVECSDataLoader::getQuery() {
  return B;
}
