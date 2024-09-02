//
// Created by shivangi on 24/8/24.
//

#include <random>
#include <chrono>

#include <omp.h>
#include <CANDY/BKTIndex.h>

bool CANDY::BKTIndex::setConfig(INTELLI::ConfigMapPtr cfg){
    AbstractIndex::setConfig(cfg);
    // vecDim = cfg->tryI64("vecDim", 768, true);
    // max_vertex_count = cfg->tryI64("maxvertexcount", 10000, true);
    // label =  0;
    /// other params later;
    //distCalcMethod = cfg->tryI64("distCalcMethod", "L2", true);
    m = cfg->tryI64("vecDim", 128, true);
    //k = cfg->tryI64("k", 3, true);
    //SPTAG::SizeType n = 1000;
    //n = cfg->tryI64("batchSize", 1000, true);


    return true;
}

bool CANDY::BKTIndex::loadInitialTensor(torch::Tensor &t) {
    int n=t.size(0);
    cout<<n<<endl;
    std::vector<float> vec(n * m);
    auto tensor_data_ptr = t.contiguous().data_ptr<float>();
    std::memcpy(vec.data(), tensor_data_ptr, n * m * sizeof(float));
    // std::vector<char> meta;
    // std::vector<std::uint64_t> metaoffset;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(i);
        meta.insert(meta.end(), a.begin(), a.end());
    }
    metaoffset.push_back((std::uint64_t)meta.size());

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray((std::uint8_t*)vec.data(), sizeof(float) * n * m, false),
        SPTAG::GetEnumValueType<float>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(meta.data()), meta.size() * sizeof(char), false),
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(metaoffset.data()), metaoffset.size() * sizeof(std::uint64_t), false),
            metaoffset.size() - 1));


    //SPTAG::ErrorCode ret= vecIndex->BuildIndex(vecset, metaset, false);
    SPTAG::ErrorCode ret= vecIndex->AddIndex(vecset, metaset, false);
    if (SPTAG::ErrorCode::Success != ret) {
        std::cerr << "Error AddIndex(" << (int)(ret) << ") for initial vector " << std::endl;
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // for (SPTAG::SizeType i = 0; i < vecset->Count(); i++) {
    //     SPTAG::ByteArray metaarr = metaset->GetMetadata(i);
    //     std::uint64_t offset[2] = { 0, metaarr.Length() };
    //     std::shared_ptr<SPTAG::MetadataSet> meta_set(new SPTAG::MemMetadataSet(metaarr, SPTAG::ByteArray((std::uint8_t*)offset, 2 * sizeof(std::uint64_t), false), 1));
    //     SPTAG::ErrorCode ret = vecIndex->AddIndex(vecset->GetVector(i), 1, vecset->Dimension(), meta_set, false);
    //     if (SPTAG::ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
    // }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "AddIndex time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(vecset->Count())) << "us" << std::endl;

    return true;

}

bool CANDY::BKTIndex::insertTensor(torch::Tensor &t){
    int n=t.size(0);
    cout<<n<<endl;
    std::vector<float> vec(n * m);
    auto tensor_data_ptr = t.contiguous().data_ptr<float>();
    //cout<<n<<" "<<m<<endl;
    std::memcpy(vec.data(), tensor_data_ptr, n * m * sizeof(float));
    // std::vector<char> meta;
    // std::vector<std::uint64_t> metaoffset;

    for (SPTAG::SizeType i = 0; i < n; i++) {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(metaoffset.size() - 1);
        // for (size_t j = 0; j < a.length(); j++)
        //     meta.push_back(a[j]);
        meta.insert(meta.end(), a.begin(), a.end());
    }
    metaoffset.push_back((std::uint64_t)meta.size());

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
       SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(vec.data()), sizeof(float) * n * m, false),
       SPTAG::GetEnumValueType<float>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(meta.data()), meta.size() * sizeof(char), false),
            SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(metaoffset.data()), metaoffset.size() * sizeof(std::uint64_t), false),
            metaoffset.size() - 1));
    //AddOneByOne<float>(algo, distCalcMethod, vecset, metaset, indexDirectory);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SPTAG::SizeType i = 0; i < vecset->Count(); i++) {
        SPTAG::ByteArray metaarr = metaset->GetMetadata(i);
        std::uint64_t offset[2] = { 0, metaarr.Length() };
        std::shared_ptr<SPTAG::MetadataSet> meta_set(new SPTAG::MemMetadataSet(metaarr, SPTAG::ByteArray((std::uint8_t*)offset, 2 * sizeof(std::uint64_t), false), 1));
        SPTAG::ErrorCode ret = vecIndex->AddIndex(vecset->GetVector(i), 1, vecset->Dimension(), meta_set, false);
        if (SPTAG::ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "AddIndex time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(vecset->Count())) << "us" << std::endl;

return true;
}




std::vector<torch::Tensor> CANDY::BKTIndex::searchTensor(torch::Tensor &q, int64_t k) {
    int n=q.size(0);

    auto N = q.size(0);
    std::vector<torch::Tensor> ru;
    auto vec = q.contiguous().data_ptr<float>();
    for (SPTAG::SizeType i = 0; i < N; i++)
    {
        //auto vec_query = reinterpret_cast<float*>(&vec[i * m]);
        SPTAG::QueryResult res(vec, k, false);
        vecIndex->SearchIndex(res);
        //std::unordered_set<std::string> resmeta;
        std::vector<float> feature_buffer;
        for (int j = 0; j < k; j++)
        {
            //resmeta.insert(std::string((char*)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()));
            //std::cout << res.GetResult(j)->Dist << "@(" << res.GetResult(j)->VID << "," << std::string((char*)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()) << ") ";
            auto vector_result = res.GetResult(j);
            auto vecID= vector_result->VID;
            cout<<"vecID: "<<vecID<<endl;
            auto vector = vecIndex->GetSample(vecID);

            if (vector != nullptr) {
                auto float_ptr = reinterpret_cast<const float*>(vector);
                feature_buffer.insert(feature_buffer.end(), float_ptr, float_ptr + m ); //m==> vecIndex->GetFeatureDim()
            }
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
        int num_features = feature_buffer.size() / m;
        torch::Tensor t = torch::from_blob(feature_buffer.data(), {num_features, m}, options).clone();
        ru.push_back(t);

    }
    // for (size_t i = 0; i < ru.size(); i++) {
    //     std::cout << "Tensor " << i << ": " << ru[i] << std::endl;
    // }

    return ru;

}

