#include <random>
#include <chrono>

#include <omp.h>
#include<CANDY/DEGIndex.h>

bool CANDY::DEGIndex::setConfig(INTELLI::ConfigMapPtr cfg){
    AbstractIndex::setConfig(cfg);
    vecDim = cfg->tryI64("vecDim", 768, true);
    max_vertex_count = cfg->tryI64("maxvertexcount", 10000, true);
    label =  0;
    /// other params later;


    return true;
}

bool CANDY::DEGIndex::loadInitialTensor(torch::Tensor &t) {
    auto n = t.size(0); // Number of vectors (rows)
    auto dims = t.size(1); // Dimensionality (columns)
    // cout << n << std::endl;
    // cout << dims << std::endl;
    auto contiguous_features = std::make_unique<float[]>(n * dims);

    auto tensor_data_ptr = t.contiguous().data_ptr<float>();
    std::memcpy(contiguous_features.get(), tensor_data_ptr, n * dims * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        auto feat = reinterpret_cast<const std::byte*>(&contiguous_features[i * dims]);
        auto feature_vector = std::vector<std::byte>{feat, feat + (dims * sizeof(float))};
        // Add the feature to the builder
        builder.addEntry(label, std::move(feature_vector));
        label++;
    }

    builder.build();

    return true;
}
bool CANDY::DEGIndex::insertTensor(torch::Tensor &t){

        auto n = t.size(0); // Number of vectors (rows)
        auto dims = t.size(1); // Dimensionality (columns)

        auto contiguous_features = std::make_unique<float[]>(n * dims);

        auto tensor_data_ptr = t.contiguous().data_ptr<float>();
        std::memcpy(contiguous_features.get(), tensor_data_ptr, n * dims * sizeof(float));

        for (size_t i = 0; i < n; i++) {
            auto feat = reinterpret_cast<const std::byte*>(&contiguous_features[i * dims]);
            auto feature_vector = std::vector<std::byte>{feat, feat + (dims * sizeof(float))};

            builder.addEntry(label, std::move(feature_vector));
            label++;
        }

        builder.build();

        return true;
}




std::vector<torch::Tensor> CANDY::DEGIndex::searchTensor(torch::Tensor &q, int64_t k) {
    // auto dims = q.size(1);
    // auto count = q.size(0);
    //cout<<"q"<<endl<<q<<endl;
    std::vector<uint32_t> entry_vertex_indices = graphIndex.getEntryVertexIndices();
    //fmt::print("external id {} \n", graph.getInternalIndex(entry_vertex_indices[0]));

    // try different eps values for the search radius
    // std::vector<float> eps_parameter = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, };    // audio
    //std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f  };       // SIFT1M k=100
    std::vector<float> eps_parameter = { 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f };             // GloVe
    // std::vector<float> eps_parameter = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, };          // Deep1M
    //auto n = q.size(0);
    float eps=0.14f;
    std::vector<torch::Tensor> ru;

    auto n = q.size(0); // Number of vectors (rows)
    auto dims = q.size(1); // Dimensionality (columns)

    auto contiguous_features = std::make_unique<float[]>(n * dims);

    auto tensor_data_ptr = q.contiguous().data_ptr<float>();
    std::memcpy(contiguous_features.get(), tensor_data_ptr, n * dims * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        std::vector<float> feature_buffer;
        auto feat = reinterpret_cast<const std::byte*>(&contiguous_features[i * dims]);
        auto feature_vector = std::vector<std::byte>{feat, feat + (dims * sizeof(float))};

        auto result_queue = graphIndex.search(entry_vertex_indices, feat, eps, k);
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getInternalIndex();
            auto external_id = graphIndex.getExternalLabel(internal_index);
            auto feat_byte = graphIndex.getFeatureVector(internal_index);
            auto float_ptr = reinterpret_cast<const float*>(feat_byte);
            feature_buffer.insert(feature_buffer.end(), float_ptr, float_ptr + dims);
            result_queue.pop();
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
        int num_features = feature_buffer.size() / dims;
        torch::Tensor t = torch::from_blob(feature_buffer.data(), {num_features, dims}, options).clone();;
        ru.push_back(t);
    }
    //cout<<ru<<endl;
    return ru;
}


//torch::Tensor result_tensor = torch::from_blob(result_queue.data(), {static_cast<long>(result_queue.size())}, torch::kInt32).clone();
// auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
// std::vector<int64_t> sizes = {n, dims}; // use a vector for sizes
// torch::Tensor t = torch::from_blob((void*)float_ptr, torch::IntArrayRef(sizes), options);
// ru.push_back(t);
// Create a tensor from the float pointer
//torch::Tensor tensor = torch::from_blob(&float_ptr, {count, dims});
//auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
//auto t = torch::from_blob(&feat_q, {count, dims});
//ru.push_back(t);
//for (size_t i = 0; i < n; i++) {
//auto label_ = q.slice(0, i, i + 1);
//const void* data_ptr = label_.data_ptr();
//onst size_t data_size = label_.numel() * label_.element_size();
// const float* feature = q.contiguous().data_ptr<float>();  // Get the pointer to the float data
// auto feat = reinterpret_cast<const std::byte*>(feature);
// // auto feature_vector = std::vector<std::byte>{feat, feat + dims * sizeof(float)};

//size_t num_elements = q.numel();  // Number of elements in the slice
//size_t data_size = num_elements * sizeof(float);  // Calculate the size in bytes
//const std::byte* byte_ptr = reinterpret_cast<const std::byte*>(feature);

// Create a vector of bytes to hold the feature data
//std::vector<std::byte> feature_vector(byte_ptr, byte_ptr + data_size);

//byte* feature = (byte*)std::malloc(data_size);
//std::memcpy(feature, data_ptr, data_size);

// static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_vertex_indices,
//                          const deglib::FeatureRepository& query_repository, const std::vector<std::unordered_set<uint32_t>>& ground_truth,
//                          const float eps, const uint32_t k, const uint32_t test_size)
// {
//     size_t total = 0;
//     size_t correct = 0;
//     for (uint32_t i = 0; i < test_size; i++)
//     {
//         auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
//         auto result_queue = graph.search(entry_vertex_indices, query, eps, k);
//
//         if (result_queue.size() != k) {
//             //fmt::print(stderr, "ANNS with k={} got only {} results for query {}\n", k, result_queue.size(), i);
//             abort();
//         }
//
//         total += result_queue.size();
//         const auto gt = ground_truth[i];
//         while (result_queue.empty() == false)
//         {
//             const auto internal_index = result_queue.top().getInternalIndex();
//             const auto external_id = graph.getExternalLabel(internal_index);
//             if (gt.find(external_id) != gt.end()) correct++;
//             result_queue.pop();
//         }
//     }
//
//     return 1.0f * correct / total;
// }

// std::vector<torch::Tensor> CANDY::DEGIndex::searchTensor(torch::Tensor &q, int64_t k){
//     auto idx = searchIndex(q,k);
//     return getTensorByIndex(idx,k);
//
// }
//
// std::vector<torch::Tensor> CANDY::DEGIndex::getTensorByIndex(std::vector<torch::Tensor> &idx, int64_t k){
//     int64_t size = idx.size() / k;
//     std::vector<torch::Tensor> ru(size);
//     for (int64_t i = 0; i < size; i++) {
//         ru[i] = torch::zeros({k, vecDim});
//         for (int64_t j = 0; j < k; j++) {
//             int64_t tempIdx = idx[i * k + j];
//
//             if (tempIdx >= 0) {
//                 ru[i].slice(0, j, j + 1) = dbTensor.slice(0, tempIdx, tempIdx + 1);
//
//             };
//         }
//     }
//     return ru;
//
// }
//
// const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
//     label_to_index_.emplace(external_label, new_internal_index);
//
//     auto vertex_memory = vertex_by_index(new_internal_index);
//     std::memcpy(vertex_memory, feature_vector, feature_byte_size_);
//     // std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), this->edges_per_vertex_, new_internal_index); // temporary self loop
//     // std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), this->edges_per_vertex_, float(0)); // 0 weight
//     // // Print values before filling neighbor indices
//     // printf("Filling neighbor indices with new_internal_index: %u\n", new_internal_index);
//     // printf("Neighbor indices offset: %u\n", neighbor_indices_offset_);
//     // printf("Edges per vertex: %u\n", this->edges_per_vertex_);
//     // printf("Size of feature_vector: %zu\n", sizeof(feature_vector));
//     // printf("Expected feature_byte_size_: %hu\n", feature_byte_size_);
//
//
//     // Fill neighbor indices with self loop
//     std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), this->edges_per_vertex_, new_internal_index);
//     //
//     // // Print values before filling neighbor weights
//     // printf("Filling neighbor weights with value: %f\n", float(0));
//     // printf("Neighbor weights offset: %u\n", neighbor_weights_offset_);
//     // printf("Edges per vertex: %u\n", this->edges_per_vertex_);
//
//     // Fill neighbor weights with 0
//     std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), this->edges_per_vertex_, float(0));
//
//     // // Print values before copying the external label
//     // printf("Copying external label: %u\n", external_label);
//     // printf("External label offset: %u\n", external_label_offset_);
//     std::memcpy(vertex_memory + external_label_offset_, &external_label, sizeof(uint32_t));
//     const auto feature_vector_float = reinterpret_cast<const float*>(this->feature_by_index(neighbor_index));
//
//     //printf("Neighbor Index: %u, Feature Vector: ", neighbor_index);
//
//     return new_internal_index;
//
