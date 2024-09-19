//
// Created by rubato on 19/9/24.
//

#pragma once

#include<vector>
#include<faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include<faiss/impl/HNSW.h>
#include<faiss/utils/utils.h>


namespace faiss{

    struct IndexMNRU;

    struct ReconstructFromNeighbors {
        typedef HNSW::storage_idx_t storage_idx_t;

        const IndexMNRU& index;
        size_t M;   // number of neighbors
        size_t k;   // number of codebook entries
        size_t nsq; // number of subvectors
        size_t code_size;
        int k_reorder; // nb to reorder. -1 = all

        std::vector<float> codebook; // size nsq * k * (M + 1)

        std::vector<uint8_t> codes; // size ntotal * code_size
        size_t ntotal;
        size_t d, dsub; // derived values

        explicit ReconstructFromNeighbors(
                const IndexMNRU& index,
                size_t k = 256,
                size_t nsq = 1);

        /// codes must be added in the correct order and the IndexHNSW
        /// must be populated and sorted
        void add_codes(size_t n, const float* x);

        size_t compute_distances(
                size_t n,
                const idx_t* shortlist,
                const float* query,
                float* distances) const;

        /// called by add_codes
        void estimate_code(const float* x, storage_idx_t i, uint8_t* code) const;

        /// called by compute_distances
        void reconstruct(storage_idx_t i, float* x, float* tmp) const;

        void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float* x) const;

        /// get the M+1 -by-d table for neighbor coordinates for vector i
        void get_neighbor_table(storage_idx_t i, float* out) const;
    };

    struct IndexMNRU: Index{
        typedef HNSW::storage_idx_t storage_idx_t;

        HNSW main_index;
        HNSW backup_index;
        bool own_fields = false;
        Index* storage=nullptr;
        int64_t backup_tau = 10000;

        ReconstructFromNeighbors* reconstruct_from_neighbors = nullptr;
        explicit IndexMNRU(int d = 0, int M = 32, MetricType metric = METRIC_L2);
        explicit IndexMNRU(Index* storage, int M = 32);

        ~IndexMNRU() override;

        void add(idx_t n, const float* x) override;

        /// Trains the storage if needed
        void train(idx_t n, const float* x) override;

        /// entry point for search
        void search(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;

        void reconstruct(idx_t key, float* recons) const override;

        void reset() override;


    };

    struct IndexMNRUFlat : IndexMNRU{
        IndexMNRUFlat();
        IndexMNRUFlat(int d, int M, MetricType metric = METRIC_L2);
    };
}

