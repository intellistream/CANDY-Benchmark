/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSWbd.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHNSWbd;

struct ReconstructFromNeighborsbd {
    typedef HNSWbd::storage_idx_t storage_idx_t;

    const IndexHNSWbd& index;
    size_t M;   // number of neighbors
    size_t k;   // number of codebook entries
    size_t nsq; // number of subvectors
    size_t code_size;
    int k_reorder; // nb to reorder. -1 = all

    std::vector<float> codebook; // size nsq * k * (M + 1)

    std::vector<uint8_t> codes; // size ntotal * code_size
    size_t ntotal;
    size_t d, dsub; // derived values

    explicit ReconstructFromNeighborsbd(
            const IndexHNSWbd& index,
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

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSWbd : Index {
    typedef HNSWbd::storage_idx_t storage_idx_t;

    // the link strcuture
    HNSWbd hnsw;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    ReconstructFromNeighborsbd* reconstruct_from_neighbors = nullptr;

    explicit IndexHNSWbd(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHNSWbd(Index* storage, int M = 32);

    ~IndexHNSWbd() override;

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

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

    /// alternative graph building
    void init_level_0_from_entry_points(
            int npt,
            const storage_idx_t* points,
            const storage_idx_t* nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();

    void permute_entries(const idx_t* perm);
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWbdFlat : IndexHNSWbd {
    IndexHNSWbdFlat() {
        is_trained = true;
    }

    IndexHNSWbdFlat(int d, int M, MetricType metric)
            : IndexHNSWbd(
                      (metric == METRIC_L2) ? new IndexFlatL2(d)
                                            : new IndexFlat(d, metric),
                      M) {
        own_fields = true;
        is_trained = true;
    }
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWbdPQ : IndexHNSWbd {
    IndexHNSWbdPQ();
    IndexHNSWbdPQ(int d, int pq_m, int M, int pq_nbits = 8);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWbdSQ : IndexHNSWbd {
    IndexHNSWbdSQ();
    IndexHNSWbdSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

/** 2-level code structure with fast random access
 */
struct IndexHNSWbd2Level : IndexHNSWbd {
    IndexHNSWbd2Level();
    IndexHNSWbd2Level(Index* quantizer, size_t nlist, int m_pq, int M);

    void flip_to_ivf();

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
