//
// Created by rubato on 19/9/24.
//

#include<faiss/IndexMNRU.h>
#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>


extern "C"{
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}
namespace faiss{
    using MinimaxHeap = HNSW::MinimaxHeap;
    using storage_idx_t = HNSW::storage_idx_t;
    using NodeDistFarther = HNSW::NodeDistFarther;

    HNSWStats hnsw_stats;


    namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

        struct NegativeDistanceComputer : DistanceComputer {
            /// owned by this
            DistanceComputer* basedis;

            explicit NegativeDistanceComputer(DistanceComputer* basedis)
                    : basedis(basedis) {}

            void set_query(const float* x) override {
                basedis->set_query(x);
            }

            /// compute distance of vector i to current query
            float operator()(idx_t i) override {
                return -(*basedis)(i);
            }

            void distances_batch_4(
                    const idx_t idx0,
                    const idx_t idx1,
                    const idx_t idx2,
                    const idx_t idx3,
                    float& dis0,
                    float& dis1,
                    float& dis2,
                    float& dis3) override {
                basedis->distances_batch_4(
                        idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
                dis0 = -dis0;
                dis1 = -dis1;
                dis2 = -dis2;
                dis3 = -dis3;
            }

            /// compute distance between two stored vectors
            float symmetric_dis(idx_t i, idx_t j) override {
                return -basedis->symmetric_dis(i, j);
            }

            virtual ~NegativeDistanceComputer() {
                delete basedis;
            }
        };

        DistanceComputer* storage_distance_computer(const Index* storage) {
            if (is_similarity_metric(storage->metric_type)) {
                return new NegativeDistanceComputer(storage->get_distance_computer());
            } else {
                return storage->get_distance_computer();
            }
        }

        void hnsw_add_vertices(
                IndexMNRU& index_hnsw,
                size_t n0,
                size_t n,
                const float* x,
                bool verbose,
                bool preset_levels = false) {
            size_t d = index_hnsw.d;
            HNSW& hnsw = index_hnsw.main_index;
            size_t ntotal = n0 + n;
            double t0 = getmillisecs();
            if (verbose) {
                printf("hnsw_add_vertices: adding %zd elements on top of %zd "
                       "(preset_levels=%d)\n",
                       n,
                       n0,
                       int(preset_levels));
            }

            if (n == 0) {
                return;
            }

            int max_level = hnsw.prepare_level_tab(n, preset_levels);

            if (verbose) {
                printf("  max_level = %d\n", max_level);
            }

            std::vector<omp_lock_t> locks(ntotal);
            for (int i = 0; i < ntotal; i++)
                omp_init_lock(&locks[i]);

            // add vectors from highest to lowest level
            std::vector<int> hist;
            std::vector<int> order(n);

            { // make buckets with vectors of the same level

                // build histogram
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    while (pt_level >= hist.size())
                        hist.push_back(0);
                    hist[pt_level]++;
                }

                // accumulate
                std::vector<int> offsets(hist.size() + 1, 0);
                for (int i = 0; i < hist.size() - 1; i++) {
                    offsets[i + 1] = offsets[i] + hist[i];
                }

                // bucket sort
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    order[offsets[pt_level]++] = pt_id;
                }
            }

            idx_t check_period = InterruptCallback::get_period_hint(
                    max_level * index_hnsw.d * hnsw.efConstruction);

            { // perform add
                RandomGenerator rng2(789);

                int i1 = n;

                for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
                    int i0 = i1 - hist[pt_level];

                    if (verbose) {
                        printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
                    }

                    // random permutation to get rid of dataset order bias
                    for (int j = i0; j < i1; j++)
                        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

                    bool interrupt = false;

//#pragma omp parallel if (i1 > i0 + 100)
                    {
                        VisitedTable vt(ntotal);

                        std::unique_ptr<DistanceComputer> dis(
                                storage_distance_computer(index_hnsw.storage));
                        int prev_display =
                                verbose && omp_get_thread_num() == 0 ? 0 : -1;
                        size_t counter = 0;

                        // here we should do schedule(dynamic) but this segfaults for
                        // some versions of LLVM. The performance impact should not be
                        // too large when (i1 - i0) / num_threads >> 1
//#pragma omp for schedule(static)
                        for (int i = i0; i < i1; i++) {
                            storage_idx_t pt_id = order[i];
                            dis->set_query(x + (pt_id - n0) * d);

                            // cannot break
                            if (interrupt) {
                                continue;
                            }

                            hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                            if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                                prev_display = i - i0;
                                printf("  %d / %d\r", i - i0, i1 - i0);
                                fflush(stdout);
                            }
                            if (counter % check_period == 0) {
                                if (InterruptCallback::is_interrupted()) {
                                    interrupt = true;
                                }
                            }
                            counter++;
                        }
                    }
                    if (interrupt) {
                        FAISS_THROW_MSG("computation interrupted");
                    }
                    i1 = i0;
                }
                FAISS_ASSERT(i1 == 0);
            }
            if (verbose) {
                printf("Done in %.3f ms\n", getmillisecs() - t0);
            }

            for (int i = 0; i < ntotal; i++) {
                omp_destroy_lock(&locks[i]);
            }
        }

        void backup_add_vertices(
                IndexMNRU& index_hnsw,
                size_t n0,//always be ZERO!
                size_t n,
                const float* x,
                bool verbose,
                bool preset_levels = false) {
            size_t d = index_hnsw.d;
            HNSW& hnsw = index_hnsw.backup_index;
            n0 = 0;
            hnsw.reset();
            size_t ntotal = n0 + n;
            double t0 = getmillisecs();
            if (verbose) {
                printf("hnsw_add_vertices: adding %zd elements on top of %zd "
                       "(preset_levels=%d)\n",
                       n,
                       n0,
                       int(preset_levels));
            }

            if (n == 0) {
                return;
            }

            int max_level = hnsw.prepare_level_tab(n, preset_levels);

            if (verbose) {
                printf("  max_level = %d\n", max_level);
            }

            std::vector<omp_lock_t> locks(ntotal);
            for (int i = 0; i < ntotal; i++)
                omp_init_lock(&locks[i]);

            // add vectors from highest to lowest level
            std::vector<int> hist;
            std::vector<int> order(n);

            { // make buckets with vectors of the same level

                // build histogram
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    while (pt_level >= hist.size())
                        hist.push_back(0);
                    hist[pt_level]++;
                }

                // accumulate
                std::vector<int> offsets(hist.size() + 1, 0);
                for (int i = 0; i < hist.size() - 1; i++) {
                    offsets[i + 1] = offsets[i] + hist[i];
                }

                // bucket sort
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    order[offsets[pt_level]++] = pt_id;
                }
            }

            idx_t check_period = InterruptCallback::get_period_hint(
                    max_level * index_hnsw.d * hnsw.efConstruction);

            { // perform add
                RandomGenerator rng2(789);

                int i1 = n;

                for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
                    int i0 = i1 - hist[pt_level];

                    if (verbose) {
                        printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
                    }

                    // random permutation to get rid of dataset order bias
                    for (int j = i0; j < i1; j++)
                        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

                    bool interrupt = false;

//#pragma omp parallel if (i1 > i0 + 100)
                    {
                        VisitedTable vt(ntotal);

                        std::unique_ptr<DistanceComputer> dis(
                                storage_distance_computer(index_hnsw.storage));
                        int prev_display =
                                verbose && omp_get_thread_num() == 0 ? 0 : -1;
                        size_t counter = 0;

                        // here we should do schedule(dynamic) but this segfaults for
                        // some versions of LLVM. The performance impact should not be
                        // too large when (i1 - i0) / num_threads >> 1
//#pragma omp for schedule(static)
                        for (int i = i0; i < i1; i++) {
                            storage_idx_t pt_id = order[i];
                            dis->set_query(x + (pt_id - n0) * d);

                            // cannot break
                            if (interrupt) {
                                continue;
                            }

                            hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                            if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                                prev_display = i - i0;
                                printf("  %d / %d\r", i - i0, i1 - i0);
                                fflush(stdout);
                            }
                            if (counter % check_period == 0) {
                                if (InterruptCallback::is_interrupted()) {
                                    interrupt = true;
                                }
                            }
                            counter++;
                        }
                    }
                    if (interrupt) {
                        FAISS_THROW_MSG("computation interrupted");
                    }
                    i1 = i0;
                }
                FAISS_ASSERT(i1 == 0);
            }
            if (verbose) {
                printf("Done in %.3f ms\n", getmillisecs() - t0);
            }

            for (int i = 0; i < ntotal; i++) {
                omp_destroy_lock(&locks[i]);
            }
        }

    } // namespace


    IndexMNRU::IndexMNRU(int d, int M, faiss::MetricType metric) :Index(d,metric){}
    IndexMNRU::IndexMNRU(faiss::Index *storage, int M) :Index(storage->d, storage->metric_type), main_index(M),
                                                        backup_index(M), storage(storage){}
    IndexMNRU::~IndexMNRU(){
        if(own_fields){
            delete storage;
        }
    }


    void IndexMNRU::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in) const {
        FAISS_THROW_IF_NOT(k > 0);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
        const SearchParametersHNSW* params = nullptr;

        int efSearch = main_index.efSearch;
        if (params_in) {
            params = dynamic_cast<const SearchParametersHNSW*>(params_in);
            FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
            efSearch = params->efSearch;
        }
        size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

        idx_t check_period =
                InterruptCallback::get_period_hint(main_index.max_level * d * efSearch);

        for (idx_t i0 = 0; i0 < n; i0 += check_period) {
            idx_t i1 = std::min(i0 + check_period, n);

//#pragma omp parallel
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(storage));

//#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder) schedule(guided)
                for (idx_t i = i0; i < i1; i++) {
                    idx_t* idxi = labels + i * k;
                    float* simi = distances + i * k;
                    dis->set_query(x + i * d);

                    maxheap_heapify(k, simi, idxi);

                    // dual index search
                    HNSWStats stats = main_index.search(*dis, k, idxi, simi, vt, params);
                    backup_index.search(*dis, k, idxi, simi, vt, params);
                    n1 += stats.n1;
                    n2 += stats.n2;
                    n3 += stats.n3;
                    ndis += stats.ndis;
                    nreorder += stats.nreorder;
                    maxheap_reorder(k, simi, idxi);

                    if (reconstruct_from_neighbors &&
                        reconstruct_from_neighbors->k_reorder != 0) {
                        int k_reorder = reconstruct_from_neighbors->k_reorder;
                        if (k_reorder == -1 || k_reorder > k)
                            k_reorder = k;

                        nreorder += reconstruct_from_neighbors->compute_distances(
                                k_reorder, idxi, x + i * d, simi);

                        // sort top k_reorder
                        maxheap_heapify(
                                k_reorder, simi, idxi, simi, idxi, k_reorder);
                        maxheap_reorder(k_reorder, simi, idxi);
                    }
                }
            }
            InterruptCallback::check();
        }

        if (is_similarity_metric(metric_type)) {
            // we need to revert the negated distances
            for (size_t i = 0; i < k * n; i++) {
                distances[i] = -distances[i];
            }
        }

        hnsw_stats.combine({n1, n2, n3, ndis, nreorder});
    }

    void IndexMNRU::train(idx_t n, const float* x) {
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
        // hnsw structure does not require training
        storage->train(n, x);
        is_trained = true;
    }

    void IndexMNRU::add(idx_t n, const float* x) {
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
        FAISS_THROW_IF_NOT(is_trained);
        int n0 = ntotal;
        storage->add(n, x);
        ntotal = storage->ntotal;

        hnsw_add_vertices(*this, n0, n, x, verbose, main_index.levels.size() == ntotal);

        if(/*ntotal%backup_tau==0||*/true){
            // perform backup index update
            backup_index.reset();
            float* entry_vector = new float[d];
            reconstruct(main_index.entry_point, entry_vector);

            std::vector<float> distances(ntotal);
            std::vector<idx_t> labels(ntotal);
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));
            main_index.search(*dis, ntotal, labels.data(), distances.data(),vt);
            std::vector<bool> present(ntotal);
            for(size_t i=0; i<ntotal; i++){
                auto idx = labels[i];
                present[idx]=true;
            }
            std::vector<idx_t> unreachables;
            for(size_t i=0; i<ntotal; i++){
                if(!present[i]){
                    unreachables.push_back(i);
                }
            }
            printf("Found %ld unreachable vectors!\n", unreachables.size());
            std::vector<float> unreachable_vectors(d*unreachables.size());
            for(size_t i=0; i<unreachables.size(); i++){
                auto idx = unreachables[i];
                reconstruct(idx, unreachable_vectors.data()+d*i);
            }

            backup_add_vertices(*this, 0, unreachables.size(), unreachable_vectors.data(),verbose, backup_index.levels.size() == ntotal);
        }
    }

    void IndexMNRU::reset() {
        main_index.reset();
        backup_index.reset();
        storage->reset();
        ntotal = 0;
    }

    void IndexMNRU::reconstruct(idx_t key, float* recons) const {
        storage->reconstruct(key, recons);
    }

    /**************************************************************
 * ReconstructFromNeighbors implementation
 **************************************************************/

    ReconstructFromNeighbors::ReconstructFromNeighbors(
            const IndexMNRU& index,
            size_t k,
            size_t nsq)
            : index(index), k(k), nsq(nsq) {
        M = index.main_index.nb_neighbors(0);
        FAISS_ASSERT(k <= 256);
        code_size = k == 1 ? 0 : nsq;
        ntotal = 0;
        d = index.d;
        FAISS_ASSERT(d % nsq == 0);
        dsub = d / nsq;
        k_reorder = -1;
    }

    void ReconstructFromNeighbors::reconstruct(
            storage_idx_t i,
            float* x,
            float* tmp) const {
        const HNSW& hnsw = index.main_index;
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        if (k == 1 || nsq == 1) {
            const float* beta;
            if (k == 1) {
                beta = codebook.data();
            } else {
                int idx = codes[i];
                beta = codebook.data() + idx * (M + 1);
            }

            float w0 = beta[0]; // weight of image itself
            index.storage->reconstruct(i, tmp);

            for (int l = 0; l < d; l++)
                x[l] = w0 * tmp[l];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t ji = hnsw.neighbors[j];
                if (ji < 0)
                    ji = i;
                float w = beta[j - begin + 1];
                index.storage->reconstruct(ji, tmp);
                for (int l = 0; l < d; l++)
                    x[l] += w * tmp[l];
            }
        } else if (nsq == 2) {
            int idx0 = codes[2 * i];
            int idx1 = codes[2 * i + 1];

            const float* beta0 = codebook.data() + idx0 * (M + 1);
            const float* beta1 = codebook.data() + (idx1 + k) * (M + 1);

            index.storage->reconstruct(i, tmp);

            float w0;

            w0 = beta0[0];
            for (int l = 0; l < dsub; l++)
                x[l] = w0 * tmp[l];

            w0 = beta1[0];
            for (int l = dsub; l < d; l++)
                x[l] = w0 * tmp[l];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t ji = hnsw.neighbors[j];
                if (ji < 0)
                    ji = i;
                index.storage->reconstruct(ji, tmp);
                float w;
                w = beta0[j - begin + 1];
                for (int l = 0; l < dsub; l++)
                    x[l] += w * tmp[l];

                w = beta1[j - begin + 1];
                for (int l = dsub; l < d; l++)
                    x[l] += w * tmp[l];
            }
        } else {
            std::vector<const float*> betas(nsq);
            {
                const float* b = codebook.data();
                const uint8_t* c = &codes[i * code_size];
                for (int sq = 0; sq < nsq; sq++) {
                    betas[sq] = b + (*c++) * (M + 1);
                    b += (M + 1) * k;
                }
            }

            index.storage->reconstruct(i, tmp);
            {
                int d0 = 0;
                for (int sq = 0; sq < nsq; sq++) {
                    float w = *(betas[sq]++);
                    int d1 = d0 + dsub;
                    for (int l = d0; l < d1; l++) {
                        x[l] = w * tmp[l];
                    }
                    d0 = d1;
                }
            }

            for (size_t j = begin; j < end; j++) {
                storage_idx_t ji = hnsw.neighbors[j];
                if (ji < 0)
                    ji = i;

                index.storage->reconstruct(ji, tmp);
                int d0 = 0;
                for (int sq = 0; sq < nsq; sq++) {
                    float w = *(betas[sq]++);
                    int d1 = d0 + dsub;
                    for (int l = d0; l < d1; l++) {
                        x[l] += w * tmp[l];
                    }
                    d0 = d1;
                }
            }
        }
    }

    void ReconstructFromNeighbors::reconstruct_n(
            storage_idx_t n0,
            storage_idx_t ni,
            float* x) const {
//#pragma omp parallel
        {
            std::vector<float> tmp(index.d);
//#pragma omp for
            for (storage_idx_t i = 0; i < ni; i++) {
                reconstruct(n0 + i, x + i * index.d, tmp.data());
            }
        }
    }

    size_t ReconstructFromNeighbors::compute_distances(
            size_t n,
            const idx_t* shortlist,
            const float* query,
            float* distances) const {
        std::vector<float> tmp(2 * index.d);
        size_t ncomp = 0;
        for (int i = 0; i < n; i++) {
            if (shortlist[i] < 0)
                break;
            reconstruct(shortlist[i], tmp.data(), tmp.data() + index.d);
            distances[i] = fvec_L2sqr(query, tmp.data(), index.d);
            ncomp++;
        }
        return ncomp;
    }

    void ReconstructFromNeighbors::get_neighbor_table(storage_idx_t i, float* tmp1)
    const {
        const HNSW& hnsw = index.main_index;
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        size_t d = index.d;

        index.storage->reconstruct(i, tmp1);

        for (size_t j = begin; j < end; j++) {
            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0)
                ji = i;
            index.storage->reconstruct(ji, tmp1 + (j - begin + 1) * d);
        }
    }

/// called by add_codes
    void ReconstructFromNeighbors::estimate_code(
            const float* x,
            storage_idx_t i,
            uint8_t* code) const {
        // fill in tmp table with the neighbor values
        std::unique_ptr<float[]> tmp1(new float[d * (M + 1) + (d * k)]);
        float* tmp2 = tmp1.get() + d * (M + 1);

        // collect coordinates of base
        get_neighbor_table(i, tmp1.get());

        for (size_t sq = 0; sq < nsq; sq++) {
            int d0 = sq * dsub;

            {
                FINTEGER ki = k, di = d, m1 = M + 1;
                FINTEGER dsubi = dsub;
                float zero = 0, one = 1;

                sgemm_("N",
                       "N",
                       &dsubi,
                       &ki,
                       &m1,
                       &one,
                       tmp1.get() + d0,
                       &di,
                       codebook.data() + sq * (m1 * k),
                       &m1,
                       &zero,
                       tmp2,
                       &dsubi);
            }

            float min = HUGE_VAL;
            int argmin = -1;
            for (size_t j = 0; j < k; j++) {
                float dis = fvec_L2sqr(x + d0, tmp2 + j * dsub, dsub);
                if (dis < min) {
                    min = dis;
                    argmin = j;
                }
            }
            code[sq] = argmin;
        }
    }

    void ReconstructFromNeighbors::add_codes(size_t n, const float* x) {
        if (k == 1) { // nothing to encode
            ntotal += n;
            return;
        }
        codes.resize(codes.size() + code_size * n);
//#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            estimate_code(
                    x + i * index.d,
                    ntotal + i,
                    codes.data() + (ntotal + i) * code_size);
        }
        ntotal += n;
        FAISS_ASSERT(codes.size() == ntotal * code_size);
    }

    IndexMNRUFlat::IndexMNRUFlat() {
        is_trained = true;
    }

    IndexMNRUFlat::IndexMNRUFlat(int d, int M, MetricType metric)
            : IndexMNRU(
            (metric == METRIC_L2) ? new IndexFlatL2(d)
                                  : new IndexFlat(d, metric),
            M) {
        own_fields = true;
        is_trained = true;
    }
}


