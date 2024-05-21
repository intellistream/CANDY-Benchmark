/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <fstream>
#include <iostream>
#define HNSW_CACHE_SIZE 8192
namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implementation is heavily influenced by the NMSlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct HNSWStats;

struct SearchParametersHNSW : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;

    ~SearchParametersHNSW() {}
};



// a global structure to maintain previously computed distance between neighbors
struct HNSWDistCache{
	struct DistSlot{
		idx_t src = -1;
		idx_t dest = -1;
		float dist = 0;
		int64_t lru_count = 0;
        uint64_t last_lru = 0;
		DistSlot()=default;
		// if valid, point to next valid, and vice versa
		int64_t next = -1;

		bool is_valid(){
			return src!=-1 && dest!=-1;
		}
	};
	bool is_modulo = true;
	int64_t first_free = 0; // occupied
	int64_t first_unfree = -1; //
	uint64_t num_hits = 0;
	uint64_t num_evictions = 0;

    float average_refault = 0;
	std::vector<DistSlot> slots = std::vector<DistSlot>(HNSW_CACHE_SIZE);
	uint64_t valid_cnt = 0;


	HNSWDistCache(){
		if(is_modulo){
			size_t i=0;
			for(i=0; i<slots.size();i++){
				slots[i].lru_count = -1;

			}


		}
		if(!is_modulo){
			size_t i=0;
			for(i=0;i<slots.size();i++){
				if(i<HNSW_CACHE_SIZE - 1){
					slots[i].next = i+1;

				} else {
					slots[i].next = -1;
				}

			}
		}
	}

	bool put(idx_t src, idx_t dest, float dist){
		if(is_modulo){
			idx_t big, little;
			if(src > dest){
				big = src;
				little = dest;
			} else {
				big = dest;
				little = src;
			}
			idx_t level_1 = big % HNSW_CACHE_SIZE;
			// here 32 is the number of max connections
			idx_t level_2 = little % HNSW_CACHE_SIZE%32;
			idx_t target = level_1 + level_2;
			// circular
			while(target > HNSW_CACHE_SIZE){
				target = target - HNSW_CACHE_SIZE;
			}
			if(slots[target].lru_count<0){
				num_evictions++;
				slots[target].src = src;
				slots[target].dest = dest;
				slots[target].dist = dist;
				slots[target].lru_count = HNSW_CACHE_SIZE;
				return true;
			} else {
				return false;

			}
		}

		// linear cache
		// no need to evict
		if(valid_cnt<HNSW_CACHE_SIZE){
            // put at first_free
            //printf("putting with valid cnt %ld \n", valid_cnt);
			int64_t next = slots[first_free].next;
			int64_t to_put = first_free;
			slots[to_put].src = src;
			slots[to_put].dest = dest;
			slots[to_put].dist = dist;
			slots[to_put].lru_count = HNSW_CACHE_SIZE;
            slots[to_put].last_lru = slots[to_put].lru_count;

			// if no valid slots, first_valid would be -1
			first_free = next;

			slots[to_put].next = first_unfree;

			first_unfree = to_put;

			valid_cnt++;
            //printf("putting at%ld complete\n", next);
			return true;
		} else {
            //printf("putting failed\n");
			return false;
		}
	}
    void print(){
        printf("number of evictions: %ld, number of hits: %ld, average refault %.2f\n", num_evictions, num_hits, average_refault);
        //int64_t next = first_free;
        //printf("free:\n");

        int64_t next = first_unfree;
        //printf("unfree:\n");
        //printf("valid cnt% ld\n", valid_cnt);
        while(next!=-1){
            //printf("unfree %ld src: %ld dest: %ld\n", next,slots[next].src,slots[next].dest);
            next = slots[next].next;
        }
    }

	bool get(idx_t src, idx_t dest, float& dist){
		if(is_modulo){
			idx_t big, little;
			if(src > dest){
				big = src;
				little = dest;
			} else {
				big = dest;
				little = src;
			}
			idx_t level_1 = big % HNSW_CACHE_SIZE;
			// here 32 is the number of max connections
			idx_t level_2 = little % HNSW_CACHE_SIZE%32;
			idx_t target = level_1 + level_2;
			// circular
			while(target > HNSW_CACHE_SIZE){
				target = target - HNSW_CACHE_SIZE;
			}
			if((slots[target].src==src && slots[target].dest==dest) || (slots[target].src == dest && slots[target].dest == src)){
				dist = slots[target].dist;
				num_hits++;
				return true;

			} else {

				slots[target].lru_count -= HNSW_CACHE_SIZE/32;
				return false;
			}
		}
           //printf("getting\n");
		int64_t next = first_unfree;
		int64_t last = -1;
		while(next!=-1){
            //printf("moving to %ld\n", next);
            //printf("next %ld\n", slots[next].next);
			if((slots[next].src==src && slots[next].dest==dest) || (slots[next].src == dest && slots[next].dest == src)){
				slots[next].lru_count+=2;
				dist = slots[next].dist;
				num_hits++;
                auto refault_distance = slots[next].last_lru - slots[next].lru_count;
                average_refault = average_refault + ( refault_distance-average_refault)/num_hits;
                slots[next].last_lru = slots[next].lru_count;
                //printf("getting success\n");
				return true;
			}
			slots[next].lru_count--;
			// free this slot
			if(slots[next].lru_count<=0){
				num_evictions++;
				if(last!=-1){

					int64_t temp_next = slots[next].next;
					int64_t temp_last = last;
					slots[last].next = slots[next].next;
					slots[next].next = first_free;
					first_free = next;
					next = temp_next;
					last = temp_last;
					valid_cnt--;
					continue;
				} else {

					int64_t temp_next = slots[next].next;
					int64_t temp_last = -1;

					first_unfree = slots[next].next;
					slots[next].next = first_free;
					first_free = next;
					next= temp_next;
					last = temp_last;
					valid_cnt--;
					continue;
				}

			}
			last = next;
			next = slots[next].next;
		}
        //printf("getting complete\n");
		return false;
	}
};

struct HNSW_breakdown_stats {
    size_t steps_greedy =
            0; // number of vertices traversing in greedy search in add
    size_t steps_iterating_add =
            0; // number of vertices visited in add_neighbors
    size_t steps_iterating_search =
            0; // number of vertices visited in searching from candidates

    size_t time_greedy_insert = 0;
    size_t time_searching_neighbors_to_add = 0;
    size_t time_add_links = 0;

    size_t time_greedy_search = 0;
    size_t time_search_from_candidates = 0;
    size_t time_dc = 0;
    size_t time_dc_linking = 0;
    size_t step_linking =0;
    size_t step_before_shrinking=0;
    HNSW_breakdown_stats() = default;

    void reset() {
        steps_greedy = 0;
        steps_iterating_add = 0;
        steps_iterating_search = 0;
        time_greedy_insert = 0;
        time_searching_neighbors_to_add = 0;
        time_add_links = 0;
        time_greedy_search = 0;
        time_search_from_candidates = 0;
	time_dc = 0;
	time_dc_linking = 0;
	step_before_shrinking = 0;
	step_linking = 0;
    }


    void print() {
        std::cout << steps_greedy << ",";
        std::cout << steps_iterating_add << ",";
        std::cout << steps_iterating_search << ",";

        std::cout << time_greedy_insert << ",";
        std::cout << time_searching_neighbors_to_add << ",";
        std::cout << time_add_links << ",";

        std::cout << time_greedy_search << ",";
        std::cout << time_search_from_candidates << ",";
	std::cout<<time_dc<<",";
	std::cout<<time_dc_linking<<",";
	std::cout<<step_before_shrinking<<",";
	std::cout<<step_linking<<"\n";
    }
};
struct HNSW {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    typedef std::pair<float, storage_idx_t> Node;

    mutable struct HNSW_breakdown_stats bd_stat;
	mutable struct HNSWDistCache dist_cache;

    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    /// assignment probability to each layer (sum=1)
    std::vector<double> assign_probas;

    /// number of neighbors stored per layer (cumulative), should not
    /// be changed after first add
    std::vector<int> cum_nneighbor_per_level;

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    std::vector<storage_idx_t> neighbors;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 16;

    int M_ = 32;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// number of entry points in levels > 0.
    int upper_beam = 1;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    // methods that initialize the tree sizes

    /// initialize the assign_probas and cum_nneighbor_per_level to
    /// have 2*M links on level 0 and M links on levels > 0
    void set_default_probas(int M, float levelMult);

    /// set nb of neighbors for this level (before adding anything)
    void set_nb_neighbors(int level_no, int n);

    // methods that access the tree sizes

    /// nb of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    /// range of entries in the neighbors table of vertex no at layer_no
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
            const;

    /// only mandatory parameter: nb of neighbors
    explicit HNSW(int M = 32);

    /// pick a random level for a new point
    int random_level();

    /// add n random levels to table (for debugging...)
    void fill_with_random_links(size_t n);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt);

    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. */
    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    /// search interface for 1 point, single thread
    HNSWStats search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            const SearchParametersHNSW* params = nullptr) const;

    /// search only in level 0 from a given vertex
    void search_level_0(
            DistanceComputer& qdis,
            int k,
            idx_t* idxi,
            float* simi,
            idx_t nprobe,
            const storage_idx_t* nearest_i,
            const float* nearest_d,
            int search_type,
            HNSWStats& search_stats,
            VisitedTable& vt) const;

    void reset();

    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;

    int prepare_level_tab(size_t n, bool preset_levels = false);

    static void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size,
	    struct HNSW_breakdown_stats& bd_stats,
		struct HNSWDistCache& dist_cache);

    void permute_entries(const idx_t* map);
};

struct HNSWStats {
    size_t n1, n2, n3;
    size_t ndis;
    size_t nreorder;

    HNSWStats(
            size_t n1 = 0,
            size_t n2 = 0,
            size_t n3 = 0,
            size_t ndis = 0,
            size_t nreorder = 0)
            : n1(n1), n2(n2), n3(n3), ndis(ndis), nreorder(nreorder) {}

    void reset() {
        n1 = n2 = n3 = 0;
        ndis = 0;
        nreorder = 0;
    }

    void combine(const HNSWStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        n3 += other.n3;
        ndis += other.ndis;
        nreorder += other.nreorder;
    }
};

// global var that collects them all
FAISS_API extern HNSWStats hnsw_stats;

} // namespace faiss
