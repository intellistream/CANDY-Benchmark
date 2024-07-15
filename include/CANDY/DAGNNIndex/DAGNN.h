//
// Created by rubato on 12/6/24.
//

#ifndef DAGNN_H
#define DAGNN_H
#include <Utils/IntelliTensorOP.hpp>
#include <vector>
#include <queue>
#include <random>
#include <CANDY/DAGNNIndex/DAGNNUtil.h>
#include <faiss/IndexFlat.h>
#include <omp.h>
#include <boost/math/constants/constants.hpp>
#include <Algorithm/EDMStream.hpp>

namespace CANDY{
struct BreakdownStats {
    size_t steps_greedy = 0;
    size_t steps_expansion = 0;
    size_t steps_pruning = 0;
    size_t steps_greedy_base = 0;
    size_t steps_expansion_base = 0;
    size_t steps_pruning_base = 0;
    size_t prune_times = 0;
    size_t prune_times_base = 0;
    std::vector<std::vector<int64_t>> upper_nodes;

    void print() {
        printf("base: ");
        auto total = steps_greedy_base+steps_expansion_base+steps_pruning_base;
        printf("%ld %f %ld %f %ld %f\n", steps_greedy_base, steps_greedy_base/(total*1.0), steps_expansion_base, steps_expansion_base/(total*1.0), steps_pruning_base, steps_pruning_base/(total*1.0));

        printf("Upper: ");
        auto total_upper = steps_greedy+steps_expansion+steps_pruning;
        printf("%ld %f %ld %f %ld %f\n", steps_greedy, steps_greedy/(total_upper*1.0), steps_expansion, steps_expansion/(total_upper*1.0), steps_pruning, steps_pruning/(total_upper*1.0));

        printf("Prune times: ");
        printf("%ld %ld %f\n", steps_pruning, prune_times, steps_pruning/(prune_times*1.0));

        printf("Prune times Base: ");
        printf("%ld %ld %f\n", steps_pruning_base, prune_times_base, steps_pruning_base/(prune_times_base*1.0));


        printf("Levels: \n");
            for(size_t l=0; l<upper_nodes.size(); l++) {
                printf("%ldth level: %ld\n", l+1, upper_nodes[l].size());
            }

    }
    void reset() {
        steps_greedy = 0;
        steps_expansion = 0;
        steps_pruning = 0;

        steps_greedy_base = 0;
        steps_expansion_base = 0;
        steps_pruning_base = 0;

        prune_times = 0;
        prune_times_base = 0;
    }

    BreakdownStats()=default;
};
/**
* @class
* @brief The dynamic parameter tuning graph that has the structure of HNSW
*/
struct DynamicTuneHNSW{
    using idx_t = int64_t;
    bool verbose = false;
    struct CandidateCloser {
        idx_t id=-1;
        float dist;
        bool operator<(const CandidateCloser& obj1) const{
            return dist<obj1.dist;
        }
        CandidateCloser(float d, idx_t i): dist(d), id(i){}
    };

    struct CandidateFarther {
        idx_t id=-1;
        float dist;
        bool operator<(const CandidateFarther& obj1) const{
            return dist>obj1.dist;
        }
        CandidateFarther(float d, idx_t i): dist(d), id(i){}
    };

    struct EdgeCloser {
        idx_t src=-1;
        idx_t dest = -1;
        size_t idx = -1;
        float dist;

        bool operator<(const EdgeCloser& obj1) const {
            return dist<obj1.dist;
        }
        EdgeCloser(float d, idx_t i, idx_t j, size_t id): dist(d), src(i), dest(j), idx(id) {}
    };

    struct EdgeFarther {
        idx_t src=-1;
        idx_t dest = -1;
        size_t idx = -1;
        float dist;


        bool operator<(const EdgeFarther& obj1) const {
            return dist>obj1.dist;
        }
        EdgeFarther(float d, idx_t i, idx_t j, size_t id): dist(d), src(i), dest(j), idx(id){}
    };

    struct Node{
        idx_t id = -1;
        size_t level = -1;
        int64_t bottom_connections = 0;
        // Neighbor* neighbors;
        std::vector<std::vector<idx_t>> neighbors;
        std::vector<std::vector<float>> distances;
        bool operator<(const Node& obj1) const {
            return bottom_connections<obj1.bottom_connections;
        }
    };


    struct ordered_map {
        std::map<idx_t, Node*> data_map;        // Using std::map for ordering by id
        std::unordered_set<idx_t> id_set;      // To maintain uniqueness of ids
        size_t max_size;
        size_t evictions = 0;
        // Function to insert or update a Node
        ordered_map()=default;
        void insert(idx_t id,  Node* node) {
            // Check if the id already exists
            auto it = data_map.find(id);
            // in case of sorting by connections
            if (it != data_map.end()) {
                // Update existing Node
                it->second = node;

            } else {
                // Insert new Node
                if (data_map.size() >= max_size) {
                    // Remove the smallest (first) element in the map
                    auto smallest_it = data_map.begin();
                    int smallest_id = smallest_it->first;
                    data_map.erase(smallest_it);
                    id_set.erase(smallest_id);
                    evictions++;
                }
                data_map.emplace(id, node);
                id_set.insert(id); // Update id_set for uniqueness check
            }
        }

        // Function to get the Node by id
        Node* get(idx_t id) {
            auto it = data_map.find(id);
            if (it != data_map.end()) {
                return it->second;
            } else {
                return nullptr; // Not found
            }
        }

        // Function to remove a Node by id
        void remove(idx_t id) {
            auto it = data_map.find(id);
            if (it != data_map.end()) {
                data_map.erase(it);
                id_set.erase(id);
            }
        }

        // Function to print all Nodes (sorted by id)
        void print_all() {
            for (const auto& pair : data_map) {
                std::cout << "ID: " << pair.first << ", Value: " << pair.second->bottom_connections << std::endl;
            }
            printf("Number of evictions: %ld\n", evictions);
        }
        void clear() {
            data_map.clear();
            id_set.clear();
            evictions=0;
        }
        // Function to check if id exists
        bool contains(idx_t id) const {
            return id_set.find(id) != id_set.end();
        }
    };




    /// The auto-tune parameters
    struct DynamicTuneParams{
        int64_t efConstruction = 40;
        int64_t efSearch = 16;
        int64_t bottom_connections_upper_bound = 64;
        int64_t bottom_connections_lower_bound = 32;
        int64_t distance_computation_opt = 0;
        float rng_alpha = 1.0;
        /* Number of outwards steps we define as a cluster. Suppose we have 1->2, 1->3, 3->4, 4->5
         and clusterExpansionStep = 2, then for cluster centering vertex_1, we have vertex_{1,2,3,4}
         as the cluster centering vertex_1
         */
        int64_t clusterExpansionStep = 2;

        int64_t clusterInnerConnectionThreshold = 1;
        int64_t optimisticN = 16;
        size_t discardN = 0;
        size_t discardClusterN =32;
        double discardClusterProp=0.3;

        double degree_std_range = 1.5;
        double degree_allow_range = 0.25;

        bool sparsePreference = true;
        double neighborDistanceThreshold = 0.0;
        std::vector<float> routeDrifts;
        DynamicTuneParams()=default;
        DynamicTuneParams(const int64_t M, const int64_t dim) {
            bottom_connections_lower_bound = M;
            bottom_connections_upper_bound = M;
            routeDrifts.resize(dim, 0.0);
        }


    };

    struct DRLStates {
        std::vector<float> value_average;
        std::vector<float> value_variance;
        size_t ntotal = 0;

    };
    struct GlobalGraphStats : DRLStates{

        int64_t degree_sum = 0.0;
        double degree_variance = 0.0;
        double neighbor_distance_sum = 0.0;
        double neighbor_distance_variance = 0.0;
        int64_t steps_taken_max = 0;
        double steps_taken_avg = 0.0;
        int64_t steps_expansion_average = 0.0;
        void print() {
            //printf("ntotal=%ld\n", ntotal);
            //printf("degree average = %lf , degree variance = %lf\n", degree_sum/(ntotal*1.0), degree_variance);
            //printf("avg neighbor distance = %f , neighbor distance var = %f\n", neighbor_distance_sum/, neighbor_distance_variance);
        }
        ///TODO: DEFINE THIS!



    };
    /// Used for updating the global graph states
    struct BatchDataStates : DRLStates{
        size_t old_ntotal = 0;
        // sigma ^2
        int64_t degree_sum_new = 0;
        double degree_variance_new = 0.0;

        /// updating old variance to continue the update
        double degree_variance_old = 0.0;
        int64_t degree_sum_old = 0;

        double neighbor_distance_sum_new = 0.0;
        double neighbor_distance_variance_new = 0.0;
        double neighbor_distance_sum_old = 0.0;
        double neighbor_distance_variance_old = 0.0;

        int64_t steps_taken_sum = 0;
        int64_t steps_taken_max = 0;

        int64_t steps_expansion_sum = 0;



        void reset() {
            ntotal = 0;
            degree_sum_new = 0;
            degree_variance_new = 0.0;
            neighbor_distance_sum_new = 0.0;
            neighbor_distance_variance_new = 0.0;
            for(size_t i=0; i<value_average.size(); i++) {
                value_average[i]=0.0;
                value_variance[i] = 0.0;
            }
        }

    };

    struct WindowStates : DRLStates {
        size_t oldWindowSize = 50;
        size_t newWindowSize = 100;
        ordered_map oldVertices;
        ordered_map newVertices;
        void init() {
            oldVertices.max_size = oldWindowSize;
            newVertices.max_size = newWindowSize;
        }
        void reset() {
            newVertices.clear();
            oldVertices.clear();
        }
        WindowStates()=default;
    };

    struct GraphStates{
        int64_t vecDim = 0;
        GlobalGraphStats global_stat;
        BatchDataStates time_local_stat;
        WindowStates window_states;
        void init() {
            global_stat.value_average.resize(vecDim,0.0);
            global_stat.value_variance.resize(vecDim, 0.0);
            time_local_stat.value_average.resize(vecDim,0.0);
            time_local_stat.value_variance.resize(vecDim,0.0);
            window_states.init();
        }
        void forward() {
            global_stat.ntotal+=time_local_stat.ntotal;
            time_local_stat.reset();
            time_local_stat.old_ntotal = global_stat.ntotal;

            /// give timelocal previous var
            time_local_stat.degree_sum_old = global_stat.degree_sum;
            time_local_stat.degree_variance_old = global_stat.degree_variance;
            time_local_stat.neighbor_distance_sum_old = global_stat.neighbor_distance_sum;
            time_local_stat.neighbor_distance_variance_old = global_stat.neighbor_distance_variance;

            time_local_stat.steps_expansion_sum = 0;
            time_local_stat.steps_taken_sum = 0;
            time_local_stat.steps_taken_max = 0;
        }
        /// others
    };

    int64_t vecDim;
    DAGNN::DistanceQueryer* disq = nullptr;

    DynamicTuneParams dynamicParams;
    GraphStates graphStates;
    BreakdownStats bd_stats;
    // The database, only for retrieving vector using reconstruct();
    faiss::IndexFlat* storage = nullptr;
    /// The graph neighbor structure. Use linkLists[idx] to locate a Nodes' neighbor list
    std::vector<Node*> linkLists;
    std::vector<idx_t> entry_points;
    /// Used for clustering
    SESAME::EDMStream* stream;
    /// HNSW level assigning
    std::vector<double> assign_probs;
    /// cumulative number of neighbors stored per level
    std::vector<size_t> cum_nneighbor_per_level;
    size_t max_level = -1;




    DynamicTuneHNSW(const int64_t M, const int64_t dim, const DAGNN::dagnn_metric_t metric, const DynamicTuneParams setting) {
        SESAME::param_t stream_param;
        stream = new SESAME::EDMStream(stream_param);
        stream->Init();


        vecDim = dim;
        graphStates.vecDim = dim;
        set_default_probs(M, 1.0/log(M));
        disq = new DAGNN::DistanceQueryer(metric,dim);
        if(metric == DAGNN_METRIC_L2){
            storage = new faiss::IndexFlatL2(vecDim);
        } else {
            storage = new faiss::IndexFlatIP(vecDim);
        }
        dynamicParams = setting;


        graphStates.init();

        assert(storage);
    }

    float* get_vector(const idx_t i) const {
        assert(i<storage->ntotal);
        auto x = new float[vecDim];
        storage->reconstruct(i, x);
        return x;
    }

    /// HNSW init
    /// initialize probs and cum_nb_neighbors_per_levels
    void set_default_probs(const int64_t M, const float levelMult){
        int64_t nn=0;
        cum_nneighbor_per_level.push_back(0);
        for(int64_t level=0;;level++){

            float a = exp(-level/levelMult);
            float b = 1-exp(-1/levelMult);
            float prob = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if(prob < 1e-9){
                break;
            }
            assign_probs.push_back(prob);

            if(level!=0){
                nn+=M;
            } else {
                nn+=M*2;
            }
            cum_nneighbor_per_level.push_back(nn);
        }
        bd_stats.upper_nodes.resize(assign_probs.size()-1);

    }

    void assign_levels(idx_t n) {
        idx_t n0 = storage->ntotal;
        std::mt19937_64 gen;
        gen = std::mt19937_64(114514);
        std::uniform_real_distribution<> distrib(0,1);

        for(idx_t i=0; i<n; i++) {
            double rand = distrib(gen);
            int64_t level = 0;
            for(level = 0; level<assign_probs.size(); level++) {
                if(rand<assign_probs[level]) {
                    break;
                }
                rand -= assign_probs[level];
            }
            if(level==assign_probs.size()) {
                level--;
            }
            //if(max_level < level) {
            //    max_level = level;
            //}
            auto new_node = new Node();
            new_node->id = n0+i;
            new_node->level = level;
            if(level>0) {
                bd_stats.upper_nodes[level-1].push_back(i);
            }
            for(size_t l=0; l<=level; l++) {
                auto l_neighbors = std::vector<idx_t>(nb_neighbors(l),-1);
                auto l_distances = std::vector<float>(nb_neighbors(l), -0.0);
                new_node->neighbors.push_back(l_neighbors);
                new_node->distances.push_back(l_distances);
            }

            // use cum_nneighbor_per_level[new_node.level] to get total number of levels
            linkLists.push_back(new_node);
        }
    }

    /// use level+1 to get level's nb_neighbors and cum_nneighbor
    size_t nb_neighbors(const size_t layer_no) const{
        return cum_nneighbor_per_level[layer_no+1]-cum_nneighbor_per_level[layer_no];
    }

    // void neighbor_range(idx_t idx, size_t level, size_t& begin, size_t& end) const{
    //     begin = cum_nneighbor_per_level[level];
    //     end = cum_nneighbor_per_level[level+1];
    // }


    /// Index functionalities
    // add break down into 4 parts: greedy, candidate_add, prune and link
    void add(idx_t n, float* x);

    void greedy_insert(DAGNN::DistanceQueryer& disq, Node& node,DAGNN::VisitedTable& vt);

    void greedy_insert_top(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, int64_t& steps_taken);

    void greedy_insert_upper(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken);

    void greedy_insert_base(DAGNN::DistanceQueryer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken);

    void link_from(DAGNN::DistanceQueryer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt);

    void link_from_base(DAGNN::DistanceQueryer& disq, idx_t idx, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt);

    void candidate_select(DAGNN::DistanceQueryer& disq, size_t level, std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt);

    void candidate_select_base(DAGNN::DistanceQueryer& disq, std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt);

    void prune(DAGNN::DistanceQueryer& disq,size_t level, std::priority_queue<CandidateCloser>& candidates);

    void prune_base(DAGNN::DistanceQueryer& disq, std::priority_queue<CandidateCloser>& candidates);

    void link(size_t level, idx_t entry, std::priority_queue<CandidateCloser>& candidates);

    void add_link(DAGNN::DistanceQueryer& disq, idx_t src, idx_t dest, size_t level);

    void add_link_base(DAGNN::DistanceQueryer& disq, idx_t src, idx_t dest, int64_t& prev_degree, int64_t& current_degree);

    void search(DAGNN::DistanceQueryer& disq, idx_t annk, idx_t* results, float* distances, DAGNN::VisitedTable& vt);

    void greedy_search(DAGNN::DistanceQueryer& disq, size_t level, idx_t entry, float dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    void greedy_search_upper(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    void greedy_search_base(DAGNN::DistanceQueryer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    int candidate_search(DAGNN::DistanceQueryer& disq, size_t level, idx_t annk, idx_t* results, float* distances,DAGNN::MinimaxHeap& candidates,DAGNN::VisitedTable& vt);

    void cutEdgesBase(idx_t src);

    /// acquire all nodes around a cluster to candidates
    void getCluster(idx_t src, std::priority_queue<EdgeFarther>& edges);
    void getCluster(idx_t src, std::priority_queue<EdgeCloser>& edges);

    void processNewWindow();

    void processOldWindow();

    /// 0 oldVertices, 1 newVertices
    void cutEdgesWindow(WindowStates& window_states, int64_t mode);
    void swapEdgesWindow(WindowStates& window_states, int64_t mode);
    void linkEdgesWindow(WindowStates& window_states, int64_t mode);

    /// use for debug
    void direct_link(idx_t x, idx_t y, size_t level) {
        auto node_x = linkLists[x];
        auto node_y = linkLists[y];
        DAGNN::DistanceQueryer disq(vecDim);
        if(node_x->level < level || node_y->level < level){
            printf("unable to add link for %ld and %ld\n", node_x->id, node_y->id);
            return;
        }
        auto vec_x = get_vector(x);
        auto vec_y = get_vector(y);
        auto dist = disq.InnerProduct(vec_x, vec_y, vecDim);
        auto size = nb_neighbors(level);
        for(size_t i=0; i<size; i++) {
            if(node_x->neighbors[level][i]==-1) {
                node_x->neighbors[level][i]=y;
                node_x->distances[level][i]=dist;
                break;
            }
        }
        for(size_t i=0; i<size; i++) {
            if(node_y->neighbors[level][i]==-1) {
                node_y->neighbors[level][i]=x;
                node_y->distances[level][i]=dist;
                break;
            }
        }
        printf("add link for %ld and %ld\n", node_x->id, node_y->id);
    }

    /// Dynamic tune specific
    bool updateParams(const DynamicTuneParams& dp){
        dynamicParams = dp;
        return true;
    }

    bool performAction(const size_t action_num){
        return true;
    }
    /// Using time_local_stat to update global_stat
    void updateGlobalState();




    /// Dynamic Actions




};

/**
* @class DynamicTuneGraph
* @brief The dynamic parameter tuning graph that has only one level
*/
struct DynamicTunePlate : public DynamicTuneHNSW{





};


}

#endif //DAGNN_H
