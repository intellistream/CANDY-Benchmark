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
#include <faiss/impl/DistanceComputer.h>
#define chronoElapsedTime(start) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
namespace CANDY{


enum dynamic_action_num{
    do_nothing,
    bad_link_cut_old,
    bad_link_cut_new,
    outwards_link_old,
    outwards_link_new,
    //DEG_refine_old,
    //DEG_refine_new,
    backtrack_candidate,
    intercluster_link,
    lift_cluster_center,
    lower_navigation_point,
    increase_rng_alpha,
    decrease_rng_alpha,
    increase_cluster_expansion,
    decrease_cluster_expansion,
    increase_cluster_innerconnection_threshold,
    decrease_cluster_innerconnection_threshold,
    increase_discardN,
    decrease_discardN,
    increase_optimisticN,
    decrease_optimisticN,
    increase_discardClusterProp,
    decrease_discardClusterProp,
    increase_discardClusterN,
    decrease_discardClusterN,
    increase_expansionConstruction,
    decrease_expansionConstruction,
    increase_degree_std_range,
    decrease_degree_std_range,
    increase_degree_allow_range,
    decrease_degree_allow_range,
    increase_sparsePreference,
    decrease_sparsePreference,
    increase_neighborDistanceThreshold,
    decrease_neighborDistanceThreshold,
    increasae_max_backtrack_steps,
    decrease_max_backtrack_steps,
    increase_steps_above_avg,
    decrease_steps_above_avg,
    increase_steps_above_max,
    decrease_steps_above_max,
    increase_nb_navigation_paths,
    decrease_nb_navigation_paths,
    increase_expiration_timestamp,
    decrease_expiration_timestamp,
    increase_degree_lift_range,
    decrease_degree_lift_range

};
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
    uint64_t timestamp = 0;
    omp_lock_t state_lock;
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
        std::unordered_map<idx_t, Node*> data_map;        // Using std::map for ordering by id
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
                    return;
                    // Remove the smallest (first) element in the map
                    auto smallest_it = data_map.begin();
                    std::advance(smallest_it, max_size/2);
                    auto smallest_id = smallest_it->first;
                    data_map.erase(smallest_it);
                    id_set.erase(smallest_id);
                    evictions++;
                }
                if(data_map.size()<max_size) {
                    data_map[id]=node;
                    id_set.insert(id); // Update id_set for uniqueness check
                }
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



    /// Action set IV: Parameters modification
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

        double clusterInnerConnectionThreshold = 0.5;
        int64_t optimisticN = 0;
        size_t discardN = 0;
        size_t discardClusterN =32;
        double discardClusterProp=0.3;

        double degree_std_range = 1.5;
        double degree_allow_range = 0.5;
        double degree_lift_range = 1.75;

        bool sparsePreference = true;
        double neighborDistanceThreshold = 0.5;
        std::vector<float> routeDrifts;

        size_t expiration_timestamp = 1500;

        size_t max_backtrack_steps = 20;
        size_t steps_above_avg = 50;
        size_t steps_above_max = 20;

        size_t nb_navigation_paths =16;

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
        size_t hierarchyWindowSize = 30;
	size_t last_action = 0;
    size_t last_insertion_latency = 0;
    size_t last_search_latency = 0;
    size_t last_dco_nums = 0;
    float last_recall = 0;
        ordered_map oldVertices;
        ordered_map newVertices;
        ordered_map hierarchyVertices;
        void init() {
            oldVertices.max_size = oldWindowSize;
            newVertices.max_size = newWindowSize;
            hierarchyVertices.max_size = hierarchyWindowSize;
        }
        void reset() {
            //newVertices.clear();
            oldVertices.clear();
            hierarchyVertices.clear();
        }
        size_t get_count(int i){
            if(i==0){
                return oldVertices.data_map.size();
            } else if(i==1){
                return newVertices.data_map.size();
            } else if(i==2){
                return hierarchyVertices.data_map.size();
            }
            return -1;
        }
        WindowStates()=default;
    };

    struct GraphStates{
        int64_t vecDim = 0;
        GlobalGraphStats global_stat;
        BatchDataStates time_local_stat;
        WindowStates window_states;
        GraphStates()=default;

        GraphStates(const GraphStates& other):
        vecDim(other.vecDim),
        global_stat(other.global_stat),
        time_local_stat(other.time_local_stat)
        {
            window_states.oldWindowSize = other.window_states.oldWindowSize;
            window_states.newWindowSize = other.window_states.newWindowSize;
            window_states.hierarchyWindowSize=other.window_states.hierarchyWindowSize;
            window_states.last_action = other.window_states.last_action;
            window_states.last_insertion_latency = other.window_states.last_insertion_latency;
            window_states.last_search_latency = other.window_states.last_search_latency;
            window_states.last_dco_nums = other.window_states.last_dco_nums;
            window_states.last_recall = other.window_states.last_recall;
            window_states.init();
            // outside this function we will add pointers to copied' nodes to order_maps


        }
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
	
        void print() {
            std::ofstream outfile("states.csv", std::ios_base::app);
            if (!outfile.is_open()){
                std::cerr << "Failed to open states.csv" << std::endl;
                return;
            }
            // Write GlobalGraphStats
            outfile << global_stat.ntotal << "," << global_stat.degree_sum << "," << global_stat.degree_variance << ","<< global_stat.neighbor_distance_sum<< "," << global_stat.neighbor_distance_variance << "," << global_stat.steps_taken_max << "," << global_stat.steps_taken_avg << "," << global_stat.steps_expansion_average << ",";
            // Write BatchDataStates
            outfile << time_local_stat.ntotal << "," << time_local_stat.old_ntotal << "," << time_local_stat.degree_sum_new << "," << time_local_stat.degree_variance_new << "," << time_local_stat.degree_variance_old << "," << time_local_stat.degree_sum_old << "," << time_local_stat.neighbor_distance_sum_new << "," << time_local_stat.neighbor_distance_variance_new << "," << time_local_stat.neighbor_distance_sum_old << "," << time_local_stat.neighbor_distance_variance_old << "," << time_local_stat.steps_taken_sum << "," << time_local_stat.steps_taken_max << "," << time_local_stat.steps_expansion_sum << ",";
            // Write WindowStates
            outfile << window_states.get_count(0) << "," << window_states.get_count(1) << "," << window_states.get_count(2)<<","<<window_states.last_action<<","<<window_states.last_insertion_latency<<","<<window_states.last_search_latency<<","<<window_states.last_dco_nums<<","<<window_states.last_recall;                                                 outfile << std::endl;
            outfile.close();
       }
       void print(size_t action_num) {
            std::ofstream outfile("states"+std::to_string(action_num)+".csv", std::ios_base::app);
            if (!outfile.is_open()){
                std::cerr << "Failed to open states.csv" << std::endl;
                return;
            }
            // Write GlobalGraphStats
            outfile << global_stat.ntotal << "," << global_stat.degree_sum << "," << global_stat.degree_variance << ","<< global_stat.neighbor_distance_sum<< "," << global_stat.neighbor_distance_variance << "," << global_stat.steps_taken_max << "," << global_stat.steps_taken_avg << "," << global_stat.steps_expansion_average << ",";
            // Write BatchDataStates
            outfile << time_local_stat.ntotal << "," << time_local_stat.old_ntotal << "," << time_local_stat.degree_sum_new << "," << time_local_stat.degree_variance_new << "," << time_local_stat.degree_variance_old << "," << time_local_stat.degree_sum_old << "," << time_local_stat.neighbor_distance_sum_new << "," << time_local_stat.neighbor_distance_variance_new << "," << time_local_stat.neighbor_distance_sum_old << "," << time_local_stat.neighbor_distance_variance_old << "," << time_local_stat.steps_taken_sum << "," << time_local_stat.steps_taken_max << "," << time_local_stat.steps_expansion_sum << ",";
            // Write WindowStates
            outfile << window_states.get_count(0) << "," << window_states.get_count(1) << "," << window_states.get_count(2)<<","<<window_states.last_action<<","<<window_states.last_insertion_latency<<","<<window_states.last_search_latency<<","<<window_states.last_dco_nums<<","<<window_states.last_recall;                                                 outfile << std::endl;
            outfile.close();
        }
    };
    bool is_datamining = false;
    bool is_training = false;
    bool is_greedy = false;
    uint64_t num_dco = 0; // used as reward
    uint64_t time_dco = 0;

    uint64_t num_dco_upper = 0;
    uint64_t time_dco_upper = 0;

    uint64_t num_dco_base = 0;
    uint64_t time_dco_base = 0;

    uint64_t num_dco_expansion = 0;
    uint64_t time_dco_expansion = 0;


    size_t datamining_search_select = 20;
    size_t datamining_search_annk = 25;
    int64_t vecDim;

    DynamicTuneParams dynamicParams;
    GraphStates graphStates;
    BreakdownStats bd_stats;
    // The database, only for retrieving vector using reconstruct();
    faiss::IndexFlat* storage = nullptr;
    /// The graph neighbor structure. Use linkLists[idx] to locate a Nodes' neighbor list
    std::vector<Node*> linkLists;


    std::vector<bool> deleteLists;
    std::vector<idx_t> entry_points;
    std::vector<uint64_t> last_visited;
    /// HNSW level assigning
    std::vector<double> assign_probs;
    /// cumulative number of neighbors stored per level
    std::vector<size_t> cum_nneighbor_per_level;
    size_t max_level = -1;
    /// 0 for tombstone; 1 for global reconnect; 2 for local reconnect
    int delete_mode = 0;



    DynamicTuneHNSW(const int64_t M, const int64_t dim, const DAGNN::dagnn_metric_t metric, const DynamicTuneParams setting) {
        vecDim = dim;
        graphStates.vecDim = dim;
        omp_init_lock(&state_lock);
        set_default_probs(M, 1.0/log(M));
        if(metric == DAGNN_METRIC_L2){
            storage = new faiss::IndexFlatL2(vecDim);
        } else {
            storage = new faiss::IndexFlatIP(vecDim);
        }
        dynamicParams = setting;


        graphStates.init();

        assert(storage);
    }

    DynamicTuneHNSW(const DynamicTuneHNSW& other):
        vecDim(other.vecDim),
        dynamicParams(other.dynamicParams),
        graphStates(other.graphStates),
        bd_stats(other.bd_stats),
        storage(other.storage),
        deleteLists(other.deleteLists),
        entry_points(other.entry_points),
        last_visited(other.last_visited),
        assign_probs(other.assign_probs),
        cum_nneighbor_per_level(other.cum_nneighbor_per_level),
        max_level(other.max_level),
        delete_mode(other.delete_mode){
        omp_init_lock(&state_lock);
        // copy linkLists
        linkLists.reserve(other.linkLists.size());
        for (const Node* node : other.linkLists) {
            if (node) {
                linkLists.push_back(new Node(*node));
            } else {
                linkLists.push_back(nullptr);
            }
        }

        // copy ordered_map with id
        // oldVertices
        for (const auto& id : other.graphStates.window_states.oldVertices.id_set) {
            auto node = linkLists[id];
            graphStates.window_states.oldVertices.insert(id, node);
        }
        // newVertices
        for (const auto& id : other.graphStates.window_states.newVertices.id_set) {
            auto node = linkLists[id];
            graphStates.window_states.newVertices.insert(id, node);
        }
        // hierarchyVertices
        for (const auto& id : other.graphStates.window_states.hierarchyVertices.id_set) {
            auto node = linkLists[id];
            graphStates.window_states.hierarchyVertices.insert(id, node);
        }

    }

    DynamicTuneHNSW& operator=(const DynamicTuneHNSW& other){
        if(this!=&other){
            vecDim = other.vecDim;
            dynamicParams = other.dynamicParams;
            graphStates = other.graphStates;
            bd_stats = other.bd_stats;
            storage = other.storage;
            deleteLists = other.deleteLists;
            entry_points = other.entry_points;
            last_visited = other.last_visited;
            assign_probs = other.assign_probs;
            cum_nneighbor_per_level = other.cum_nneighbor_per_level;
            max_level = other.max_level;
            delete_mode = other.delete_mode;
            omp_init_lock(&state_lock);
            linkLists.reserve(other.linkLists.size());
            for (const Node* node : other.linkLists) {
                if (node) {
                    linkLists.push_back(new Node(*node));
                } else {
                    linkLists.push_back(nullptr);
                }
            }
            // copy ordered_map with id
            // oldVertices
            for (const auto& id : other.graphStates.window_states.oldVertices.id_set) {
                auto node = linkLists[id];
                graphStates.window_states.oldVertices.insert(id, node);
            }
            // newVertices
            for (const auto& id : other.graphStates.window_states.newVertices.id_set) {
                auto node = linkLists[id];
                graphStates.window_states.newVertices.insert(id, node);
            }
            // hierarchyVertices
            for (const auto& id : other.graphStates.window_states.hierarchyVertices.id_set) {
                auto node = linkLists[id];
                graphStates.window_states.hierarchyVertices.insert(id, node);
            }
        }
        return *this;
    };

    ~DynamicTuneHNSW(){
            for (Node* node : linkLists) {
                delete node;
            }
    };


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
            last_visited.push_back(0);
            deleteLists.push_back(false);
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

    void greedy_insert(faiss::DistanceComputer& disq, Node& node,DAGNN::VisitedTable& vt, std::vector<omp_lock_t>& locks);

    void greedy_insert_top(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, int64_t& steps_taken);

    void greedy_insert_upper(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken);

    void greedy_insert_base(faiss::DistanceComputer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates, int64_t& steps_taken);

    void link_from(faiss::DistanceComputer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt);

    void link_from_lock(faiss::DistanceComputer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt, omp_lock_t* locks);

    void link_from_base(faiss::DistanceComputer& disq, idx_t idx, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt);

    void link_from_base_lock(faiss::DistanceComputer& disq, idx_t idx, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates,DAGNN::VisitedTable& vt, omp_lock_t* locks);

    void candidate_select(faiss::DistanceComputer& disq, size_t level, std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt);

    void candidate_select_base(faiss::DistanceComputer& disq, std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt);

    void prune(faiss::DistanceComputer& disq,size_t level, std::priority_queue<CandidateCloser>& candidates);

    void prune_base(faiss::DistanceComputer& disq, std::priority_queue<CandidateCloser>& candidates);

    void link(size_t level, idx_t entry, std::priority_queue<CandidateCloser>& candidates);

    void add_link(faiss::DistanceComputer& disq, idx_t src, idx_t dest, size_t level);

    void add_link_base(faiss::DistanceComputer& disq, idx_t src, idx_t dest, int64_t& prev_degree, int64_t& current_degree);

    void search(faiss::DistanceComputer& disq, idx_t annk, idx_t* results, float* distances, DAGNN::VisitedTable& vt);

    void greedy_search(faiss::DistanceComputer& disq, size_t level, idx_t entry, float dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    void greedy_search_upper(faiss::DistanceComputer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    void greedy_search_base(faiss::DistanceComputer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates);

    int candidate_search(faiss::DistanceComputer& disq, size_t level, idx_t annk, idx_t* results, float* distances,DAGNN::MinimaxHeap& candidates,DAGNN::VisitedTable& vt);

    void cutEdgesBase(idx_t src);

    /// acquire all nodes around a cluster to candidates
    void getCluster(idx_t src, std::priority_queue<EdgeFarther>& edges, int expansion);
    void getCluster(idx_t src, std::priority_queue<EdgeCloser>& edges, int expansion);

    void processNewWindow();

    void processOldWindow();

    /// Action set I: Cluster Optimization
    /// 0 oldVertices, 1 newVertices
    void cutEdgesWindow(WindowStates& window_states, int64_t mode);
    void swapEdgesWindow(WindowStates& window_states, int64_t mode);
    void linkEdgesWindow(WindowStates& window_states, int64_t mode);
    bool improveEdges(idx_t vertex1, idx_t vertex2, float dist12);
    bool improveEdges(idx_t vertex1, idx_t vertex2, idx_t vertex3, idx_t vertex4, float total_gain, const uint8_t steps);

    /// Action set III: Hierarchy optimization
    void liftClusterCenter(faiss::DistanceComputer& disq, idx_t src, DAGNN::VisitedTable& vt);
    void degradeNavigationPoint(faiss::DistanceComputer& disq, idx_t src, DAGNN::VisitedTable& vt);
    void hierarchyOptimizationDegradeWIndow(WindowStates& window_states);
    void hierarchyOptimizationLiftWIndow(WindowStates& window_states);

    /// Action set II: Explorative navigation
    void backtrackCandidate(faiss::DistanceComputer& disq, idx_t src, DAGNN::VisitedTable& vt);
    void linkClusters(faiss::DistanceComputer& disq, idx_t src, idx_t dest, DAGNN::VisitedTable& vt);
    void navigationBacktrackWindow(WindowStates& window_states);
    void linkClusterWindow(WindowStates& window_states);

    void deleteVector(idx_t n, float* x);
    void deleteVectorByIndex(const std::vector<idx_t>& idx);
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

    bool performAction(const size_t action_num);
    /// Using time_local_stat to update global_stat
    void updateGlobalState();




    /// Dynamic Actions




    void randomPickAction();

    /// would check if y is in x's neighbors at the bottom level
    bool hasEdge(idx_t x, idx_t y) {
        auto node_x = linkLists[x];
        auto node_y = linkLists[y];
        auto neighbor_y = node_y->neighbors[0];
        auto neighbor_x = node_x->neighbors[0];
        for (int i = 0; i < nb_neighbors(0); i++) {
            if (neighbor_x[i] == y) {
                return true;
            }
            if (neighbor_x[i] == -1) {
                break;
            }
        }
        return false;
    }

    void changeEdge(idx_t src, idx_t old_dest, idx_t new_dest, float distance);

    void removeDuplicateEdge(idx_t src);

    bool amendEdge(idx_t src, idx_t dest, float distance){
        auto node_src = linkLists[src];
        auto neighbor_src = node_src->neighbors[0];
        auto distance_src = node_src->distances[0];
        for(int i=0; i<nb_neighbors(0); i++){
            if(neighbor_src[i]<0){
                neighbor_src[i] = dest;
                distance_src[i] = distance;
                return true;
            }
        }
        return false;
    }




//    bool checkRNG(idx_t vertex_index, idx_t target_index, float vertex_target_distance){
//        auto vertex = linkLists[vertex_index];
//        auto target = linkLists[target_index];
//        auto neighbors = vertex->neighbors[0];
//        auto target_neighbors = target->neighbors[0];
//        auto distances = vertex->distances[0];
//        auto target_distances = target->distances[0];
//
//        for(int i=0; i<nb_neighbors(0); i++){
//            if(neighbors[i]==-1){
//                return dynamicParams.sparsePreference;
//            }
//            if(neighbors[i]==-2){
//                continue;
//            }
//
//        }
//    }

        struct NegativeDistanceComputer : faiss::DistanceComputer {
            /// owned by this
            DistanceComputer* basedis;
            using idx_t = int64_t;

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

        faiss::DistanceComputer* storage_distance_computer(const faiss::Index* storage) {
            if (is_similarity_metric(storage->metric_type)) {
                return new NegativeDistanceComputer(storage->get_distance_computer());
            } else {
                return storage->get_distance_computer();
            }
        }
};

/**
* @class DynamicTuneGraph
* @brief The dynamic parameter tuning graph that has only one level
*/
struct DynamicTunePlate : public DynamicTuneHNSW{





};


}

#endif //DAGNN_H
