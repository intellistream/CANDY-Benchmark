//
// Created by rubato on 12/6/24.
//
#include<CANDY/DAGNNIndex/DAGNN.h>


void CANDY::DynamicTuneHNSW::add(idx_t n, float* x) {
    // here params are already set or updated
    assert(n>0);
    assert(x);

    assign_levels(n);
    idx_t n0 = storage->ntotal;
    storage->add(n,x);
    DAGNN::DistanceQueryer disq(vecDim);
    for(int64_t d=0; d<vecDim; d++) {
        printf("%.2f, ", graphStates.global_stat.value_average[d]);
    }
    for(idx_t i=0;i<2; i++) {
        for(int64_t d=0; d<vecDim; d++) {
            auto m = (x[vecDim*i+d]-graphStates.global_stat.value_average[d])/(n0+i+1);
            graphStates.global_stat.value_average[d] += m;
        }
    }
    for(idx_t i=0; i<n; i++) {
        disq.set_query(x+vecDim*i);
        std::priority_queue<Candidate> candidates;
        auto node = linkLists[i+n0];
        greedy_insert(disq, *node, candidates);
    }

    // for(int64_t d=0; d<vecDim; d++) {
    //     printf("%.2f, ", graphStates.global_stat.value_average[d]);
    // }
    // printf("\n");



}



void CANDY::DynamicTuneHNSW::greedy_insert(DAGNN::DistanceQueryer& disq, CANDY::DynamicTuneHNSW::Node& node, std::priority_queue<Candidate>& candidates){
    idx_t nearest = -1;
    auto assigned_level = node.level;
    if(entry_points.empty()) {
        entry_points.push_back(node.id);
        max_level = assigned_level;
    }

    nearest = entry_points[0];

    if(assigned_level > max_level) {
        max_level = assigned_level;
        entry_points[0] = node.id;
        printf("changing max level to %ld\n", max_level);
    }
    auto vector = get_vector(nearest);
    float dist_nearest = disq(vector);
    delete[] vector;
    for(size_t l=max_level; l>assigned_level; l--) {
        greedy_insert_top(disq, l, nearest, dist_nearest, candidates);
    }

    for(size_t l=assigned_level; l>0; l--) {
        greedy_insert_top(disq, l, nearest, dist_nearest, candidates);
    }

    greedy_insert_base(disq, nearest, dist_nearest, candidates);

}



void CANDY::DynamicTuneHNSW::greedy_insert_top(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<Candidate>& candidates){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        size_t nb_neighbor_level = nb_neighbors(level);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            if(level > node->level) {
                break;
            }
            auto visiting = node->neighbors[level][i];
            if(visiting < 0) {
                break;
            }

            auto vector = get_vector(visiting);

            auto dist = disq(vector);
            printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
            delete[] vector;
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_insert_base(DAGNN::DistanceQueryer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<Candidate>& candidates){
    for(;;) {
        idx_t prev_nearest = nearest;
        auto node = linkLists[nearest];
        size_t nb_neighbor_level = nb_neighbors(0);
        for(size_t i=0; i<nb_neighbor_level; i++) {
            auto visiting = node->neighbors[0][i];
            if(visiting < 0) {
                break;
            }
            auto vector = get_vector(visiting);
            auto dist = disq(vector);
            printf("trying to %ld with dist = %.2f on level 0 \n", visiting, dist);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                printf("stepping to %ld with dist = %.2f on level% 0 \n", visiting, dist);
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::search(idx_t n, const float* x, idx_t annk, idx_t* results, float* distances) {

}