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

        DAGNN::VisitedTable vt(n0+n);
        greedy_insert(disq, *node, vt);


    }

    // for(int64_t d=0; d<vecDim; d++) {
    //     printf("%.2f, ", graphStates.global_stat.value_average[d]);
    // }
    // printf("\n");



}



void CANDY::DynamicTuneHNSW::greedy_insert(DAGNN::DistanceQueryer& disq, CANDY::DynamicTuneHNSW::Node& node, DAGNN::VisitedTable& vt){
    /// Greedy Phase
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
        std::priority_queue<Candidate> candidates;
        greedy_insert_top(disq, l, nearest, dist_nearest, candidates);
        /// Candidate Phase ON TOP LEVELS and linking
        auto entry_this_level = Candidate(dist_nearest, nearest);
        candidates.push(entry_this_level);
        link_from(disq, node.id, l, nearest, dist_nearest, candidates, vt);


    }
    std::priority_queue<Candidate> candidates;
    greedy_insert_base(disq, nearest, dist_nearest, candidates);
    auto entry_base_level = Candidate(dist_nearest, nearest);
    candidates.push(entry_base_level);
    link_from(disq, node.id, 0, nearest, dist_nearest, candidates, vt);
    /// Candidate phase on base level and linking





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
/// We need the nearest point to consult while linking
void CANDY::DynamicTuneHNSW::link_from(DAGNN::DistanceQueryer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<Candidate>& candidates, DAGNN::VisitedTable& vt){
    /// Candidate phase
    candidate_select
    /// Link Phase
}

void CANDY::DynamicTuneHNSW::candidate_select(DAGNN::DistanceQueryer& disq, size_t level, std::priority_queue<Candidate>& candidates,  DAGNN::VisitedTable& vt){
    std::priority_queue<Candidate> selection;
    /// candidates might contain long-edge from greedy phase
    if(!candidates.empty()){
        selection.emplace(candidates.top());
    }
    vt.set(candidates.top().id);
    while(!selection.empty()){
        auto curr = selection.top();
        if(curr.dist > candidates.top().dist){
            break;
        }
        int curr_id = curr.id;
        selection.pop();
        auto curr_node = linkLists[curr_id];
        size_t neighbor_length = nb_neighbors(level);
        for(size_t i=0; i<neighbor_length; i++){
            auto expansion = curr->neighbors[i];
            if(expansion<0){
                break;
            }
            if(vt.get(expansion)){
                continue;
            }
            vt.set(expansion);

            auto exp_vector = get_vector(expansion);
            float dis = disq(exp_vector);
            delete[] exp_vector;

            auto exp_node = Candidate(dis, expansion);
            // TODO: MODIFY CANDIDATE SELECTION
            if(candidates.size() < efConstruction || candidates.top().dist > dis){
                candidates.push(dis, expansion);
                selection.emplace(dis, expansion);
                if(candidates.size()>efConstruction){
                    candidates.pop();
                }
            }
        }
    }
    vt.advance();

}

void CANDY::DynamicTuneHNSW::prune(size_t level, idx_t entry, float distance_entry,  idx_t nearest, float dist_nearest, std::priority_queue<Candidate>& candidates){
    /// default version, selectively transfer candidates to output and then copy output to candidates
    int M = nb_neighbors(level);
    std::priority_queue<Candidate> output;
    while(candidates.size()>0){
        auto v1 = candidates.top();
        candidates.pop();
        float dist_v1_q = v1.dist;

        bool rng_good=true;
        auto v1_vector = get_vector(v1.id);
        for(auto v2: output){
            auto v2_vector = get_vector(v2.id);

            float dist_v1_v2 = disq.distance(v1.vector, v2.vector);
            delete[] v2_vector;
            // from DiskANN Vanama
            if(dist_v1_v2 < (dist_v1_q / dynamicParams.rng_alpha)){
                rng_good = false;
                break;
            }
        }
        delete[] v1_vector;

        if(rng_good){
            output.push(v1);
            if(output.size()>=M){
                // transfer output to candidate
                while(!candidates.empty()){
                    candidates.pop()
                }
                while(!output.empty()){
                    candidates.push(output.top());
                    output.pop();
                }
                return;
            }

        }
    }
}

void CANDY::DynamicTuneHNSW::search(idx_t n, const float* x, idx_t annk, idx_t* results, float* distances) {

}