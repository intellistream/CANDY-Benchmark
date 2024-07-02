//
// Created by rubato on 12/6/24.
//
#include<CANDY/DAGNNIndex/DAGNN.h>
#include <faiss/utils/Heap.h>


void CANDY::DynamicTuneHNSW::updateGlobalState() {
    /// update value average
    auto prev_ntotal = graphStates.global_stat.ntotal;
    auto new_ntotal = graphStates.time_local_stat.ntotal;
    for(int64_t d=0; d<vecDim; d++) {
        auto prev = graphStates.global_stat.value_average[d];
        auto newv = graphStates.time_local_stat.value_average[d];
        graphStates.global_stat.value_average[d] = (prev_ntotal*prev + new_ntotal*newv)/(prev_ntotal + new_ntotal);
    }

    /// update degree stats
    {
        graphStates.global_stat.degree_sum = graphStates.time_local_stat.degree_sum_old + graphStates.time_local_stat.degree_sum_new;
        double degree_avg = graphStates.global_stat.degree_sum/(prev_ntotal+new_ntotal);

        double degree_avg_prev;
        if(prev_ntotal==0) {
            degree_avg_prev = 0;
        } else {
            degree_avg_prev = graphStates.time_local_stat.degree_sum_old/(prev_ntotal*1.0);
        }
        double degree_avg_new = graphStates.time_local_stat.degree_sum_new/(new_ntotal*1.0);
        // D = (n1*n2)/(n1+n2) * (avg1-avg2)^2;
        double D = (prev_ntotal*new_ntotal*1.0)/(prev_ntotal*1.0+new_ntotal*1.0) * ((degree_avg_prev-degree_avg_new) * (degree_avg_prev-degree_avg_new));
        // combine variance = 1/(n1+n2) * ((n1-1)* var1 + (n2-1) * var2 + D)
        printf("previous data degree var %lf\n", graphStates.global_stat.degree_variance);
        printf("UPdated previous data degree var %lf , new data degree var %lf\n", graphStates.time_local_stat.degree_variance_old, graphStates.time_local_stat.degree_variance_new);
        graphStates.global_stat.degree_variance = 1.0/(prev_ntotal*1.0 + new_ntotal*1.0) * (
                                                        (prev_ntotal*1.0-1.0)*graphStates.time_local_stat.degree_variance_old
                                                    +   (new_ntotal*1.0-1.0)*graphStates.time_local_stat.degree_variance_new
                                                    + D);

    }


    {

    }




    graphStates.global_stat.ntotal+=graphStates.time_local_stat.ntotal;
    graphStates.time_local_stat.reset();
    graphStates.time_local_stat.old_ntotal = graphStates.global_stat.ntotal;
    graphStates.time_local_stat.degree_sum_old = graphStates.global_stat.degree_sum;
    graphStates.time_local_stat.degree_variance_old = graphStates.global_stat.degree_variance;
    graphStates.global_stat.print();
}

void CANDY::DynamicTuneHNSW::add(idx_t n, float* x) {
    // here params are already set or updated
    assert(n>0);
    assert(x);

    assign_levels(n);
    idx_t n0 = storage->ntotal;
    storage->add(n,x);
    DAGNN::DistanceQueryer disq(vecDim);
    for(int64_t d=0; d<vecDim; d++) {
        //printf("%.2f, ", graphStates.global_stat.value_average[d]);
    }

    for(idx_t i=0; i<n; i++) {
        for(int64_t d=0; d<vecDim; d++) {
            auto m = (x[vecDim*i+d]-graphStates.global_stat.value_average[d])/(n0+i+1);
            graphStates.time_local_stat.value_average[d] += m;
        }
        graphStates.time_local_stat.ntotal+=1;
        /// Update degree measure, since insert a 0-point into data
        double old_degree_avg;
        if(graphStates.time_local_stat.ntotal==1) {
            old_degree_avg =0;
        } else {
            old_degree_avg = graphStates.time_local_stat.degree_sum_new*1.0/((graphStates.time_local_stat.ntotal-1)*1.0);
        }
        double new_degree_avg = graphStates.time_local_stat.degree_sum_new*1.0/(graphStates.time_local_stat.ntotal*1.0);
        double variance_without = graphStates.time_local_stat.degree_variance_new;
        double new_degree_variance = (graphStates.time_local_stat.ntotal-1)*1.0/(graphStates.time_local_stat.ntotal*1.0)*(variance_without + (0-old_degree_avg)*(0-old_degree_avg)/graphStates.time_local_stat.ntotal);
        graphStates.time_local_stat.degree_variance_new = new_degree_variance;

        disq.set_query(x+vecDim*i);

        auto node = linkLists[i+n0];

        DAGNN::VisitedTable vt(n0+n);
        greedy_insert(disq, *node, vt);

    }
    updateGlobalState();



}



void CANDY::DynamicTuneHNSW::greedy_insert(DAGNN::DistanceQueryer& disq, CANDY::DynamicTuneHNSW::Node& node, DAGNN::VisitedTable& vt){
    /// Greedy Phase
    idx_t nearest = -1;
    auto assigned_level = node.level;
    if(entry_points.empty()) {
        entry_points.push_back(node.id);
        max_level = assigned_level;
    }


    //printf("node %ld assigned level at %ld\n", node.id, node.level);
    if(assigned_level > max_level) {
        max_level = assigned_level;
        entry_points[0] = node.id;
        printf("changing max level to %ld\n", max_level);
    }
    nearest = entry_points[0];
    auto vector = get_vector(nearest);
    float dist_nearest = disq(vector);
    delete[] vector;
    if(assigned_level>0) {
        //printf("%ld with level=%ld\n", node.id, assigned_level);
    }
    for(size_t l=max_level; l>assigned_level; l--) {
        greedy_insert_top(disq, l, nearest, dist_nearest);
    }

    for(size_t l=assigned_level; l>0; l--) {
        std::priority_queue<CandidateCloser> candidates;
        //greedy_insert_upper(disq, l, nearest, dist_nearest, candidates);
        /// Candidate Phase ON TOP LEVELS and linking
        auto entry_this_level = CandidateCloser(dist_nearest, nearest);
        //printf("pushing %ld with %f to candidates on level %ld\n", nearest, dist_nearest, l);
        candidates.push(entry_this_level);
        link_from(disq, node.id, l, nearest, dist_nearest, candidates, vt);


    }
    std::priority_queue<CandidateCloser> candidates;
    greedy_insert_base(disq, nearest, dist_nearest, candidates);
    //printf("stepping to %ld with %f\n", nearest, dist_nearest);
    auto entry_base_level = CandidateCloser(dist_nearest, nearest);
    candidates.push(entry_base_level);
    link_from(disq, node.id, 0, nearest, dist_nearest, candidates, vt);
    /// Candidate phase on base level and linking





}



void CANDY::DynamicTuneHNSW::greedy_insert_top(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest){
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
            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
              //  printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
            delete[] vector;
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_insert_upper(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
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
            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
            delete[] vector;
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_insert_base(DAGNN::DistanceQueryer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
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
            //printf("trying to %ld with dist = %.2f on level 0 \n", visiting, dist);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level% 0 \n", visiting, dist);
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}
/// We need the nearest point to consult while linking
void CANDY::DynamicTuneHNSW::link_from(DAGNN::DistanceQueryer& disq, idx_t idx, size_t level, idx_t nearest, float dist_nearest, std::priority_queue<CandidateCloser>& candidates, DAGNN::VisitedTable& vt){
    /// Candidate phase
    priority_queue<CandidateFarther> selection;
    candidate_select(disq, level, selection, candidates, vt);
    /// Link Phase
    /// Prune this neighbor's candidate
    prune(disq, level, candidates);
    std::vector<idx_t> neighbors;
    neighbors.reserve(candidates.size());


    while(!candidates.empty()) {
        idx_t other_id = candidates.top().id;
        add_link(disq, idx, other_id, level);
        neighbors.push_back(other_id);
        candidates.pop();
    }

    for(auto nei: neighbors) {
        add_link(disq, nei, idx, level);

    }

}

void CANDY::DynamicTuneHNSW::candidate_select(DAGNN::DistanceQueryer& disq, size_t level,  std::priority_queue<CandidateFarther> selection, std::priority_queue<CandidateCloser>& candidates,  DAGNN::VisitedTable& vt){
    /// candidates might contain long-edge from greedy phase
    if(!candidates.empty()){
        selection.emplace(candidates.top().dist, candidates.top().id);
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
        auto curr_linkList = curr_node->neighbors[level];
        //auto curr_distList = curr_node->distances[level];
        for(size_t i=0; i<neighbor_length; i++){
            auto expansion = curr_linkList[i];
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

            //float dis=curr_distList[i];

            // TODO: MODIFY CANDIDATE SELECTION
            if(candidates.size() < dynamicParams.efConstruction || candidates.top().dist > dis){
                candidates.emplace(dis, expansion);
                selection.emplace(dis, expansion);
                if(candidates.size()>dynamicParams.efConstruction){
                    candidates.pop();
                }
            }
        }
    }
    vt.advance();

}

void CANDY::DynamicTuneHNSW::prune(DAGNN::DistanceQueryer& disq,size_t level,  std::priority_queue<CandidateCloser>& candidates){
    /// default version, selectively transfer candidates to output and then copy output to candidates
    int M = nb_neighbors(level);
    //TODO: modify
    if(candidates.size()<M) {
        return;
    }
    std::vector<CandidateCloser> output;
    int count = 0;
    std::priority_queue<CandidateCloser> candidates_reverse;
    while(!candidates.empty()) {
        candidates_reverse.emplace(-candidates.top().dist, candidates.top().id);
        candidates.pop();
    }
    while(!candidates_reverse.empty()){
        auto v1 = candidates_reverse.top();
        candidates_reverse.pop();
        float dist_v1_q = -v1.dist;

        bool rng_good=true;
        auto v1_vector = get_vector(v1.id);
        for(auto v2: output){
            auto v2_vector = get_vector(v2.id);

            float dist_v1_v2 = disq.distance(v1_vector, v2_vector);
            //printf("v1=%f v2=%f v1_v2=%f\n", v1.dist,v2.dist,dist_v1_v2);
            delete[] v2_vector;
            // from DiskANN Vanama
            if(dist_v1_v2 < (dist_v1_q /* dynamicParams.rng_alpha*/)){
                rng_good = false;
                break;
            }
        }
        delete[] v1_vector;

        if(rng_good){
            output.emplace_back(v1);
            if(output.size()>=M){
                // transfer output to candidate
                for(auto k : output) {
                    count++;
                    candidates.emplace(-k.dist, k.id);
                }
                //printf("avg distance %f with size %d\n", dis_sum/count, count);
                return;
            }

        }
    }
    for(auto k : output) {
        count++;
        candidates.emplace(-k.dist, k.id);
    }
    //printf("avg distance %f with size %d\n", dis_sum/count, count);
}


void breakpoint() {
    return;
}
int64_t CANDY::DynamicTuneHNSW::add_link(DAGNN::DistanceQueryer& disq, idx_t src, idx_t dest, size_t level) {
    if(src==dest) {
        return -1;
    }
    int nb_neighbors_level = nb_neighbors(level);
    auto src_linkList = &linkLists[src]->neighbors[level];
    auto src_dist = &linkLists[src]->distances[level];
    if((*src_linkList)[nb_neighbors_level-1]==-1) {
        size_t i=nb_neighbors_level;
        while(i>0) {
            if((*src_linkList)[i-1]!=-1) {
                break;
            }
            i--;
        }
        //printf("updating links for %ld: ", src);

        (*src_linkList)[i] = dest;
        int64_t prev_degree = i;
        int64_t current_degree = i+1;
        ///TODO: Update degree stats
        if(level==0 ) {
            if(src<graphStates.time_local_stat.old_ntotal) {

                auto n = graphStates.time_local_stat.old_ntotal;
                auto n2 = 1;
                double old_degree_avg = (graphStates.time_local_stat.degree_sum_old * 1.0)/(n*1.0);
                double new_degree_avg = current_degree;
                double old_entry = (prev_degree-old_degree_avg)*(prev_degree - old_degree_avg);
                double variance_without = (graphStates.time_local_stat.degree_variance_old * n - old_entry)/(n-1);

                double old_degree_avg_without = (graphStates.time_local_stat.degree_sum_old - prev_degree)/(n-1);
                auto D = (n-1)/n * (old_degree_avg_without - current_degree)*(old_degree_avg_without-current_degree);
                auto variance = 1.0/n*((n-1-1)*variance_without +D);




                graphStates.time_local_stat.degree_variance_old = variance;
                printf("variance = %lf\n", variance);
                graphStates.time_local_stat.degree_sum_old += (current_degree-prev_degree);

            } else {

                auto n = graphStates.time_local_stat.ntotal;
                if(n!=1) {
                    auto n2 = 1;
                    double old_degree_avg = (graphStates.time_local_stat.degree_sum_new * 1.0)/n*1.0;
                    double new_degree_avg = current_degree;
                    double old_entry = (prev_degree-old_degree_avg)*(prev_degree - old_degree_avg);
                    double variance_without = (graphStates.time_local_stat.degree_variance_new * n - old_entry)/(n-1);

                    double old_degree_avg_without = (graphStates.time_local_stat.degree_sum_new - prev_degree)/(n-1);
                    auto D = (n-1.0)/n * (old_degree_avg_without - current_degree)*(old_degree_avg_without-current_degree);
                    auto variance = 1.0/n*((n-1-1)*variance_without +D);




                    graphStates.time_local_stat.degree_variance_new = variance;
                }
                graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
            }
        }

        /// TODO: weighted edges
        auto src_vec = get_vector(src);
        auto dest_vec = get_vector(dest);

        (*src_dist)[i] = disq.distance(src_vec, dest_vec);
        //printf("from %ld to %ld\n", src, dest);
        //printf("%ld \n", dest);
        return current_degree;
    }

    std::priority_queue<CandidateCloser> final_neighbors;
    auto src_vector=get_vector(src);
    auto dest_vector = get_vector(dest);

    final_neighbors.emplace(disq.distance(src_vector, dest_vector), dest);
    delete[] dest_vector;
    for(size_t i=0; i<nb_neighbors_level; i++) {
        auto neighbor  = (*src_linkList)[i];
        /// TODO: WEIGHTED EDGES
        auto dist = (*src_dist)[i];

        //auto neighbor_vector = get_vector(neighbor);
        //auto dist = disq.distance(neighbor_vector, src_vector);
        //printf("neighbor %ld with dist %f       ", neighbor, dist);

        final_neighbors.emplace(dist, neighbor);
        //delete[] neighbor_vector;
    }
    //printf("before pruning %ld avg distance = %f with size %d\n", src, dis_sum/count, count);

    prune(disq, level, final_neighbors);

    size_t i=0;


    while(!final_neighbors.empty()) {
        //printf("%ld ", final_neighbors.top().id);
        (*src_linkList)[i++] = final_neighbors.top().id;
        //(*src_dist)[i++] = final_neighbors.top().dist;


        final_neighbors.pop();
    }
    //printf("after pruning %ld avg distance = %f with size %d\n\n", src, dis_sum/count, count);
    //printf("\n");
    int64_t current_degree = i;
    int64_t prev_degree = nb_neighbors(level);
    if(level==0 ) {
        if(src<graphStates.time_local_stat.old_ntotal) {

            auto n = graphStates.time_local_stat.old_ntotal;
            auto n2 = 1;
            double old_degree_avg = (graphStates.time_local_stat.degree_sum_old * 1.0)/n*1.0;
            double new_degree_avg = current_degree;
            double old_entry = (prev_degree-old_degree_avg)*(prev_degree - old_degree_avg);
            double variance_without = (graphStates.time_local_stat.degree_variance_old * n - old_entry)/(n-1);

            double old_degree_avg_without = (graphStates.time_local_stat.degree_sum_old - prev_degree)/(n-1);
            auto D = (n-1)/n * (old_degree_avg_without - current_degree)*(old_degree_avg_without-current_degree);
            auto variance = 1.0/n*((n-1-1)*variance_without +D);




            graphStates.time_local_stat.degree_variance_old = variance;
            printf("variance = %lf\n", variance);
            graphStates.time_local_stat.degree_sum_old+=(current_degree-prev_degree);

        } else {

            auto n = graphStates.time_local_stat.ntotal;
            if(n!=1) {
                auto n2 = 1;
                double old_degree_avg = (graphStates.time_local_stat.degree_sum_new * 1.0)/n;
                double new_degree_avg = current_degree;
                double old_entry = (prev_degree-old_degree_avg)*(prev_degree - old_degree_avg);
                double variance_without = (graphStates.time_local_stat.degree_variance_new * n - old_entry)/(n-1);

                double old_degree_avg_without = (graphStates.time_local_stat.degree_sum_new - prev_degree)/(n-1);
                auto D = (n-1.0)/n * (old_degree_avg_without - current_degree)*(old_degree_avg_without-current_degree);
                auto variance = 1.0/n*((n-1-1)*variance_without +D);




                graphStates.time_local_stat.degree_variance_new = variance;
            }

            graphStates.time_local_stat.degree_sum_new+=(current_degree-prev_degree);
        }

    }
    while(i<nb_neighbors_level) {
        (*src_linkList)[i++]=-1;
    }
    return current_degree;
}

void CANDY::DynamicTuneHNSW::search(DAGNN::DistanceQueryer &disq, idx_t annk, idx_t* results, float* distances, DAGNN::VisitedTable& vt) {
    if(entry_points.empty()) {
        printf(" EMPTY INDEX!\n");
        return;
    }

    auto nearest = entry_points[0];
    auto nearest_vec = get_vector(nearest);
    float d_nearest = disq(nearest_vec);
    delete[] nearest_vec;

    for(size_t l=max_level; l>=1; l--) {
        std::priority_queue<CandidateCloser> candidates_greedy;
        greedy_search_upper(disq, l, nearest, d_nearest, candidates_greedy);

    }
    std::priority_queue<CandidateCloser> candidates_greedy;
    greedy_search_base(disq,nearest,d_nearest,candidates_greedy);
    int64_t efSearch = annk>dynamicParams.efSearch? annk:dynamicParams.efSearch;
    DAGNN::MinimaxHeap candidates(efSearch);

    candidates.push(nearest, d_nearest);

    candidate_search(disq, 0, annk, results, distances,candidates,vt);
    vt.advance();

}

void CANDY::DynamicTuneHNSW::greedy_search_upper(DAGNN::DistanceQueryer& disq, size_t level, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
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
            //printf("trying to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level%ld\n", visiting, dist, level);
            }
            delete[] vector;
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

void CANDY::DynamicTuneHNSW::greedy_search_base(DAGNN::DistanceQueryer& disq, idx_t& nearest, float& dist_nearest, std::priority_queue<CandidateCloser>& candidates){
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
            //printf("trying to %ld with dist = %.2f on level 0 \n", visiting, dist);
            if(dist < dist_nearest) {
                nearest = visiting;
                dist_nearest = dist;
                //printf("stepping to %ld with dist = %.2f on level 0 \n", visiting, dist);
            }
        }
        // out condition
        if(nearest == prev_nearest) {
            return;
        }
    }
}

int CANDY::DynamicTuneHNSW::candidate_search(DAGNN::DistanceQueryer& disq, size_t level, idx_t annk, idx_t* results, float* distances,DAGNN::MinimaxHeap& candidates, DAGNN::VisitedTable& vt) {
    idx_t nres = 0;

    int nstep = 0;
    for(int i=0; i<candidates.size(); i++) {
        idx_t v1=candidates.ids[i];
        float d = candidates.dis[i];
        if(nres<annk) {
            faiss::maxheap_push(++nres, distances, results, d, v1);
        } else if(d<distances[0]) {
            faiss::maxheap_replace_top(nres, distances, results, d,v1);
        }
        vt.set(v1);
    }

    while(candidates.size()>0) {
        float d0 = 0;
        idx_t v0 = candidates.pop_min(&d0);
        //printf("popping v0=%ld\n", v0);
        auto nb_neighbors_level = nb_neighbors(level);
        auto v0_linkList = linkLists[v0]->neighbors[level];
        auto v0_vector=get_vector(v0);
        auto v0_distList = linkLists[v0]->distances[level];
        for(size_t j=0; j<nb_neighbors_level; j++) {
            idx_t v1 = v0_linkList[j];
            //printf("stepping to %ld\n", v1);
            if(v1<0) {
                break;
            }

            if(vt.get(v1)) {
                continue;
            }

            vt.set(v1);
            auto v1_vector = get_vector(v1);
            //float d1 = v0_distList[j];
            float d1=disq(v1_vector);
            float d1_d0 = disq.distance(v1_vector, v0_vector);
            float d0=disq(v0_vector);
            //printf("v1=%ld d0=%f d1=%f d1_d0=%f\n", v1, d0, d1, d1_d0);

            delete[] v1_vector;
            if(nres<annk) {
                faiss::maxheap_push(++nres, distances, results, d1, v1);
            } else if(d1<distances[0]) {
                faiss::maxheap_replace_top(nres, distances, results, d1, v1);
            }
            candidates.push(v1,d1);

        }
        delete v0_vector;
        nstep++;
        if(nstep>dynamicParams.efSearch && nstep>annk) {
            break;
        }

    }
    //printf("nstep=%d\n", nstep);
    return nres;


}