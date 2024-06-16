//
// Created by rubato on 12/6/24.
//
#include<CANDY/DAGNNIndex/DAGNN.h>

void CANDY::DynamicTuneHNSW::add(idx_t n, const float* x) {
    // here params are already set or updated
    assert(n>0);
    assert(x);


    assign_levels(n);
    idx_t n0 = storage->ntotal;
    storage->add(n,x);



}

void CANDY::DynamicTuneHNSW::search(idx_t n, const float* x, idx_t annk, idx_t* results, float* distances) {

}