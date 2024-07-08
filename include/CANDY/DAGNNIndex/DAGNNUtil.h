//
// Created by rubato on 16/6/24.
//

#ifndef DAGNNUTIL_H
#define DAGNNUTIL_H
#define DAGNN_METRIC_IP 0
#define DAGNN_METRIC_L2 1
#include <faiss/IndexFlat.h>
#include <faiss/utils/Heap.h>

namespace CANDY::DAGNN{
    typedef int32_t dagnn_metric_t;
    struct DistanceQueryer {
        dagnn_metric_t dagnn_metric = DAGNN_METRIC_IP;
        float *query = nullptr;
        faiss::IndexFlat* storage = nullptr;
        const size_t vecDim;
        DistanceQueryer(const dagnn_metric_t metric, const size_t dim, faiss::IndexFlat* s):dagnn_metric(metric), vecDim(dim), storage(s){}
        DistanceQueryer(const dagnn_metric_t metric, const size_t dim):dagnn_metric(metric), vecDim(dim){}
        DistanceQueryer(const size_t dim):vecDim(dim){}
        void set_query(float* newq) {
            query = newq;
        }

        float operator()(const float *y) const{
            switch(dagnn_metric) {
                case(DAGNN_METRIC_L2):
                    return L2Sqr(query, y, vecDim);
                default:
                    return InnerProduct(query, y, vecDim);
            }
        }

        float distance(const float *x, const float * y){
            switch(dagnn_metric) {
                case(DAGNN_METRIC_L2):
                    return L2Sqr(x, y, vecDim);
                default:
                    return InnerProduct(x, y, vecDim);
            }
        }
        static float L2Sqr(const float* x, const float* y, const size_t vecDim) {
            float result = 0;
            for (size_t i = 0; i < vecDim; i++) {
                float t = *x - *y;
                x++;
                y++;
                result += t * t;
            }
            return result;
        }

        static float InnerProduct(const float* x, const float* y, const size_t vecDim) {
            float result = 0;
            for (size_t i = 0; i < vecDim; i++) {
                result += (*x) * (*y);
                x++;
                y++;
            }
            return -result;
        }
    };

    /// Table to store visited iteration number during search and insert; Now only
    /// update the number and store nothing
    class VisitedTable {
    public:
        std::vector<int> visited_;
        int visno;
        VisitedTable() : visno(1){};
        void set(int64_t idx) {
            visited_[idx] = visno;
        }
        VisitedTable(size_t length) {
            visno = 1;
            visited_.resize(length, 0);
        }
        bool get(int64_t idx) { return visited_[idx] == visno; }

        void advance() {
            if (visno > 250) {
                visno = 0;
                return;
            }
            visno++;
        }
        ~VisitedTable(){};
    };

    /// a tiny heap that is used during search
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<int64_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, int64_t> HC;
        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}
        void push(int64_t i, float v) {
            if (k == n) {
                if (v >= dis[0]) {
                    return;
                }
                if (ids[0] != -1) {
                    --nvalid;
                }
                faiss::heap_pop<HC>(k--, dis.data(), ids.data());
            }
            faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
            ++nvalid;
        };
        float max() const { return dis[0]; };
        int size() const { return nvalid; };
        void clear() {
            nvalid = 0;
            k = 0;
        };
        int64_t pop_min(float *vmin_out = nullptr) {
            assert(k > 0);
            // returns min. This is an O(n) operation
            int i = k - 1;
            while (i >= 0) {
                if (ids[i] != -1) {
                    break;
                }
                i--;
            }
            if (i == -1) {
                return -1;
            }
            int imin = i;
            float vmin = dis[i];
            i--;
            while (i >= 0) {
                if (ids[i] != -1 && dis[i] < vmin) {
                    vmin = dis[i];
                    imin = i;
                }
                i--;
            }
            if (vmin_out) {
                *vmin_out = vmin;
            }
            auto ret = ids[imin];
            ids[imin] = -1;
            --nvalid;
            return ret;
        };
        int count_below(float thresh) {
            int n_below = 0;
            for (int i = 0; i < k; i++) {
                if (dis[i] < thresh) {
                    n_below++;
                }
            }
            return n_below;
        }
    };
    /// combine two groups with avg and var

}
#endif //DAGNNUTIL_H
