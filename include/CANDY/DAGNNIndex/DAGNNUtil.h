//
// Created by rubato on 16/6/24.
//

#ifndef DAGNNUTIL_H
#define DAGNNUTIL_H
#define DAGNN_METRIC_IP 0
#define DAGNN_METRIC_L2 1
#include <faiss/IndexFlat.h>
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
        std::vector<uint8_t> visited_;
        int visno;
        VisitedTable() : visno(1){};
        void set(int64_t idx) {
            visited_[visno] = visno;
            return;
        }
        bool get(int64_t idx) { return visited_[idx] == visno; }

        void advance() {
            if (visno > 250) {
                visno = 0;
                return;
            }
            visno++;
        }
    };

}
#endif //DAGNNUTIL_H
