//
// Created by rubato on 16/6/24.
//

#ifndef DAGNNUTIL_H
#define DAGNNUTIL_H
#define DAGNN_METRIC_IP 0
#define DAGNN_METRIC_L2 1
namespace CANDY::DAGNN{
    typedef int32_t dagnn_metric_t;
    struct DistanceQueryer {
        dagnn_metric_t dagnn_metric = DAGNN_METRIC_IP;
        float *query = nullptr;
        const size_t vecDim;
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


}
#endif //DAGNNUTIL_H
