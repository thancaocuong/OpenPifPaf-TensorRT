#ifndef _PIFPAF_NET_HPP_
#define _PIFPAF_NET_HPP_

#include <stdint.h>
#include <string>
#include <vector>

#include "openpifpaf/tensorrt_net.hpp"
#include "openpifpaf/trt_utils.hpp"
#include "openpifpaf/human.hpp"



class PifPafNet : public TensorRTNet
{
    public:
        PifPafNet(const uint batchSize, const NetworkInfo& networkInfo);
        void nvPreprocess(const std::vector<cv::Mat>& cvmats, int processingWidth, int processingHeight) override;
        std::vector<InferResult> decodeBatchOfFrames(const int imageH, const int imageW) override;
};
#endif // _PIFPAF_NET_HPP_