#include "openpifpaf/human.hpp"
#include "openpifpaf/color.hpp"


void draw_human(cv::Mat& img, const HumanInfo& human, std::vector<vector<int>> skeleton)
{
    float n = 1, s = 0, w = 1, e = 0;
    int num_parts = skeleton.size();
    for (auto p: human.bodyJoints)
    {
        if (p.hasValue()) {
            n = std::min(n, static_cast<float>(p.y)/static_cast<float>(img.rows));
            s = std::max(s, static_cast<float>(p.y)/static_cast<float>(img.rows));
            w = std::min(w, static_cast<float>(p.x)/static_cast<float>(img.cols));
            e = std::max(e, static_cast<float>(p.x)/static_cast<float>(img.cols));
        }

    }

    const int thickness = std::max(1, static_cast<int>(std::sqrt((e - w) * (s - n) * img.size().area())) / 32);
    for (int pair_id = 0; pair_id < num_parts; pair_id++) {

        int src_part_idx = skeleton[pair_id][0];
        int dst_part_idx = skeleton[pair_id][1];
        auto p1 = human.bodyJoints[src_part_idx];
        auto p2 = human.bodyJoints[dst_part_idx];
        const auto color = openpifpaf::coco_colors[pair_id];
        if (p1.hasValue())
            cv::circle(img, cv::Point(p1.x, p1.y), thickness, color, cv::FILLED);
        if (p2.hasValue())
            cv::circle(img, cv::Point(p2.x, p2.y), thickness, color, cv::FILLED);
        if (p1.hasValue() && p2.hasValue())
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color, thickness);
    }
}
