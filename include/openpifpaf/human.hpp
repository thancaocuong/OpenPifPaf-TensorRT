#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <cstring>
using namespace std;

struct Keypoint
{
    int x, y;
    int getX(){
        return x;
    }
    int getY(){
        return y;
    }
    const bool hasValue()
    {
        return !(x<=0 && y<=0);
    }
};

struct HumanInfo
{
    int number_of_keypoints;
    std::vector<Keypoint> bodyJoints;
    int idx;
    int get_num_keypoints(){
        return number_of_keypoints;
    }
    int getSize(){
        return 2*number_of_keypoints;
    }
    void toVector(std::vector<int> &kps, int& count_kps, float scale_x, float scale_y)
    {
        count_kps = 0;
        // only evaluate on 14 keypoints
        for (int i=0; i<number_of_keypoints-1; i++)
        {
            int visible = 0;
            kps.push_back(bodyJoints.at(i).getX() * scale_x);
            kps.push_back(bodyJoints.at(i).getY() * scale_y);
            if(bodyJoints.at(i).hasValue())
                {
                    visible = 2;
                    count_kps++;
                }
            kps.push_back(visible);
        }
    }
    int* toList(){
        int* keypoints_list = new int[2*number_of_keypoints];
        for(int i=0; i<number_of_keypoints; i++)
        {
            keypoints_list[2*i] = bodyJoints.at(i).getX();
            keypoints_list[2*i+1] = bodyJoints.at(i).getY();
        }
        return keypoints_list;
    }

};
struct InferResult
{
    std::vector<HumanInfo> humans;
    int getNumObjects(){
        return humans.size();
    }
    int get_poses_size()
    {
        int num_objects = getNumObjects();
        if(num_objects > 0)
        {
            int num_keypoints = humans[0].get_num_keypoints();
            return 2 * num_keypoints * num_objects;
        }
        return  0;
    }
    void toList(int* outputs_array, int &current_position, int poses_size)
    {
        int num_objects = getNumObjects();
        int current_index = 0;
        while(current_position < poses_size && current_index < num_objects)
        {
            memcpy(&outputs_array[current_position],
                   humans[current_index].toList(),
                   humans[current_index].getSize()*sizeof(int));
            current_position += humans[current_index].getSize();
            current_index += 1;
        }
    }
};

void draw_human(cv::Mat& image, const HumanInfo& human, vector<vector<int>> skeleton);
