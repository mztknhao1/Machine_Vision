//
// Created by mztkn on 2019/11/24.
//

#ifndef SECOND_HOMWORK_MYUTILS_H
#define SECOND_HOMWORK_MYUTILS_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <sophus/se3.h>
#include <pangolin/pangolin.h>

using namespace std;
typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


class myutils{
private:
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> _pointcloud;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> _pointSphere;

public:
    myutils(const vector<cv::Point3f>& ballPoint3d, const vector<cv::Point3f>& spherePoint3d);
    ~myutils();
    void showPointCloud();

};


#endif //SECOND_HOMWORK_MYUTILS_H
