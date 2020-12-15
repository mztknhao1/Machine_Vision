//
// Created by mztkn on 2019/11/19.
//

#ifndef SECOND_HOMWORK_HEAD_H
#define SECOND_HOMWORK_HEAD_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <iostream>
#include <Eigen/Core>
#include <map>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.h>
#include <chrono>

#define EPSILON 1e-7



using namespace cv;
using namespace std;

typedef struct{
   cv::Mat M1;
   cv::Mat D1;
   cv::Mat M2;
   cv::Mat D2;
   cv::Mat R;
   cv::Mat T;
   cv::Mat Q;
   double baseline;
   cv::Mat F;
}StereoParameter;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> VecVector2f;
typedef vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VecVector3f;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

#endif //
