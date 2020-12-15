//
// Created by sunzhengmao on 2019/11/29.
//

#ifndef SECOND_HOMWORK_BA_G2O_H
#define SECOND_HOMWORK_BA_G2O_H

#include "head.h"

/// 顶点: 给定初始值, 以及更新方式
//-- vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //-- 重置, 莫非是初始点?
    virtual void setToOriginImpl() override
    {
        _estimate = Sophus::SE3();
    }

    /// left multiplication on SE3, 更新这个R和t
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

/// 边: 定义一个误差, 以及要如何求导
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override
    {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3 T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d); // 内参*外参*3D点
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>(); //重投影误差
    }

    virtual void linearizeOplus() override
    {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3 T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
                << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
                0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};





#endif //SECOND_HOMWORK_BA_G2O_H
