//
// Created by mztkn on 2019/11/19.
//

#ifndef SECOND_HOMWORK_FITBALL_H
#define SECOND_HOMWORK_FITBALL_H

#include"head.h"
#include "BA_g2o.h"
#include "../include/FastLightExtraction.h"

using namespace cv;

class generateCloud {
private:
    Mat _imgLeft;
    Mat _imgRight;
    vector<Point2f> _leftPoint;
    vector<Point2f> _rightPoint;
    vector<Point3f> _linePoint3D;

    StereoParameter _stereoPara;
    int threshold_line = 210;
    FastLightExtraction* helpMeCalHessian;

    Mat matPointLeft;
    Mat matPointRight;
    vector<Point2f> _leftPointMatched;


public:
    generateCloud(Mat& imgLeft, Mat& imgRight, StereoParameter& stereoParameter);
    ~generateCloud() = default;

    inline vector<Point3f> getPoint3D(){
        return this->_linePoint3D;
    };


    bool fMatchPointfromLeft2Right(const Point2f point_, const vector<Point2f> PointVector, Point2f& result, Mat& img);
    bool fMatchPointfromRight2Left(const Point2f point_, const vector<Point2f> PointVector, Point2f& result, Mat& img);
    void matchLine(int numPoint, Mat& img);//第一步：先将两个图片的光条进行匹配，放到两个vector中
    bool generate3D(int numPoint);//第二步：得到这次观测的光条的3D坐标


};

class FitBall {
private:
    vector<string> _leftImage;
    vector<string> _rightImage;
    vector<Point3f> _ballCloud;
    vector<vector<Point3f> > _partCloud;
    vector<Mat>    _Rs;//依次保存每个相机的位姿
    vector<Mat> _ts;
    StereoParameter _stereoParameter;
    float diameter = 0.0;
    // Cauchy 2.3849, Huber 1.345
    double c = 1.345;

    //维护第一次拍摄时观测到的关键点，用来做后面的匹配工作，由initialKeyPoint来完成这项工作
    vector<KeyPoint> _firstKeyLeft;
    vector<KeyPoint> _firstKeyRight;
    Mat _firstKeyLeftDescriptor; //存储第一个位置处的左图像的描述子,以防后面还要计算
    vector<DMatch> _globalMatches;
    void initialKeyPoint(Mat& firstLeft, Mat& firstRight);

    //私有函数，用SURF方法匹配两张图像，传出关键点和匹配向量vector<DMatch> matches
    void matchImage(Mat& img1, Mat& img2, vector<KeyPoint>& kps1, vector<KeyPoint>& kps2, vector<DMatch>& matches,int flag, int countImg);

    //从2D算3D（仅第一次拍摄的使用)
    static bool cam2dTo3d(KeyPoint& kp1, KeyPoint& kp2, StereoParameter& _stereoPara, Point3f& outPoint3d);


public:

    //传左图像和右图像的txt
    FitBall(const string& leftImage, const string& rightImage, StereoParameter  stereoParameter);
    ~FitBall() = default;

    Point3f centre;


    //通过左图像计算R和t
    void computeRt(Mat& leftOne, Mat& leftTwo, cv::Mat& R, cv::Mat& t, int countImg);

    //把two的所有点云通过Rs和ts转到第一张左图像坐标系下
    void transform();

    void bundleAdjustmentGaussNewton(
            const VecVector3d &points_3d,
            const VecVector2d &points_2d,
            const Mat &K,
            Sophus::SE3 &pose,
            bool useKernel = false);

    void bundleAdjustmentG2O(
            const VecVector3d &points_3d,
            const VecVector2d &points_2d,
            const Mat &K,
            Sophus::SE3 &pose);

    //拟合直径
    float fitDiameter();

    //辅助函数，返回3D点
    inline vector<Point3f> getBall3D(){
        int i = this->_ballCloud.size();
        return this->_ballCloud;
    };

};




#endif
