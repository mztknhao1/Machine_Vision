#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/head.h"
#include "include/FitBall.h"
#include "include/myutils.h"


using namespace std;

int GlobalConfig(const string& extrinsics,const string& intrinsics,StereoParameter& StereoPara);

int main() {

    //初始化一些参数
    StereoParameter stereoPara;
    string leftImgPath = "../image/leftImage.txt";
    string rightImgPath = "../image/rightImage.txt";
    GlobalConfig("../extrinsics.yml","../intrinsics.yml",stereoPara);

    FitBall fitBall(leftImgPath,rightImgPath,stereoPara);
    fitBall.transform();

    float diameter = fitBall.fitDiameter();
    float r = diameter/2;
    float x0 = fitBall.centre.x;
    float y0 = fitBall.centre.y;
    float z0 = fitBall.centre.z;
    vector<Point3f> fitSphere;
    for(float x=-diameter/2;x<diameter/2;x+=0.05){
        for(float y=-diameter/2;y<diameter/2;y+=0.05){
            float tmp = r*r-x*x-y*y;
            if(tmp>=0){
                float z1 = sqrt(tmp) + z0;
                float z2 = -sqrt(tmp) + z0;
                Point3f pt1(x0+x,y0+y,z1);
                Point3f pt2(x0+x,y0+y,z2);
                fitSphere.push_back(pt1);
                fitSphere.push_back(pt2);
            }
        }
    }
    vector<Point3f> lightLine = fitBall.getBall3D();
    myutils ball(lightLine,fitSphere);
    ball.showPointCloud();
    cout << diameter << endl;
//
//    myutils ball(fitBall.getBall3D());
//    ball.showPointCloud();


    return 0;
}


//全局初始化，传入相机内外参数
int GlobalConfig(const string& extrinsics,const string& intrinsics,StereoParameter& StereoPara){

    FileStorage fs(extrinsics, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Global Config Failed!" << endl;
        return -1;
    }
    fs["R"] >> StereoPara.R;
    fs["T"] >> StereoPara.T;
    fs["Q"] >> StereoPara.Q;

    fs.open(intrinsics, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Filed to Global Config" << endl;
        return -1;
    }
    fs["M1"] >> StereoPara.M1;
    fs["D1"] >> StereoPara.D1;
    fs["M2"] >> StereoPara.M2;
    fs["D2"] >> StereoPara.D2;

    StereoPara.baseline = norm(StereoPara.T);

    float tz = StereoPara.T.at<float>(2, 0);
    float ty = StereoPara.T.at<float>(1, 0);
    float tx = StereoPara.T.at<float>(0, 0);
    Mat S = (Mat_<float>(3, 3) <<
                               0, -tz, ty,
            tz, 0, -tx,
            -ty, tx, 0);
    StereoPara.F = StereoPara.M2.t().inv() * S * StereoPara.R * StereoPara.M1.inv();
    StereoPara.F = StereoPara.F / StereoPara.F.at<float>(2, 2);
    return 0;
}

