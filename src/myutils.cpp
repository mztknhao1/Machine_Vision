//
// Created by mztkn on 2019/11/24.
//

#include "../include/myutils.h"

void myutils::showPointCloud() {
    if (this->_pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 2048, 2448);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(5);
        glBegin(GL_POINTS);
        for (auto &p: this->_pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        for (auto &p: this->_pointSphere) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

myutils::~myutils() {

}

myutils::myutils(const vector<cv::Point3f>& ballPoint3d,const vector<cv::Point3f>& shperePoint3d) {
    for(auto i:ballPoint3d){
        Eigen::Matrix<double,6,1> v;

        int r = 255;
        int g = 0;
        int b = 0;

        v << i.x,i.y,i.z,r,g,b;
        this->_pointcloud.push_back(v);
    }
    for(auto i:shperePoint3d){
        Eigen::Matrix<double,6,1> v;

        int r = 200;
        int g = 200;
        int b = 200;

        v << i.x,i.y,i.z,r,g,b;
        this->_pointSphere.push_back(v);
    }
}


//void showPointCloud(vector<) {
//    if (this->_pointcloud.empty()) {
//        cerr << "Point cloud is empty!" << endl;
//        return;
//    }
//
//    pangolin::CreateWindowAndBind("Point Cloud Viewer", 2048, 2448);
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//    pangolin::OpenGlRenderState s_cam(
//            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
//            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
//    );
//
//    pangolin::View &d_cam = pangolin::CreateDisplay()
//            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
//            .SetHandler(new pangolin::Handler3D(s_cam));
//
//    while (pangolin::ShouldQuit() == false) {
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//        d_cam.Activate(s_cam);
//        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//
//        glPointSize(5);
//        glBegin(GL_POINTS);
//        for (auto &p: this->_pointcloud) {
//            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
//            glVertex3d(p[0], p[1], p[2]);
//        }
//        glEnd();
//        pangolin::FinishFrame();
//        usleep(5000);   // sleep 5 ms
//    }
//    return;
//}