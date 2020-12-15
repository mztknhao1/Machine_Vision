//
// Created by gsjzy on 2019/11/19.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "../include/FitBall.h"
#include "fstream"
#include <utility>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <algorithm>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include "../include/myutils.h"

#define EPSILON 1e-7




FitBall::FitBall(const string& leftImage, const string& rightImage, StereoParameter  stereoParameter) :_stereoParameter(std::move(stereoParameter))
{
    fstream in(leftImage);
    string line;
    //先读取左边图片
    while(getline(in,line))
        this->_leftImage.push_back(line);
    in.close();

    //再读取右边图片
    in.open(rightImage);
    while(getline(in,line))
        this->_rightImage.push_back(line);
    in.close();

    Mat left = cv::imread(_leftImage[0]);
    Mat right = cv::imread(_rightImage[0]);

    //初始化关键点信息
    initialKeyPoint(left,right);
}

void FitBall::transform()
{
    //计算每幅图像获得的光条3D点坐标
    for (int i = 0; i < this->_leftImage.size(); ++i)
    {
        Mat left = cv::imread(_leftImage[i],0);
        if (!left.data)
            cerr << "第" << i << "个位置处的左图片读取出错" << endl;
        Mat right = cv::imread(_rightImage[i],0);
        if (!right.data)
            cerr << "第" << i << "个位置处的右图片读取出错" << endl;

        // 计算第i个位置处的点云坐标，并存入partCloud里去
        auto generate_cloud = new generateCloud(left, right,this->_stereoParameter);
        if (!generate_cloud->generate3D(10))
            cerr << "!!! the " << i << "th wrong !!!" << endl;
        vector<Point3f> tmp = generate_cloud->getPoint3D();
        this->_partCloud.push_back(generate_cloud->getPoint3D());
    }

    //计算R和T
    for (int i = 0; i < this->_leftImage.size()-1; ++i)
    {
        Mat left1 = cv::imread(_leftImage[0],0);
        Mat left2 = cv::imread(_leftImage[i+1],0);
        Mat R;
        Mat t;
        this->computeRt(left1, left2, R, t, i+1);

        this->_Rs.push_back(R);
        this->_ts.push_back(t);
    }

    //转化到第一张图片的坐标系下,然后将所有点云保存到this->ballCloud中
    //第一次的就不用转换了，直接加在ballCloud后面
    this->_ballCloud.insert(_ballCloud.end(),this->_partCloud[0].begin(),this->_partCloud[0].end());
    for(int i=0;i<this->_leftImage.size()-1;++i)
    {
        vector<Point3f> partPointVec = this->_partCloud[i+1];
        Mat t(this->_ts[i]);
        Mat R(this->_Rs[i]);
        for(auto j:partPointVec)
        {
            Mat local3d(j);
            Mat global3d = local3d + t;
            Point3f pt3f;
            pt3f.x = global3d.at<float>(0,0);
            pt3f.y = global3d.at<float>(1,0);
            pt3f.z = global3d.at<float>(2,0);
            this->_ballCloud.push_back(pt3f);
        }
    }
}

/**
 * 这里面都是double,我想办法把他们都搞成double的试一下
 * @param points_3d
 * @param points_2d
 * @param K
 * @param pose
 */
void FitBall::bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3 &pose,
        bool useKernel)
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> H_kernel = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        Vector6d b_kernel = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d f = points_2d[i] - proj;

            /// kernel
            double sk = f.transpose() * f;
            double rou = c * c * log(1 + sk / (c * c));
            double rou_primer = 1 / (1 + sk / (c * c));
            double rou_primer2 = -1 * rou_primer * rou_primer / (c * c);
            Eigen::Matrix2d W = rou_primer * Eigen::Matrix2d::Identity() + 2 * rou_primer2 * f * f.transpose();

            cost += f.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * f;

            /// kernel
            H_kernel += J.transpose() * W * J;
            b_kernel += - rou_primer * J.transpose() * f;
        }

        Vector6d dx;
        if(useKernel)
            dx = H_kernel.ldlt().solve(b_kernel);
        else
            dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

//    cout << "pose by g-n: \n" << pose.matrix() << endl;
}

/**
 * 用g2o看看能不能把匹配错误的点给拉回来
 * @param points_3d
 * @param points_2d
 * @param K
 * @param pose
 */
void FitBall::bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3 &pose)
{
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // vertex
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i)
    {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
//        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
//    cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}

void FitBall::computeRt(Mat &leftOne, Mat &leftTwo, cv::Mat &R, cv::Mat& t, int countImg)
{
    vector<KeyPoint> kps1 = this->_firstKeyLeft;
    vector<KeyPoint> kps2;
    vector<DMatch>   localmatches;      //这次匹配上的点
    vector<DMatch>   globalmatches;     //第一幅图像匹配上的点，因为第一幅图像计算了点的3D坐标，可以利用这个信息来使用PnP的方法

    //获得这次的匹配点
    matchImage(leftOne,leftTwo,kps1,kps2,localmatches, 1, countImg);
    globalmatches = this->_globalMatches;
    //寻找两次匹配拥有的共同特征点
    //使用PnP方法求解R,t矩阵
    vector<Point2f> points2d;  //第二幅图像的2d点
    vector<Point3f> points3d;   //对应的3D点

    VecVector2d Points2d_double;
    VecVector3d Points3d_double;

    Mat imgLeftone(leftOne);
    Mat imgLefttwo(leftTwo);

    vector<DMatch> testMatches;
    for(auto i:globalmatches)
    {
        for(auto j:localmatches)
        {
            //第一次的左图，第n次的左图，匹配上的点的索引相等
            if (i.queryIdx == j.queryIdx)
            {

                //由于实验台上的点可能导致无匹配,现在不对实验台上的点进行PnP计算
                int error_ = kps1[i.queryIdx].pt.y;
//                if(error_ == 1921)
//                    break;
//                cout<<error_<<endl;

                Point2f p2d(kps2[j.trainIdx].pt);
                Point3f p3d;

                testMatches.push_back(j);
                //通过第一幅图的关键点和保存好的对应点，计算该点的3D坐标
                if (!cam2dTo3d(this->_firstKeyLeft[i.queryIdx], this->_firstKeyRight[i.trainIdx],
                               this->_stereoParameter, p3d))
                {
                    break;
                }
                points2d.push_back(p2d);
                points3d.push_back(p3d);

                /// 搞成double
                Eigen::Vector2d p2d_double;
                p2d_double << double(p2d.x), double(p2d.y);
                Points2d_double.push_back(p2d_double);

                Eigen::Vector3d p3d_double;
                p3d_double << double(p3d.x), double(p3d.y), double(p3d.z);
                Points3d_double.push_back(p3d_double);

                // 2D 坐标
                float p2d_x = p2d.x;
                float p2d_y = p2d.y;
                // 3D 坐标
                float p3d_x = p3d.x;
                float p3d_y = p3d.y;
                float p3d_z = p3d.z;


                circle(imgLeftone, kps1[j.queryIdx].pt, 5, Scalar(255, 0, 0), 3);
                circle(imgLefttwo, kps2[j.trainIdx].pt, 5, Scalar(255, 0, 0), 3);


                string name_left = "testleft"+to_string(countImg)+".png";
                string name_right = "testright"+to_string(countImg)+".png";
                imwrite(name_left, imgLeftone);
                imwrite(name_right, imgLefttwo);
                int a = 0;
            }
        }
    }

    Mat testMatchesImg;
    drawMatches(leftOne,kps1,leftTwo,kps2,testMatches,testMatchesImg);
    string name_file = "testMatches1_" + to_string(countImg) + ".png";
    imwrite(name_file,testMatchesImg);

    //最后根据这些求解两个位置的旋转和平移矢量
    Mat rvec,tvec;
    //solvePnP(points3d,points2d,this->_stereoParameter.M1,this->_stereoParameter.D1,rvec,tvec,false,SOLVEPNP_ITERATIVE);
    solvePnPRansac(points3d, points2d, this->_stereoParameter.M1, this->_stereoParameter.D1, rvec, tvec, false);

    //使用罗德里格斯公式转换旋转向量为旋转矩阵
    Mat R_;
    Rodrigues(rvec,R_);
    //这里统一化成32F类型，避免后续计算出错
    tvec.convertTo(t,CV_32F);
    R_.convertTo(R,CV_32F);
    cout<<"===================================================================="<<endl;
    cout<<"第1到第"<<countImg+1<<"个位置优化前的位姿: "<<endl;
    cout<<"R: "<<R<<endl;
    cout<<"t: "<<t.t()<<endl;

    //-- 现在进行优化
    Mat K_double, R_mat, t_mat;
    this->_stereoParameter.M1.convertTo(K_double, CV_64F);
    Sophus::SE3 pose;
//    bundleAdjustmentG2O(Points3d_double, Points2d_double, K_double, pose);
    bundleAdjustmentGaussNewton(Points3d_double, Points2d_double, K_double, pose, false);
    cout<<"优化后的转换矩阵:"<<endl;
    Eigen::Matrix3d R_GN = pose.matrix().block(0,0,3,3);
    Eigen::Vector3d t_GN = pose.matrix().block(0,3,3,1);
    eigen2cv(R_GN, R_mat);
    eigen2cv(t_GN, t_mat);
    t_mat.convertTo(t,CV_32F);
    R_mat.convertTo(R,CV_32F);
    cout<<"R: "<<R<<endl;
    cout<<"t: "<<t.t()<<endl;
}

void FitBall::initialKeyPoint(Mat &firstLeft, Mat &firstRight)
{
    //使用SURF特征 获取图像特征点
    std::vector<cv::KeyPoint> keyPointsL;
    std::vector<cv::KeyPoint> keyPointsR;

    //-- 特征点与描述子匹配
    matchImage(firstLeft,firstRight,keyPointsL,keyPointsR,this->_globalMatches,0, 1);
    this->_firstKeyLeft = keyPointsL;
    this->_firstKeyRight = keyPointsR;

    //-- LK光流法
//    vector<KeyPoint> kp1;
//    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
//    detector->detect(firstLeft, kp1);
//
//    vector<Point2f> pt1, pt2;
//    for (auto &kp: kp1) pt1.push_back(kp.pt);
//    vector<uchar> status;
//    vector<float> error;
//    cv::calcOpticalFlowPyrLK(firstLeft, firstRight, pt1, pt2, status, error);
//
//    Mat img2_CV(firstRight);
//    for (int i = 0; i < pt2.size(); i++) {
//        if (status[i]) {
//            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
//        }
//    }
//    imwrite("tracked_opencv.png", img2_CV);
}

//flag==0代表第一次初始化的时候使用这个函数，计算出kps1，之后kps1是传入的参数
void FitBall::matchImage(Mat &img1, Mat &img2, vector<KeyPoint>& kps1, vector<KeyPoint>& kps2, vector<DMatch>& matches, int flag, int countImg)
{
    //使用SIFT特征 获取图像特征点
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    //单独提取出两个图像中的特征点
    Mat descL, descR;
    if(flag == 0)
    {
        sift->detectAndCompute(img1, noArray(), kps1, descL);
        this->_firstKeyLeftDescriptor = descL;
    }
    else
        descL = this->_firstKeyLeftDescriptor;


/*************************************************************************************************************************/
//    if(flag == 1)
//    {
//        vector<Point2f> pt1, pt2;
//        for (auto &kp: kps1) pt1.push_back(kp.pt);
//        vector<uchar> status;
//        vector<float> error;
//        cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
//
//        Mat img2_CV(img2);
//        for (int i = 0; i < pt2.size(); i++) {
//            if (status[i]) {
//                cv::circle(img2_CV, pt2[i], 2, cv::Scalar(255, 255, 255), 2);
//                cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(255, 255, 255));
//            }
//        }
//        string imgName = "tracked_opencv" + to_string(countImg) + ".png";
//        imwrite(imgName, img2_CV);
//        int m = 0;
//    }

/*************************************************************************************************************************/
    sift->detectAndCompute(img2, noArray(), kps2, descR);

    vector<vector<DMatch>> knn_matches;
    BFMatcher bfMatcher(NORM_L2);
    bfMatcher.knnMatch(descL,descR,knn_matches,2);

    float max_dist = 0;
    for(auto& knn_matche:knn_matches)
    {
        if(max_dist<knn_matche[0].distance)
        {
            max_dist = knn_matche[0].distance;
        }
    }

    for (auto & knn_matche : knn_matches)
    {
        if (knn_matche[0].distance > 0.8*knn_matche[1].distance ) continue;
        if(knn_matche[0].distance > 0.25*max_dist) continue;
        matches.push_back(knn_matche[0]);
    }

    vector<Point2f> queryOne, trainOne;
    for (int i = 0; i < matches.size(); ++i)
    {
        queryOne.push_back(kps1[matches[i].queryIdx].pt);
        trainOne.push_back(kps2[matches[i].trainIdx].pt);
    }

    if(flag==0)
    {
        //画出良好匹配结果
        Mat ShowGoodMatches;
        drawMatches(img1,kps1,img2,kps2,matches,ShowGoodMatches);
        imwrite("good_matches.png", ShowGoodMatches);
    }
    else
    {
        //画出良好匹配结果///画这个没用,一会删掉
        Mat ShowGoodMatches;
        drawMatches(img1,kps1,img2,kps2,matches,ShowGoodMatches);
        string name = "1_" + to_string(countImg) + "matches.png";
        imwrite(name, ShowGoodMatches);
    }

}

bool FitBall::cam2dTo3d(KeyPoint &kp1, KeyPoint &kp2, StereoParameter& _stereoPara, Point3f &outPoint3d)
{
    //一些计算需要的参数,就是第一个相机的内参和两个相机之间的基线
    //开始计算3D点坐标
    Eigen::Matrix3f R_eigen, A_left, A_right;
    cv2eigen(_stereoPara.R, R_eigen);
    cv2eigen(_stereoPara.M1, A_left);
    cv2eigen(_stereoPara.M2, A_right);

    Eigen::Matrix<float, 3, 4> T_left, T_right;
    T_left.leftCols<3>() = Eigen::Matrix3f::Identity();
    T_left.rightCols<1>() = Eigen::Matrix<float, 3, 1>::Zero();
    T_right.leftCols<3>() = R_eigen;
    Eigen::Matrix<float, 3, 1> tt;
    tt << _stereoPara.T.at<float>(0, 0), _stereoPara.T.at<float>(1, 0), _stereoPara.T.at<float>(2, 0);
    T_right.rightCols<1>() = tt;
    Eigen::Matrix<float, 3, 4> M_left, M_right;
    M_left = A_left * T_left;
    M_right = A_right * T_right;

    vector<float> tmp;

    Eigen::Matrix4f D(Eigen::Matrix4f::Zero());

    //-- 我这里是想把匹配错的3D点给拉回来,结果还是不行
    int kp2y = kp2.pt.y;
    if(kp2y == 1961)
    {
        D.block(0, 0, 1, 4) = kp1.pt.x * M_left.block(2, 0, 1, 4) - M_left.block(0, 0, 1, 4);
        D.block(1, 0, 1, 4) = kp1.pt.y * M_left.block(2, 0, 1, 4) - M_left.block(1, 0, 1, 4);
       D.block(2, 0, 1, 4) = 2148.0 * M_right.block(2, 0, 1, 4) - M_right.block(0, 0, 1, 4);
        D.block(3, 0, 1, 4) = 2006.0 * M_right.block(2, 0, 1, 4) - M_right.block(1, 0, 1, 4);
    }
    else
    {
        D.block(0, 0, 1, 4) = kp1.pt.x * M_left.block(2, 0, 1, 4) - M_left.block(0, 0, 1, 4);
        D.block(1, 0, 1, 4) = kp1.pt.y * M_left.block(2, 0, 1, 4) - M_left.block(1, 0, 1, 4);
        D.block(2, 0, 1, 4) = kp2.pt.x * M_right.block(2, 0, 1, 4) - M_right.block(0, 0, 1, 4);
        D.block(3, 0, 1, 4) = kp2.pt.y * M_right.block(2, 0, 1, 4) - M_right.block(1, 0, 1, 4);

    }

    /// **************************判断此次三角化是否有效************************ ///
    MatXX DTD;
    DTD = D.transpose() * D;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(DTD, Eigen::ComputeFullU | Eigen::ComputeFullV);
    float sigma4, sigma3;
    sigma3 = svd.singularValues()[2];
    sigma4 = svd.singularValues()[3] + 1e-16; // 防止出现除以0的情况
    if ((sigma3 / sigma4) > 120)
    {
        Eigen::Vector4f u4;
        u4 = svd.matrixU().block(0, 3, 4, 1);
        outPoint3d.x = u4[0] / u4[3];
        outPoint3d.y = u4[1] / u4[3];
        outPoint3d.z = u4[2] / u4[3];
        return true;
    }
    return false;
}

float FitBall::fitDiameter()
{
    //TODO
    float x0 = 3.0, y0 = 1.0, z0 = 35.0, r = 2.5;
    float Stop = pow((_ballCloud[0].x-x0),2)+pow((_ballCloud[0].y-y0),2)+pow((_ballCloud[0].z-z0),2)-pow(r,2);
    const int Samples = this->_ballCloud.size();

    for (int k = 0; k < 100; ++k)
    {
        // compute error
        MatXX error(MatXX::Zero(Samples, 1));
        int j=0;
        for (auto point : this->_ballCloud)
        {
//            double test = pow((point.x-x0),2)+pow((point.y-y0),2)+pow((point.z-z0),2)-pow(r,2);

            error.block(j,0,1,1) << pow((point.x-x0),2)+pow((point.y-y0),2)+pow((point.z-z0),2)-pow(r,2);
            j++;
        }

        if(error.col(0).norm() < 1e-5 * Stop){
            cout<<"因为r本身很小而退出"<<endl;
            break;
        }

        // compute A
        MatXX A(MatXX::Zero(Samples, 4));
        int i=0;
        for (auto point : this->_ballCloud)
        {
            Eigen::Matrix<float, 1, 4> tmp;
            tmp << -2*(point.x - x0), -2*(point.y - y0), -2*(point.z - z0), -2*r;
            A.block(i,0,1,4) = tmp;
            i++;
        }


        // compute s
        Eigen::Vector4f s = -1 * (A.transpose() * A).inverse() * A.transpose() * error;

        if(s.norm() < 1e-7)
        {
            cout<<"因为s太小而退出"<<endl;
            cout<<"迭代了"<<k<<"次"<<endl;
            break;
        }


        // update
        x0 += s[0];
        y0 += s[1];
        z0 += s[2];
        r  += s[3];
    }

    this->diameter = abs(r*2);
    this->centre.x = x0;
    this->centre.y = y0;
    this->centre.z = z0;

    return this->diameter;
}



/// 构造函数
generateCloud::generateCloud(Mat &imgLeft, Mat &imgRight, StereoParameter& stereoParameter)
:_imgLeft(imgLeft),_imgRight(imgRight),_stereoPara(stereoParameter)
{}

/**
 * 生成点云图像的3D坐标, 并保存到_linePoint3D中
 * @param numPoint 想要多少个点
 * @return
 */
bool generateCloud::generate3D(int numPoint)
{
// ① 先将两个图片中的光条2D坐标提取出来，放到各自的点中
    Mat showLeftImg = cv::Mat::zeros(_imgLeft.size(), CV_8UC1);
    Mat showRightImg = cv::Mat::zeros(_imgRight.size(), CV_8UC1);
    unsigned int pointNum = 0;
    Mat tmpleft = this->_imgLeft;
    Mat tmpRight = this->_imgRight;

    helpMeCalHessian->XFastLineExtraction(this->_imgLeft, 8.0, 0, threshold_line, 1.0, this->_leftPoint, pointNum, showLeftImg);
    helpMeCalHessian->XFastLineExtraction(this->_imgRight, 8.0, 0, threshold_line, 0., this->_rightPoint, pointNum, showRightImg);

    // ② 对左图像中的点，通过对极几何，找到右图像中的那条线
    matchLine(numPoint, showRightImg);

    return true;
}

/**
 * 对线上的点进行匹配
 * @param numPoint 输入想要多少个匹配点
 * @param img 可以输入个图片看看极线的焦点
 */
void generateCloud::matchLine(int numPoint, Mat& img)
{
    //计算一个评分
    map<float, int> scores;
    for (int i = 0; i < this->_leftPoint.size(); i++)
    {
        Point2f matchPoint;
        if (!fMatchPointfromLeft2Right(_leftPoint[i], _rightPoint, matchPoint, img))
        {
            _leftPointMatched.push_back(Point2f(0.0, 0.0));
            continue;
        }
        Point2f RematchPoint;
        if (!fMatchPointfromRight2Left(matchPoint, _leftPoint, RematchPoint, img))
        {
            _leftPointMatched.push_back(Point2f(0.0, 0.0));
            continue;
        }

        _leftPointMatched.push_back(matchPoint);//单独存一下左图对应的点

        //计算重投影误差
        float error = norm(_leftPoint[i] - RematchPoint);
        scores.insert(make_pair(error, i));//按照error进行自动排序
    }

    //取出error最低的前numPoint个点
    vector<Point2f> PointLeftToMatch, PointRightToMatch;
    map<float, int>::iterator iter = scores.begin();
    if (iter != scores.end())
    {
        for (int i = 0; i < numPoint; i++, iter++)
        {
            Mat tmpLeft = this->_imgLeft;
            Mat tmpRight = this->_imgRight;
            Point2f leftOne = _leftPoint[iter->second];
            Point2f rightOne = _leftPointMatched[iter->second];
            circle(_imgLeft, leftOne, 2, Scalar(0, 0, 0));
            circle(_imgRight, rightOne, 2, Scalar(0, 0, 0));
            PointLeftToMatch.push_back(_leftPoint[(*iter).second]);
            PointRightToMatch.push_back(_leftPointMatched[(*iter).second]);
        }

        if (PointLeftToMatch.size() != numPoint || PointRightToMatch.size() != numPoint)
            cerr << "!!! matchLine error !!!" << endl;
    }
    else
        cerr << "!!! scores error !!!" << endl;


    //开始计算3D点坐标
    Eigen::Matrix3f R_eigen, A_left, A_right;
    cv2eigen(this->_stereoPara.R, R_eigen);
    cv2eigen(this->_stereoPara.M1, A_left);
    cv2eigen(this->_stereoPara.M2, A_right);

    Eigen::Matrix<float, 3, 4> T_left, T_right;
    T_left.leftCols<3>() = Eigen::Matrix3f::Identity();
    T_left.rightCols<1>() = Eigen::Matrix<float, 3, 1>::Zero();
    T_right.leftCols<3>() = R_eigen;
    Eigen::Matrix<float, 3, 1> tt;
    tt << this->_stereoPara.T.at<float>(0, 0), this->_stereoPara.T.at<float>(1, 0), this->_stereoPara.T.at<float>(2, 0);
    T_right.rightCols<1>() = tt;

    //这里的M一定要是世界坐标系到图像坐标系
    Eigen::Matrix<float, 3, 4> M_left, M_right;
    M_left = A_left * T_left;
    M_right = A_right * T_right;

    vector<float> tmp;
    for (int i = 0; i < PointLeftToMatch.size(); i++)
    {
        Eigen::Matrix4f D(Eigen::Matrix4f::Zero());
        D.block(0, 0, 1, 4) = PointLeftToMatch[i].x * M_left.block(2, 0, 1, 4) - M_left.block(0, 0, 1, 4);
        D.block(1, 0, 1, 4) = PointLeftToMatch[i].y * M_left.block(2, 0, 1, 4) - M_left.block(1, 0, 1, 4);
        D.block(2, 0, 1, 4) = PointRightToMatch[i].x * M_right.block(2, 0, 1, 4) - M_right.block(0, 0, 1, 4);
        D.block(3, 0, 1, 4) = PointRightToMatch[i].y * M_right.block(2, 0, 1, 4) - M_right.block(1, 0, 1, 4);
        float max_D = D.maxCoeff();
        MatXX S = Eigen::MatrixXf::Identity(4, 4) / max_D;
        D = D * S;//为了值更稳定，除以一下D中最大值的逆


        /// **************************判断此次三角化是否有效************************ ///
        MatXX DTD;
        DTD = D.transpose() * D;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(DTD, Eigen::ComputeFullU | Eigen::ComputeFullV);
        float sigma4, sigma3;
        sigma3 = svd.singularValues()[2];
        sigma4 = svd.singularValues()[3] + 1e-16; // 防止出现除以0的情况
        tmp.push_back(sigma3 / sigma4);
        if ((sigma3 / sigma4) > 100)//应该是越大越好的
        {
            Eigen::Vector4f u4;
            u4 = svd.matrixU().block(0, 3, 4, 1);
            if (u4[2] / u4[3] > 0)//应该投影到相机平面的前方，即z>0
                this->_linePoint3D.push_back(Point3f(u4[0] / u4[3], u4[1] / u4[3], u4[2] / u4[3]));
        }
        else
            cout << "!!!the" << i <<"th' result is invalid!!!" << endl;
    }
}

/**
 * 已知左图像的点找有图像的点
 * @param point_ 左图的某一个点
 * @param PointVector 右图的点的集合
 * @param result 找到了右图中与之匹配的结果
 * @param img 画出图像看看
 * @return
 */
bool generateCloud::fMatchPointfromLeft2Right(const Point2f point_, const vector<Point2f> PointVector, Point2f &result, Mat &img)
{
    Mat F = _stereoPara.F;
    bool TheyCanCross = false;

    Mat matPoint = (Mat_<float>(3, 1) << point_.x, point_.y, 1);
    Mat poleline2 = F * matPoint;//3x1
    float a = poleline2.at<float>(0, 0);
    float b = poleline2.at<float>(1, 0);
    float c = poleline2.at<float>(2, 0);//那条线是ax+by+c=0

//    int col = _imgLeft.cols;
//    for (int i = 0; i < col; i++)
//    {
//        int y = (-a * i - c) / b;
//        circle(img, Point(i, y), 1, Scalar(255, 255, 255), 1);
//    }

    //寻找右图像中到poleline2最短的上下两个点
    Point2f PointAboveLine, PointBelowLine;
    float ddAboveMin = 10.0;
    float ddBelowMin = 10.0;
    float down = pow((a*a + b * b), 0.5);
    for (const auto & i : PointVector)
    {
        float dd = (a * i.x + b * i.y + c) / down;

        float u = i.x;
        float v = i.y;
        if (dd == 0.0)
        {
            result = i;
            return true;
        }

        if (dd > 0.0)//在线的下方
        {
            if (dd < ddAboveMin)
            {
                if (!TheyCanCross)
                    TheyCanCross = true;
                PointAboveLine = i;
                ddAboveMin = dd;
                continue;
            }
        }
        if (dd < 0.0)//在线的上方
        {
            dd = -1 * dd;
            if (dd < ddBelowMin)
            {
                if (!TheyCanCross)
                    TheyCanCross = true;
                PointBelowLine = i;
                ddBelowMin = dd;
            }
        }
    }

    //计算交点
    float x1 = PointAboveLine.x, y1 = PointAboveLine.y;
    float x2 = PointBelowLine.x, y2 = PointBelowLine.y;
    float x = -(c * (x2 - x1) + b * (y1 * x2 - y2 * x1)) / (b * (y2 - y1) + a * (x2 - x1));
    float y = (a * (y1 * x2 - y2 * x1) - c * (y2 - y1)) / (b * (y2 - y1) + a * (x2 - x1));
    result = Point2f(x, y);
    return TheyCanCross;
}

/**
 * 从右图像中找左图像的点, 以下参数与上类似
 * @param point_
 * @param PointVector
 * @param result
 * @param img
 * @return
 */
bool generateCloud::fMatchPointfromRight2Left(const Point2f point_, const vector<Point2f> PointVector, Point2f &result, Mat &img)
{
    Mat F = _stereoPara.F;

    bool TheyCanCross = false;

    Mat matPoint = (Mat_<float>(3, 1) << point_.x, point_.y, 1);
    Mat poleline2 = matPoint.t() * F;//1x3
    float a = poleline2.at<float>(0, 0);
    float b = poleline2.at<float>(0, 1);
    float c = poleline2.at<float>(0, 2);//那条线是ax+by+c=0

//    int row = _imgLeft.rows;
//    for (int i = 0; i < row; i++)
//    {
//        int y = (-a * i - c) / b;
//        circle(img, Point(i, y), 1, Scalar(255, 255, 255), 5);
//    }

    //寻找右图像中到poleline2最短的上下两个点
    Point2f PointAboveLine, PointBelowLine;
    float ddAboveMin = 1.0;
    float ddBelowMin = 1.0;
    float down = pow((a*a + b * b), 0.5);
    for (const auto & i : PointVector)
    {
        float dd = (a * i.x + b * i.y + c) / down;

        if (dd == 0.0)
        {
            result = i;
            return true;
        }

        if (dd > 0.0)//在线的上方
        {
            if (dd < ddAboveMin)
            {
                if (!TheyCanCross)
                    TheyCanCross = true;
                PointAboveLine = i;
                ddAboveMin = dd;
                continue;
            }
        }
        if (dd < 0.0)//在线的下方
        {
            dd = -1 * dd;
            if (dd < ddBelowMin)
            {
                if (!TheyCanCross)
                    TheyCanCross = true;
                PointBelowLine = i;
                ddBelowMin = dd;
            }
        }
    }

    //计算交点
    float x1 = PointAboveLine.x, y1 = PointAboveLine.y;
    float x2 = PointBelowLine.x, y2 = PointBelowLine.y;
    float x = -(c * (x2 - x1) + b * (y1 * x2 - y2 * x1)) / (b * (y2 - y1) + a * (x2 - x1));
    float y = (a * (y1 * x2 - y2 * x1) - c * (y2 - y1)) / (b * (y2 - y1) + a * (x2 - x1));
    result = Point2f(x, y);
    return TheyCanCross;
}


/// ===========================================
/// ======      下面是一些没用的东西    =======
/// ===========================================
/*就用最简单的方法，阈值法，且滤掉一些单独存在的点
void generateCloud::simpleMethod(Mat& img, vector<Point2f>& Point)//这次是栽倒这里了，加个引用的符号，否则值无法传出去
{
    //img.convertTo(img, CV_8UC1);
    int i = 1, j = 1;
    for (i = 1; i < img.rows-1; i++)
    {
        uchar* p = img.ptr<uchar>(i);
        int sum_y = 0;
        int num_y = 0;
        for (j = 1; j < img.cols-1; j++)
        {
            if(p[j]>threshold_line)
            {
                sum_y += j;
                num_y++;
            }
        }
        if(num_y>0)
            Point.push_back(Point2f(sum_y/num_y, i));
    }
}



//steger(_imgLeft, _leftPoint);
//steger(_imgRight, _rightPoint);
void generateCloud::steger(Mat& img, vector<Point2f>& Point)
{
    Mat tmp_imgLeft;// = _imgLeft.clone()

    threshold(img, tmp_imgLeft, threshold_line, 255, THRESH_BINARY);
    //高斯滤波
    tmp_imgLeft.convertTo(tmp_imgLeft, CV_32FC1);
    GaussianBlur(tmp_imgLeft, tmp_imgLeft, Size(9, 9), 3., 3.);

    //一阶偏导数
    Mat m1, m2;
    m1 = (Mat_<float>(1, 3) << 1, 0, -1);  //x偏导
    m2 = (Mat_<float>(3, 1) << 1, 0, -1);  //y偏导

    Mat dx, dy;
    filter2D(tmp_imgLeft, dx, CV_32FC1, m1);
    filter2D(tmp_imgLeft, dy, CV_32FC1, m2);

    //二阶偏导数
    Mat m3, m4, m5;
    m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
    m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
    m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导

    Mat dxx, dyy, dxy;
    filter2D(tmp_imgLeft, dxx, CV_32FC1, m3);
    filter2D(tmp_imgLeft, dyy, CV_32FC1, m4);
    filter2D(tmp_imgLeft, dxy, CV_32FC1, m5);

    //hessian矩阵
    double maxD = -1;
    int imgcol = _imgLeft.cols;
    int imgrow = _imgLeft.rows;
    Mat tmp2 = tmp_imgLeft.clone();


    不得不记录一下，搞了我一天。。。
    .at<float>(i, j)意思是第i行第j列的点
    .ptr(i)[j]意思是第i行的第j列的点
    Point2d要存的时候还是要存（x，y）很难理解吗。。。


    for (int i = 0; i < imgrow; i++)
    {
        bool CanIstop = false;
        float* p = tmp2.ptr<float>(i);
        for (int j = 0; j < imgcol; j++)
        {
            float ttt = p[j];//第i行第j列
            if (ttt > 0)
            {
                CanIstop = true;
                Mat hessian(2, 2, CV_32FC1);
                //hessian<< dxx.at<float>(j, i), dxy.at<float>(j, i), dxy.at<float>(j, i), dyy.at<float>(j, i);
                hessian.at<float>(0, 0) = dxx.at<float>(i, j);//第i行第j列
                hessian.at<float>(0, 1) = dxy.at<float>(i, j);
                hessian.at<float>(1, 0) = dxy.at<float>(i, j);
                hessian.at<float>(1, 1) = dyy.at<float>(i, j);

                Mat eValue;
                Mat eVectors;
                eigen(hessian, eValue, eVectors);

                float nx, ny;
                float fmaxD = 0;
                if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //求特征值最大时对应的特征向量
                {
                    nx = eVectors.at<float>(0, 0);
                    ny = eVectors.at<float>(0, 1);
                    fmaxD = eValue.at<float>(0, 0);
                }
                else
                {
                    nx = eVectors.at<float>(1, 0);
                    ny = eVectors.at<float>(1, 1);
                    fmaxD = eValue.at<float>(1, 0);
                }
                float t = -(nx*dx.at<float>(i, j) + ny * dy.at<float>(i, j)) / (nx*nx*dxx.at<float>(i, j) + 2 * nx*ny*dxy.at<float>(i, j) + ny * ny*dyy.at<float>(i, j));

                if (fabs(t*nx) <= 0.5 && fabs(t*ny) <= 0.5)
                    Point.push_back(Point2f(j, i));//只有涉及点坐标存储的时候才需要将这些点进行转换到x，y坐标
            }
        }
    }

    Mat tmp1 = _imgLeft.clone();

    for (int k = 0; k < Point.size(); k++)
    {
        Point2f rpt = _leftPoint[k];
        circle(tmp1, rpt, 1, Scalar(0, 0, 0), 1);
    }

    //for (int k = 0; k < _rightPoint.size(); k++)
    //{
    //	Point2d rpt = _rightPoint[k];
    //	circle(tmpr, rpt, 1, Scalar(0, 0, 0), 1);
    //}
}
*/

//Mat generateCloud::computeF()
//{
//	Mat F;
//	Matx33f S;
//	S << 0., -1 * t_left_right.z, t_left_right.y,
//		t_left_right.z, 0., -1 * t_left_right.x,
//		-1 * t_left_right.y, t_left_right.x, 0.;
//	F = rightCamIntrinsics.t().inv() * S * R_left_right * leftCamIntrinsics.inv();
//	return F;
//}

