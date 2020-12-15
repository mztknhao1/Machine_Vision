#include "../include/FastLightExtraction.h"


FastLightExtraction::FastLightExtraction()
{
}


FastLightExtraction::~FastLightExtraction()
{
}


float* FastLightExtraction::cvGKdx_vct32f(float sigma, int n)
{
	/*
	一阶导Gy
	*/
	int n1 = n - 1 - n / 2;
	float a = (float)(1.0 / (2 * CV_PI * pow(sigma, 4)));
	float e = (float)2.718281828;
	float *fK = new float[n*n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fK[i*n + j]
				= a * (j - n1) * pow(e, (float)(-1.0 * ((i - n1) * (i - n1) + (j - n1) * (j - n1)) / (2.0 * sigma * sigma)));
		}
	}
	return fK;
}


float* FastLightExtraction::cvGKdy_vct32f(float sigma, int n)
{
	/*
	二阶导Gxx
	*/
	int n1 = n - 1 - n / 2;
	float a = (float)(1.0 / (2 * CV_PI * pow(sigma, 4)));
	float e = (float)2.718281828;
	float *fK = new float[n*n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fK[i*n + j]
				= a * (i - n1)*pow(e, (float)(-1.0*((i - n1)*(i - n1) + (j - n1)*(j - n1)) / (2.0*sigma*sigma)));
		}
	}
	return fK;
}


float* FastLightExtraction::cvGKdxx_vct32f(float sigma, int n)
{
	/*
	二阶导Gxx
	*/
	int n1 = n - 1 - n / 2;
	float a = (float)(1.0 / (2 * CV_PI * pow(sigma, 6)));
	float e = (float)2.718281828;
	float *fK = new float[n*n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fK[i*n + j]
				= a * ((n1 - j)*(n1 - j) - sigma * sigma)*pow(e, (float)(-1.0*((i - n1)*(i - n1) + (j - n1)*(j - n1)) / (2.0*sigma*sigma)));
		}
	}
	return fK;
}

float* FastLightExtraction::cvGKdxy_vct32f(float sigma, int n)
{
	/*
	二阶导Gxy
	*/

	int n1 = n - 1 - n / 2;
	float a = (float)(1.0 / (2 * CV_PI * pow(sigma, 6)));
	float e = (float)2.718281828;
	float *fK = new float[n*n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fK[i*n + j]
				= a * (n1 - i)*(n1 - j)*pow(e, (float)(-1.0*((i - n1)*(i - n1) + (j - n1)*(j - n1)) / (2.0*sigma*sigma)));
		}
	}
	return fK;
}

float* FastLightExtraction::cvGKdyy_vct32f(float sigma, int n)
{
	/*
	二阶导Gyy
	*/
	int n1 = n - 1 - n / 2;
	float a = (float)(1.0 / (2 * CV_PI * pow(sigma, 6)));
	float e = (float)2.718281828;
	float *fK = new float[n*n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fK[i*n + j]
				= a * ((n1 - i)*(n1 - i) - sigma * sigma)*pow(e, (float)(-1.0*((i - n1)*(i - n1) + (j - n1)*(j - n1)) / (2.0*sigma*sigma)));
		}
	}
	return fK;
}

float* FastLightExtraction::xGKAll_Vct32F(float sigma, int n)
{

	//如果输入n为0，就自动计算模板宽度
	if (0 == n)
	{
		n = (int)(5.65252f*sigma + 1) / 2 * 2 + 1;
	}
	float *pKersData = new float[5 * n*n];
	float *pGKdx = cvGKdx_vct32f(sigma, n);
	float *pGKdy = cvGKdy_vct32f(sigma, n);
	float *pGKd2x = cvGKdxx_vct32f(sigma, n);
	float *pGKd2y = cvGKdyy_vct32f(sigma, n);
	float *pGKdxy = cvGKdxy_vct32f(sigma, n);
	memcpy(pKersData, pGKdx, n*n * sizeof(float));
	memcpy(&(pKersData[n*n]), pGKdy, n*n * sizeof(float));
	memcpy(&(pKersData[2 * n*n]), pGKd2x, n*n * sizeof(float));
	memcpy(&(pKersData[3 * n*n]), pGKd2y, n*n * sizeof(float));
	memcpy(&(pKersData[4 * n*n]), pGKdxy, n*n * sizeof(float));
	delete[]pGKdx;
	delete[]pGKdy;
	delete[]pGKd2x;
	delete[]pGKd2y;
	delete[]pGKdxy;
	return pKersData;
}

void FastLightExtraction::XFastLineExtraction(Mat &img, float fSigma, int nKerSize, unsigned char ucGrayThd, float fNSDT, 
	vector<Point2f>&ptLine, unsigned int unPtsNum, Mat &pByLinePointBMP)
{
	/*
	Eigen::MatrixXf& ptMat
	提取光条中心
	@param img: 输入图像
	@param fSigma: 高斯函数的sigma
	@param nKerSize: 模板大小
	@param ucGrayThd: 图像灰度阈值
	@param fNSDT：二阶导的阈值
	@param ptLine： 返回值，光条中心点坐标
	@param unPtsNum： 返回值，点的个数
	*/

	int nHeight = img.rows;
	int nWidth = img.cols;


	//Mat pByLinePointBMP(img.rows, img.cols, CV_8UC1);
	//Mat pByLinePointBMP_1(img.rows, img.cols, CV_8UC1);

	//声明
	//extern std::map<std::string, Point2f> supPixelIndex;
	//extern std::map<std::string, Point2f> supPixelNxy;

	float	*pFlSubPixelX = new float[nHeight*nWidth];
	float	*pFlSubPixelY = new float[nHeight*nWidth];
	float	*pFlEgVctX = new float[nHeight*nWidth];
	float	*pFlEgVctY = new float[nHeight*nWidth];

	unsigned int *pUnLinePointX = new unsigned int[10000];
	unsigned int *pUnLinePointY = new unsigned int[10000];

	// 若输入的nKerSize为0，则根据输入的fsigma大小给nKerSize赋值
	if (0 == nKerSize)
	{
		nKerSize = (int)(5.65252f*fSigma + 1) / 2 * 2 + 1;
	}

	float *pfKersData = xGKAll_Vct32F(fSigma, nKerSize);

	int row = 0, col = 0, row1 = 0, col1 = 0;
	float ev1 = 0, ev2 = 0, aev1 = 0, aev2 = 0;
	float nx1 = 0, ny1 = 0, nx2 = 0, ny2 = 0;
	float rx = 0, ry = 0, rxx = 0, ryy = 0, rxy = 0;
	int nRowWidth = 0, nRow1Width = 0;
	float *pfGKdx = pfKersData;
	float *pfGKdy = &(pfKersData[nKerSize*nKerSize]);
	float *pfGKd2x = &(pfKersData[2 * nKerSize*nKerSize]);
	float *pfGKd2y = &(pfKersData[3 * nKerSize*nKerSize]);
	float *pfGKdxy = &(pfKersData[4 * nKerSize*nKerSize]);

	unsigned int unNum = 0;

	int r00 = nKerSize / 2 + 1, r01 = nHeight - nKerSize / 2;//遍历所有的像素，为了避免像素越界
	int c00 = nKerSize / 2 + 1, c01 = nWidth - nKerSize / 2;
	for (row = r00; row < r01; row++)
	{
		nRowWidth = row * nWidth;
		for (col = c00; col < c01; col++)
		{
			float a = img.ptr<unsigned char>(row)[col] < ucGrayThd;
			if (img.ptr<unsigned char>(row)[col] < ucGrayThd) continue;//图像灰度小于给定阈值，跳过

			if (unNum == 10000) continue;//找到10000个点跳过

			int nInc = 0;
			float fPixel = 0;
			rx = 0;
			ry = 0;
			rxx = 0;
			ryy = 0;
			rxy = 0;

			//对满足阈值要求的点 ：求一阶和二阶导卷积
			int r10 = row - nKerSize / 2, r11 = row + nKerSize / 2 + 1;//模板的边界
			int c10 = col - nKerSize / 2, c11 = col + nKerSize / 2 + 1;//模板的边界
			for (row1 = r10; row1 < r11; row1++)
			{
				nRow1Width = row1 * nWidth;
				for (col1 = c10; col1 < c11; col1++)
				{
					fPixel = (float)(img.ptr<unsigned char>(row1)[col1] / 255.0);//归一化后的灰度值
					rx += pfGKdx[nInc] * fPixel;//一阶导数模板运算
					ry += pfGKdy[nInc] * fPixel;
					rxx += pfGKd2x[nInc] * fPixel;
					ryy += pfGKd2y[nInc] * fPixel;
					rxy += pfGKdxy[nInc] * fPixel;
					nInc++;
				}
			}
			//计算特征值特征向量
			if (rxy == 0)
			{
				ev1 = rxx;
				ev2 = ryy;
				nx1 = 1;
				ny1 = 0;
				nx2 = 0;
				ny2 = 1;
			}
			else
			{
				float theta = (ryy - rxx) / (2 * rxy);
				float t = (float)(1 / (fabs(theta) + sqrt(theta*theta + 1)));
				if (theta < 0) t = -1 * t;
				float c = (float)(1 / sqrt(t*t + 1));
				float s = t * c;
				ev1 = rxx - t * rxy;
				ev2 = ryy + t * rxy;
				nx1 = c;
				ny1 = -1 * s;
				nx2 = s;
				ny2 = c;
			}
			//选取绝对值大的方向作为线条的法线方向，
			//对中间明亮的线条，法线方向的二阶导为负值，其值小于一定的阈值即可被作为候选中心点
			aev1 = (float)fabs(ev1);
			aev2 = (float)fabs(ev2);
			if (aev1 > aev2)
			{
				if (ev1 < fNSDT)
				{
					float t = -1 * (nx1 * rx + ny1 * ry) / ev1;//沿线条的法线方向寻找一阶方向导数为0的点对应的t
					float p1 = t * nx1;
					float p2 = t * ny1;
					if ((p1 < 0.505) && (p1 > -0.505) && (p2 < 0.505) && (p2 > -0.505))//发现方向一阶导数为0的点位于当前像素内
					{

						pByLinePointBMP.ptr<unsigned char>(row)[col] = 255;
						pFlEgVctX[nRowWidth + col] = nx1;
						pFlEgVctY[nRowWidth + col] = ny1;
						pFlSubPixelX[nRowWidth + col] = p1 + col;
						pFlSubPixelY[nRowWidth + col] = p2 + row;
						pUnLinePointX[(unNum)] = col;
						pUnLinePointY[(unNum)] = row;
						ptLine.push_back(Point2f(p1 + col, p2 + row));

						//supPixelIndex.insert(pair<string, Point2f>(to_string(col) + " " + to_string(row), Point2f(p1 + col, p2 + row)));
						//supPixelNxy.insert(pair<string, Point2f>(to_string(col) + " " + to_string(row), Point2f(nx1, ny1)));
						unNum++;
					}
				}
			}
			else
			{
				if (ev2 < fNSDT)
				{
					float t = -1 * (nx2 * rx + ny2 * ry) / ev2;
					float p1 = t * nx2;
					float p2 = t * ny2;
					if ((p1 < 0.505) && (p1 > -0.505) && (p2 < 0.505) && (p2 > -0.505))
					{

						pByLinePointBMP.ptr<unsigned char>(row)[col] = 255;
						pFlEgVctX[nRowWidth + col] = nx2;
						pFlEgVctY[nRowWidth + col] = ny2;
						pFlSubPixelX[nRowWidth + col] = p1 + col;
						pFlSubPixelY[nRowWidth + col] = p2 + row;
						pUnLinePointX[(unNum)] = col;
						pUnLinePointY[(unNum)] = row;
						ptLine.push_back(Point2f(p1 + col, p2 + row));
						
						//supPixelIndex.insert(pair<string, Point2f>(to_string(col) + " " + to_string(row), Point2f(p1 + col, p2 + row)));
						//supPixelNxy.insert(pair<string, Point2f>(to_string(col) + " " + to_string(row), Point2f(nx2, ny2)));
						unNum++;
					}
				}
			}
		}
	}

	delete[]pFlSubPixelX;
	delete[]pFlSubPixelY;
	delete[]pFlEgVctX;
	delete[]pFlEgVctY;
	delete[]pUnLinePointX;
	delete[]pUnLinePointY;
	delete[]pfKersData;

	unPtsNum = unNum;
}

//输出亚像素的点vector
//输出像素点图像

//输出四通道的图像：1-2 表示亚像素坐标 3-4表示该点的法向分量
//重载+1
void FastLightExtraction::XFastLineExtraction(Mat &img, float fSigma, int nKerSize, unsigned char ucGrayThd, float fNSDT, vector<Point2f>&ptLine, unsigned int *unPtsNum,
	Mat &pByLinePointBMP, Mat &SupPixelinfImg)
{
	/*
	提取光条中心
	@param img: 输入图像
	@param fSigma: 高斯函数的sigma
	@param nKerSize: 模板大小
	@param ucGrayThd: 图像灰度阈值
	@param fNSDT：二阶导的阈值
	@param ptLine： 返回值，光条中心点坐标
	@param unPtsNum： 返回值，点的个数
	*/

	int nHeight = img.rows;
	int nWidth = img.cols;

	//Mat pByLinePointBMP(img.rows, img.cols, CV_8UC1);
	//Mat pByLinePointBMP_1(img.rows, img.cols, CV_8UC1);


	float	*pFlSubPixelX = new float[nHeight*nWidth];
	float	*pFlSubPixelY = new float[nHeight*nWidth];
	float	*pFlEgVctX = new float[nHeight*nWidth];
	float	*pFlEgVctY = new float[nHeight*nWidth];

	unsigned int *pUnLinePointX = new unsigned int[10000];
	unsigned int *pUnLinePointY = new unsigned int[10000];

	// 若输入的nKerSize为0，则根据输入的fsigma大小给nKerSize赋值
	if (0 == nKerSize)
	{
		nKerSize = (int)(5.65252f*fSigma + 1) / 2 * 2 + 1;
	}

	float *pfKersData = xGKAll_Vct32F(fSigma, nKerSize);

	int row = 0, col = 0, row1 = 0, col1 = 0;
	float ev1 = 0, ev2 = 0, aev1 = 0, aev2 = 0;
	float nx1 = 0, ny1 = 0, nx2 = 0, ny2 = 0;
	float rx = 0, ry = 0, rxx = 0, ryy = 0, rxy = 0;
	int nRowWidth = 0, nRow1Width = 0;
	float *pfGKdx = pfKersData;
	float *pfGKdy = &(pfKersData[nKerSize*nKerSize]);
	float *pfGKd2x = &(pfKersData[2 * nKerSize*nKerSize]);
	float *pfGKd2y = &(pfKersData[3 * nKerSize*nKerSize]);
	float *pfGKdxy = &(pfKersData[4 * nKerSize*nKerSize]);

	unsigned int unNum = 0;

	int r00 = nKerSize / 2 + 1, r01 = nHeight - nKerSize / 2;//遍历所有的像素，为了避免像素越界
	int c00 = nKerSize / 2 + 1, c01 = nWidth - nKerSize / 2;
	for (row = r00; row < r01; row++)
	{
		nRowWidth = row * nWidth;
		for (col = c00; col < c01; col++)
		{
			float a = img.ptr<unsigned char>(row)[col] < ucGrayThd;
			if (img.ptr<unsigned char>(row)[col] < ucGrayThd) continue;//图像灰度小于给定阈值，跳过

			if (unNum == 10000) continue;//找到10000个点跳过

			int nInc = 0;
			float fPixel = 0;
			rx = 0;
			ry = 0;
			rxx = 0;
			ryy = 0;
			rxy = 0;
			int r10 = row - nKerSize / 2, r11 = row + nKerSize / 2 + 1;//模板的边界
			int c10 = col - nKerSize / 2, c11 = col + nKerSize / 2 + 1;//模板的边界
			for (row1 = r10; row1 < r11; row1++)
			{
				nRow1Width = row1 * nWidth;
				for (col1 = c10; col1 < c11; col1++)
				{
					fPixel = (float)(img.ptr<unsigned char>(row1)[col1] / 255.0);//归一化后的灰度值
					rx += pfGKdx[nInc] * fPixel;//一阶导数模板运算
					ry += pfGKdy[nInc] * fPixel;
					rxx += pfGKd2x[nInc] * fPixel;
					ryy += pfGKd2y[nInc] * fPixel;
					rxy += pfGKdxy[nInc] * fPixel;
					nInc++;
				}
			}
			//计算特征值特征向量
			if (rxy == 0)
			{
				ev1 = rxx;
				ev2 = ryy;
				nx1 = 1;
				ny1 = 0;
				nx2 = 0;
				ny2 = 1;
			}
			else
			{
				float theta = (ryy - rxx) / (2 * rxy);
				float t = (float)(1 / (fabs(theta) + sqrt(theta*theta + 1)));
				if (theta < 0) t = -1 * t;
				float c = (float)(1 / sqrt(t*t + 1));
				float s = t * c;
				ev1 = rxx - t * rxy;
				ev2 = ryy + t * rxy;
				nx1 = c;
				ny1 = -1 * s;
				nx2 = s;
				ny2 = c;
			}
			//选取绝对值大的方向作为线条的法线方向，
			//对中间明亮的线条，法线方向的二阶导为负值，其值小于一定的阈值即可被作为候选中心点
			aev1 = (float)fabs(ev1);
			aev2 = (float)fabs(ev2);
			if (aev1 > aev2)
			{
				if (ev1 < fNSDT)
				{
					float t = -1 * (nx1 * rx + ny1 * ry) / ev1;//沿线条的法线方向寻找一阶方向导数为0的点对应的t
					float p1 = t * nx1;
					float p2 = t * ny1;
					if ((p1 < 0.505) && (p1 > -0.505) && (p2 < 0.505) && (p2 > -0.505))//发现方向一阶导数为0的点位于当前像素内
					{

						pByLinePointBMP.ptr<unsigned char>(row)[col] = 255;

						SupPixelinfImg.ptr<Vec4f>(row)[col][0] = p1 + col;
						SupPixelinfImg.ptr<Vec4f>(row)[col][1] = p2 + row;
						SupPixelinfImg.ptr<Vec4f>(row)[col][2] = nx2;
						SupPixelinfImg.ptr<Vec4f>(row)[col][3] = ny2;


						pFlEgVctX[nRowWidth + col] = nx1;
						pFlEgVctY[nRowWidth + col] = ny1;
						pFlSubPixelX[nRowWidth + col] = p1 + col;
						pFlSubPixelY[nRowWidth + col] = p2 + row;
						pUnLinePointX[(unNum)] = col;
						pUnLinePointY[(unNum)] = row;
						ptLine.push_back(Point2f(p1 + col, p2 + row));
						unNum++;
					}
				}
			}
			else
			{
				if (ev2 < fNSDT)
				{
					float t = -1 * (nx2 * rx + ny2 * ry) / ev2;
					float p1 = t * nx2;
					float p2 = t * ny2;
					if ((p1 < 0.505) && (p1 > -0.505) && (p2 < 0.505) && (p2 > -0.505))
					{

						pByLinePointBMP.ptr<unsigned char>(row)[col] = 255;
						SupPixelinfImg.ptr<Vec4f>(row)[col][0] = p1 + col;
						SupPixelinfImg.ptr<Vec4f>(row)[col][1] = p2 + row;
						SupPixelinfImg.ptr<Vec4f>(row)[col][2] = nx1;
						SupPixelinfImg.ptr<Vec4f>(row)[col][3] = ny1;

						pFlEgVctX[nRowWidth + col] = nx2;
						pFlEgVctY[nRowWidth + col] = ny2;
						pFlSubPixelX[nRowWidth + col] = p1 + col;
						pFlSubPixelY[nRowWidth + col] = p2 + row;
						pUnLinePointX[(unNum)] = col;
						pUnLinePointY[(unNum)] = row;
						ptLine.push_back(Point2f(p1 + col, p2 + row));
						unNum++;
					}
				}
			}
		}
	}

	delete[]pFlSubPixelX;
	delete[]pFlSubPixelY;
	delete[]pFlEgVctX;
	delete[]pFlEgVctY;
	delete[]pUnLinePointX;
	delete[]pUnLinePointY;
	delete[]pfKersData;

	*unPtsNum = unNum;
}

bool FastLightExtraction::undistortLabelPoints(vector<Point2f>&Pts, Mat K, Mat dis) {
	/*
	实现图像像素坐标的畸变矫正
	@param Pts: 有畸变的图像点坐标
	@param undisPts: 返回值，无畸变的图像点坐标
	@param K: 相机内参
	@param dis: 相机畸变参数
	*/

	vector<Point2f> Pts_dis, undisPts;
	//undistortPoints(Pts, Pts_dis, K, dis);

	// 需要将变换后的点先变换成齐次坐标，然后乘以相机参数矩阵，才是最终的去畸变点坐标
	Mat Pts_ThreeCols = Mat::ones(1, 3, CV_32FC1);

	size_t LenPts = Pts_dis.size();
	for (size_t i = 0; i < LenPts; i++)
	{
		// Pts_ThreeCols形如[x, y, 1]
		Pts_ThreeCols.at<float>(0, 0) = Pts_dis[i].x;
		Pts_ThreeCols.at<float>(0, 1) = Pts_dis[i].y;

		// 3*1的矩阵
		Mat undistortPts_ThreeCols = K * Pts_ThreeCols.t();
		Point2f Point = Point2f(undistortPts_ThreeCols.at<float>(0, 0), undistortPts_ThreeCols.at<float>(1, 0));
		undisPts.push_back(Point);
	}

	Pts = undisPts;

	return true;
}