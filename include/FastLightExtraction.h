#pragma once
#include"head.h"
class FastLightExtraction
{
public:
	FastLightExtraction();
	~FastLightExtraction();

	float* cvGKdx_vct32f(float sigma, int n);
	float* cvGKdy_vct32f(float sigma, int n);
	float* cvGKdxx_vct32f(float sigma, int n);
	float* cvGKdxy_vct32f(float sigma, int n);
	float* cvGKdyy_vct32f(float sigma, int n);

	float* xGKAll_Vct32F(float sigma, int n);

	void XFastLineExtraction(Mat& img, float fsigma, int nKerSize, unsigned char ucGrayThd, float fNSDT, 
		vector<Point2f>&ptLine, unsigned int unPtsNum, Mat& pByLinePointBMP);
	void XFastLineExtraction(Mat &img, float fSigma, int nKerSize, unsigned char ucGrayThd, float fNSDT, vector<Point2f>&ptLine, unsigned int *unPtsNum,
		Mat &pByLinePointBMP, Mat &SupPixelinfImg);


	bool undistortLabelPoints(vector<Point2f>&Pts, Mat K, Mat dis);
};

