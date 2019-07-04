#include <iostream>
#include<io.h>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include"opencv2/flann.hpp"
#include"opencv2/xfeatures2d.hpp"
#include"opencv2/ml.hpp"

using namespace cv;
using namespace std;
using namespace xfeatures2d;
using namespace ml;

#define CLASS_COUNT 10
#define METHOD_COUNT 5

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{ 
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void main()
{
	string strDateset[CLASS_COUNT];
	strDateset[0] = "anemone"; strDateset[1] = "buttercup"; strDateset[2] = "coltsfoot"; strDateset[3] = "daffodils"; strDateset[4] = "daisy"; strDateset[5] = "fritillariae"; strDateset[6] = "galanthus"; strDateset[7] = "iris"; strDateset[8] = "sunflower"; strDateset[9] = "tiger lily";
	string strMethod[METHOD_COUNT];
	strMethod[0] = "SIFT"; strMethod[1] = "SURF"; strMethod[2] = "BRISK"; strMethod[3] = "ORB"; strMethod[4] = "FREAK";
	////递归读取目录下全部文件
	Mat T;
	Mat Img;
	Mat descriptors1;
	Mat descriptors2;

	vector<string> files;
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	vector<DMatch> matches;
	vector<DMatch> good_matches;
	vector<double> t_average(METHOD_COUNT);
	
	int data_size = 769;

	cout << "TEST BEGIN" << endl;
	//遍历各种特征点寻找方法
	for (int method = 0; method < METHOD_COUNT - 1; method++) //遍历每种方法
	{
		string buf = strMethod[method];
		cout << "当前方法为：" << buf << endl;
		t_average[method] = 0;                                       
		//遍历各个路径
		for (int n = 0; n< CLASS_COUNT; n++) //遍历每个数据集
		{
			//获得测试图片绝对地址
			string path = "F:\\OpenCV\\img\\" + strDateset[n];
			cout << "当前数据集为" << strDateset[n];
			//获得当个数据集中的图片
			getFiles(path, files);
			cout << " 共" << files.size() << "张图片" << endl;
			for (int j = 1; j < files.size(); j++) //遍历每张图像
			{
				T = imread(files[0], 0); //使用T对比余下的图片，得出结果	
				Img = imread(files[j], 0);
				//生成特征点算法及其匹配方法
				double t = (double)getTickCount();
				Ptr<Feature2D>  extractor;
				BFMatcher matcher;
				switch (method)
				{
				case 0: //SIFT
					extractor = SIFT::create();
					matcher = BFMatcher(NORM_L2);
					break;
				case 1: //SURF
					extractor = SURF::create();
					matcher = BFMatcher(NORM_L2);
					break;
				case 2: //BRISK
					extractor = BRISK::create();
					matcher = BFMatcher(NORM_HAMMING);
					break;
				case 3: //ORB
					extractor = ORB::create();
					matcher = BFMatcher(NORM_HAMMING);
					break;
				case 4: //FREAK
					extractor = FREAK::create();
					matcher = BFMatcher(NORM_HAMMING);
					break;
				}

				//提取特征点
				try
				{
					extractor->detectAndCompute(T, Mat(), keypoints1, descriptors1);
					extractor->detectAndCompute(Img, Mat(), keypoints2, descriptors2);
					matcher.match(descriptors1, descriptors2, matches);
				}
				catch (...)
				{
					cout << "特征点提取时发生错误" << endl;
					continue;
				}

				//提取有效特征点good_matches
				sort(matches.begin(), matches.end());
				int ptsPairs = min(50, (int)(matches.size()*0.15));
				cout << ptsPairs << endl;
				for (int i = 0; i < ptsPairs; i++)
				{
					good_matches.push_back(matches[i]);
				}
				if (good_matches.size()<4)
				{
					cout << "有效特征点数目小于4个，粗匹配失败" << endl;
					continue;
				}

				//画出结果
				Mat result;
				drawMatches(T, keypoints1, Img, keypoints2, good_matches, result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				vector<Point2f> obj;
				vector<Point2f> scene;
				for (size_t i = 0; i < good_matches.size(); i++)
				{
					obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
					scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
				}

				vector<Point2f> obj_corners(4);
				vector<Point2f> scene_corners(4);
				obj_corners[0] = Point(0, 0);
				obj_corners[1] = Point(T.cols, 0);
				obj_corners[2] = Point(T.cols, T.rows);
				obj_corners[3] = Point(0, T.rows);
				
				Mat H = findHomography(obj, scene, RANSAC);      //寻找匹配的图像
				perspectiveTransform(obj_corners, scene_corners, H);

				line(result, scene_corners[0] + Point2f((float)T.cols, 0), scene_corners[1] + Point2f((float)T.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);       //绘制
				line(result, scene_corners[1] + Point2f((float)T.cols, 0), scene_corners[2] + Point2f((float)T.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
				line(result, scene_corners[2] + Point2f((float)T.cols, 0), scene_corners[3] + Point2f((float)T.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
				line(result, scene_corners[3] + Point2f((float)T.cols, 0), scene_corners[0] + Point2f((float)T.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
				t = (double)getTickCount() - t;
				cout << "运行时间：" <<t / ((double)getTickFrequency())*1000. << endl;
				t_average[method] += t / ((double)getTickFrequency()) * 1000 / data_size;
				char charJ[255];
				sprintf_s(charJ, "_%4d.jpg", j);
				string strResult = "F:\\OpenCV\\FastTemplateMatching\\result\\" + strDateset[n] + "\\" + buf + charJ;
				imwrite(strResult, result);
				matches.clear();
				good_matches.clear();
			}
			cout << "方法" << buf << "平均运行时间为：" << t_average[method] << endl;
			files.clear();
		}
	}
	getchar();
	cv::waitKey();
	return ;
};



