#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>
#include <cmath>
#include <math.h>
#include <tgmath.h>

using namespace std;
using namespace cv;

/*Global Declarations*/
int        g_slider_position = 0;
CvCapture* g_capture         = NULL;
int counter = 1,group_counter = 0;
int Video_Numbers = 0;

/*
 * Function definitions
 */

//cosine helper
static double cos_angle(cv::Point point1, cv::Point point2, cv::Point point0)
{
	double diff_x1 = point1.x - point0.x;
	double diff_y1 = point1.y - point0.y;
	double diff_x2 = point2.x - point0.x;
	double diff_y2 = point2.y - point0.y;
	return (diff_x1*diff_x2 + diff_y1*diff_y2)/sqrt((diff_x1*diff_x1 + diff_y1*diff_y1)*(diff_x2*diff_x2 + diff_y2*diff_y2) + 1e-10);
}

// Video tracker
void onTrackbarSlide(int pos) {
	cvSetCaptureProperty(
			g_capture,
			CV_CAP_PROP_POS_FRAMES,
			pos
	);

	Mat frame = cvQueryFrame(g_capture);

	std::string location = "./W_group/";
	std::string ext = ".jpg";
	std::string strcounter = std::to_string(counter);
	std::string strgroupcounter = std::to_string(group_counter);
	string fileprefix = "W_";
	std::string filename = fileprefix.append(strgroupcounter);
	filename.append(strcounter);
	filename.append(ext);
	std::string path = location.append(filename);
	imwrite(path,frame);
	counter++;
}

//Video capture helper
void CaptureVideo(string pathStr)
{		g_slider_position = 0;
counter = 0;
group_counter++;
const char *cstr = pathStr.c_str();
cvNamedWindow("Video Stream", CV_WINDOW_AUTOSIZE );
g_capture = cvCreateFileCapture(cstr);

int frames = (int) cvGetCaptureProperty(
		g_capture,
		CV_CAP_PROP_FRAME_COUNT
);
if( frames!= 0 ) {
	cvCreateTrackbar(
			"Position",
			"Video Stream",
			&g_slider_position,
			frames,
			onTrackbarSlide
	);
}
Mat frame;

while(1) {
	frame = cvQueryFrame(g_capture);
	if( !frame.data ) break;
	imshow("Video Stream",frame);
	//cvSetTrackbarPos("Position","Video Stream",++g_slider_position);
	char c = cvWaitKey(33);
	if( c == 27 ) break;
}
cvReleaseCapture( &g_capture );
cvDestroyWindow( "Video Stream" );
//return(0);
}

Mat formatImagesForPCA(const vector<Mat> &data)
{
	Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_64F);
	for(unsigned int i = 0; i < data.size(); i++)
	{
		Mat image_row = data[i].clone().reshape(1,1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i,CV_64F);
	}
	return dst;
}

vector<float> getPhi(vector<float> featureVector)
{
	vector<float> phiVector;
	int length = featureVector.size();

	phiVector.push_back(1.0);

	for(int i=0;i<length;i++)
	{
		phiVector.push_back(2*featureVector[i]);
	}

	for(int i=0;i<length;i++)
	{
		for(int j=i;j<length;j++)
		{
			phiVector.push_back(2*featureVector[i]*2*featureVector[j]);
		}
	}

	for(int i=0;i<length;i++)
	{
		phiVector.push_back(4*featureVector[i]*featureVector[i]-2);
	}

	for(int i=0;i<phiVector.size();i++)
	{
		cout<<"\t"<<phiVector[i];
	}
	return phiVector;
}
vector<float> getPhi(Mat featureVector)
{
	vector<float> phiVector;
	int length = featureVector.cols;

	phiVector.push_back(1.0);

	for(int i=0;i<length;i++)
	{
		phiVector.push_back(2*featureVector.at<float>(0,i));
	}

	for(int i=0;i<length;i++)
	{
		for(int j=i;j<length;j++)
		{
			phiVector.push_back(2*featureVector.at<float>(0,i)*2*featureVector.at<float>(0,j));
		}
	}

	for(int i=0;i<length;i++)
	{
		phiVector.push_back(4*featureVector.at<float>(0,i)*featureVector.at<float>(0,i)-2);
	}

	return phiVector;
}
vector<float> getAlpha(vector<Mat> cluster)
{
	int N = cluster.size();
	//float sum=0;
	vector<float> alphaVector;

	vector<vector<float>> phiVectors;
	for(int i=0;i<N;i++)
	{
		phiVectors.push_back(getPhi(cluster[i]));
	}


	for(int i=0;i<N;i++)
	{
		float sum = 0;
		for(int j=0;j<N;j++)
		{
			sum= sum+phiVectors[j][i];
			cout<<"sum inside"<<sum;
		}
		cout<<"\n\n\nsum"<<sum/N;
		alphaVector.push_back(sum/N);
	}
	return alphaVector;
}

float getProbability(Mat testFeatureVector,vector<Mat> cluster)
{
	float sum = 0;
	vector<float> phiVector = getPhi(testFeatureVector);
	cout<<"\n\n Phi Vector";
	for(int i=0;i<phiVector.size();i++)
	{
		cout<<"   "<<phiVector[i];
	}

	vector<float> alphaVector = getAlpha(cluster);
	cout<<"\n\n\n"<<alphaVector.size()<<"\n\nsum in prob";
	for(int i=0;i<alphaVector.size();i++)
	{
		sum+=alphaVector[i]*phiVector[i];
		cout<<"   "<<sum;
	}

	return sum;
}

// Feature Extraction from stored images
//std::map<int, cv::Mat> FindFeatures(const char* directory){
std::vector<Mat> FindFeatures(const char* directory){
	cv::Mat gray,thresh,dst,frame;
	std::vector<Mat> featureVectorGroup;
	std::map<int,cv::Mat> feat;
	vector< vector<Point> > contours,result;
	vector<Point> approx;
	vector<Moments> mu(contours.size() );
	vector<Point2f> mc( contours.size() );

	DIR *dir;
	struct dirent *ent;
	dir = opendir(directory);

	if(dir != NULL)
	{


		while((ent = readdir(dir)) != NULL)
		{
			cout << ent->d_name<<endl;
			string path = string(directory).append("/");
			if (!strcmp (ent->d_name, "."))
				continue;
			if (!strcmp (ent->d_name, ".."))
				continue;

			path.append(ent->d_name);
			string name = std::string(ent->d_name);
			Mat destination, kernel;
			kernel = Mat::ones(3,3,CV_32F)/(float) 3*3;
			frame = imread(path,1);
			cv::cvtColor(frame, gray, CV_BGR2GRAY);
			filter2D(gray,destination,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
			threshold(gray,thresh,214,255,CV_THRESH_BINARY);
			Canny(gray,destination,100,300,3);
			findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			frame.copyTo(dst);
			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

				Moments mom = moments( contours[i], false );
				double hu[7];

				// Skip small or non-convex objects
				vector<Rect> boundRect( contours.size() );
				if (std::fabs(cv::contourArea(contours[i])) < 2000 || std::fabs(cv::arcLength(contours[i],true)) < 150 || std::fabs(cv::arcLength(contours[i],true)) > 550 || !cv::isContourConvex(approx))
					continue;

				if (approx.size() == 4)
				{
					int vtc = approx.size();

					// Get the cosines of all corners
					vector<double> cos;
					for (int j = 2; j < vtc+1; j++)
						cos.push_back(cos_angle(approx[j%vtc], approx[j-2], approx[j-1]));

					// Sort ascending the cosine values
					std::sort(cos.begin(), cos.end());

					// Get the lowest and the highest cosine
					double mincos = cos.front();
					double maxcos = cos.back();

					// Use the degrees obtained above and the number of vertices
					// to determine the shape of the contour
					if (vtc == 4 ){
						boundRect[i] = boundingRect(contours[i]);
						double ratio = abs(1 - (double)boundRect[i].width / boundRect[i].height);

						line(dst, approx.at(0), approx.at(1), cvScalar(0,0,255),4);
						line(dst, approx.at(1), approx.at(2), cvScalar(0,0,255),4);
						line(dst, approx.at(2), approx.at(3), cvScalar(0,0,255),4);
						line(dst, approx.at(3), approx.at(0), cvScalar(0,0,255),4);
						double area = contourArea(contours[i],false);
						cout<<"Area is : "<<area<<endl;

						double length = arcLength(contours[i],true);
						//cout << "Length is : "<<length<<endl;

						double xbar = mom.m10/mom.m00;
						double ybar = mom.m01/mom.m00;
						//cout<<"X bar is: "<< xbar <<endl;
						//cout<< "Y bar is"<<ybar <<endl;

						double centroidx = boundRect[i].x + (boundRect[i].width / 2);
						double centroidy = boundRect[i].y + (boundRect[i].height / 2);

						HuMoments(mom,hu);

						//						for(int k=0;k<7;k++){
						//							cout<<"Moment is:"<<hu[k]<<endl;
						//						}
						double x = mom.m20 + mom.m02;
						double y = 4*(mom.m11*mom.m11) + ((mom.m20 - mom.m02)*(mom.m20-mom.m02));
						double theta = (x+sqrt(y))/(x-sqrt(y));

						//cout << imgno << endl;
						int imgno=0;
						Mat featureVector(1,15,CV_64F);
						featureVector.at<double>(imgno,0) = log(mom.mu20);
						featureVector.at<double>(imgno,1) = log(mom.mu11);
						featureVector.at<double>(imgno,2) = log(mom.mu02);
						featureVector.at<double>(imgno,3) = log(mom.mu30);
						featureVector.at<double>(imgno,4) = log(mom.mu21);
						featureVector.at<double>(imgno,5) = log(mom.mu12);
						featureVector.at<double>(imgno,6) = log(mom.mu03);
						featureVector.at<double>(imgno,7) = log(hu[0]);
						featureVector.at<double>(imgno,8) = log(hu[1]);
						featureVector.at<double>(imgno,9) = log(hu[2]);
						featureVector.at<double>(imgno,10) = log(hu[3]);
						featureVector.at<double>(imgno,11) = log(hu[4]);
						featureVector.at<double>(imgno,12) = log(hu[5]);
						featureVector.at<double>(imgno,13) = log(hu[6]);
						featureVector.at<double>(imgno,14) = log(theta)*100;

						featureVectorGroup.push_back(featureVector);

					}
				}

			}
		}
		for(int i =0;i<featureVectorGroup.size();i++)
		{
			//cout<<featureVectorGroup[i]<<"\n";
		}
	}
	closedir (dir);
	return featureVectorGroup;
} 

int main() {
	DIR *dir;
	struct dirent *ent;
	dir = opendir("./Clips");

	if (dir != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			Video_Numbers++;
			//cout << ent->d_name<<endl;
			string path = "./Clips/";
			if (!strcmp (ent->d_name, "."))
				continue;
			if (!strcmp (ent->d_name, ".."))
				continue;

			path.append(ent->d_name);
			cout << path;
			CaptureVideo(path);

		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
		return EXIT_FAILURE;
	}

	std::vector<Mat> inputFeatureGroup = FindFeatures("./W_group");
	Mat featureMat = formatImagesForPCA(inputFeatureGroup);
	//cout <<endl<<"size"<<featureMat.size();
	PCA pca(featureMat,cv::Mat(),CV_PCA_DATA_AS_ROW, 10);
	Mat vp = pca.project(featureMat);


	int clusterCount = 5;
	Mat labels;
	int attempts = 250;
	Mat centers;

	vp.convertTo(vp,CV_32F);
	kmeans(vp, clusterCount, labels,
			TermCriteria( 250, 250, 1.0),
			attempts, KMEANS_RANDOM_CENTERS , centers);

	cout << centers<<endl;
	cout<< labels<<endl;
	cout<<"\n\n"<<vp;

	vector<Mat> cluster1;
	vector<Mat> cluster2;
	vector<Mat> cluster3;
	vector<Mat> cluster4;
	vector<Mat> cluster5;

	for (int i = 0; i < vp.rows; ++i)
	{
		Mat vec = vp.row(i);
		if (labels.at<int>(i) == 0)
		{
			cluster1.push_back(vec);
		}
		if (labels.at<int>(i) == 1)
		{
			cluster2.push_back(vec);
		}
		if (labels.at<int>(i) == 2)
		{
			cluster3.push_back(vec);
		}
		if (labels.at<int>(i) == 3)
		{
			cluster4.push_back(vec);
		}
		if (labels.at<int>(i) == 4)
		{
			cluster5.push_back(vec);
		}


	}

	std::vector<Mat> testFeatureGroup = FindFeatures("./test");
	cout<<testFeatureGroup.size()<<endl<<endl;
	//Mat testFeatureRaw = testFeatureGroup[0];

	Mat featureMat1 = formatImagesForPCA(testFeatureGroup);
		//cout <<endl<<"size"<<featureMat.size();
	PCA pca1(featureMat1,cv::Mat(),CV_PCA_DATA_AS_ROW, 10);
	Mat testFeaturePCA = pca1.project(featureMat1);
	cout<<"================================================================================="<<endl;
	cout<<testFeaturePCA.row(0)<<endl;
	cout<<"================================================================================="<<endl;
	// Eigenvalue and Eigenvector calculation

	Mat mean = pca.mean.clone();
	Mat eigenvalues = pca.eigenvalues.clone();
	Mat eigenvectors = pca.eigenvectors.clone();

	// Count D1

	float d1 = getProbability(testFeaturePCA.row(0),cluster1);
	float d2 = getProbability(testFeaturePCA.row(0),cluster2);
	float d3 = getProbability(testFeaturePCA.row(0),cluster3);
	float d4 = getProbability(testFeaturePCA.row(0),cluster4);
	float d5 = getProbability(testFeaturePCA.row(0),cluster5);

	cout<<endl;

	float d[5] = {d1,d2,d3,d4,d5};

	int max = 0;
	for(int i=0;i<5;i++)
	{
		if(d[i]<0)
		{
			d[i]=d[i]*-1;
		}

		if(d[i]>d[max])
		{
			max=i;
		}
	}

	cout<<"\n\nDistance"<<d[max]<<"   i+ group"<<max;



	waitKey(0);

	return 0;

}
