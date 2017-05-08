///*
// * feat.cpp
// *
// *  Created on: Mar 16, 2017
// *      Author: pranav and kathan
// */
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/opencv.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//#include <iostream>
//#include <string>
//#include <dirent.h>
//#include<cmath>
//#include<math.h>
//
//using namespace std;
//using namespace cv;
//
//static double cos_angle(cv::Point point1, cv::Point point2, cv::Point point0)
//{
//    double diff_x1 = point1.x - point0.x;
//    double diff_y1 = point1.y - point0.y;
//    double diff_x2 = point2.x - point0.x;
//    double diff_y2 = point2.y - point0.y;
//    return (diff_x1*diff_x2 + diff_y1*diff_y2)/sqrt((diff_x1*diff_x1 + diff_y1*diff_y1)*(diff_x2*diff_x2 + diff_y2*diff_y2) + 1e-10);
//}
//
//
//int        g_slider_position = 0;
//CvCapture* g_capture         = NULL;
//int counter = 0;
//
//void onTrackbarSlide(int pos) {
//    cvSetCaptureProperty(
//        g_capture,
//        CV_CAP_PROP_POS_FRAMES,
//        pos
//    );
//
//    Mat frame = cvQueryFrame(g_capture);
//    Mat gray,thresh,dst;
//    vector< vector<Point> > contours,result;
//    vector<Point> approx;
//    vector<Moments> mu(contours.size() );
//    vector<Point2f> mc( contours.size() );
//
//    std::string location = "/home/pranav/Documents/CMPE297_Video/CMPE297-1/W_group/";
//    std::string ext = ".jpg";
//    std::string filename = std::to_string(counter);
//    filename.append(ext);
//    std::string path = location.append(filename);
//    imwrite(path,frame);
//    counter++;
//
//    cv::cvtColor(frame, gray, CV_BGR2GRAY);
//    threshold(gray,thresh,210,255,CV_THRESH_BINARY);
//    findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//    frame.copyTo(dst);
//
//    for (int i = 0; i < contours.size(); i++)
//               {
//                   approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
//
//                   Moments mom = moments( contours[i], false );
//                   double hu[7];
//
//                   // Skip small or non-convex objects
//                   vector<Rect> boundRect( contours.size() );
//                   if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
//                       continue;
//
//                   if (approx.size() == 4)
//                   {
//                       int vtc = approx.size();
//
//                       // Get the cosines of all corners
//                       vector<double> cos;
//                       for (int j = 2; j < vtc+1; j++)
//                           cos.push_back(cos_angle(approx[j%vtc], approx[j-2], approx[j-1]));
//
//                       // Sort ascending the cosine values
//                       std::sort(cos.begin(), cos.end());
//
//                       // Get the lowest and the highest cosine
//                       double mincos = cos.front();
//                       double maxcos = cos.back();
//
//                       // Use the degrees obtained above and the number of vertices
//                       // to determine the shape of the contour
//                       if (vtc == 4 ){
//                       	boundRect[i] = boundingRect(contours[i]);
//                       	double ratio = abs(1 - (double)boundRect[i].width / boundRect[i].height);
//
//                       	line(dst, approx.at(0), approx.at(1), cvScalar(0,0,255),4);
//                       	line(dst, approx.at(1), approx.at(2), cvScalar(0,0,255),4);
//                       	line(dst, approx.at(2), approx.at(3), cvScalar(0,0,255),4);
//                       	line(dst, approx.at(3), approx.at(0), cvScalar(0,0,255),4);
//                           double area = contourArea(contours[i],false);
//                           cout<<"Area is : "<<area<<endl;
//
//                           double length = arcLength(contours[i],true);
//                           cout << "Length is : "<<length<<endl;
//
//                           double xbar = mom.m10/mom.m00;
//                           double ybar = mom.m01/mom.m00;
//                           cout<<"X bar is: "<< xbar <<endl;
//                           cout<< "Y bar is"<<ybar <<endl;
//
//                           double centroidx = boundRect[i].x + (boundRect[i].width / 2);
//                           double centroidy = boundRect[i].y + (boundRect[i].height / 2);
//
//                           HuMoments(mom,hu);
//
//                           for(int k=0;k<7;k++){
//                           	cout<<"Moment %d is:"<<i+1<<hu[k]<<endl;
//                           }
//
//
//                       }
//                   }
//
//               }
//    cv::imshow("src", frame);
//    cv::imshow("dst", dst);
//}
//
//int CaptureVideo(string pathStr)
//{
//	    const char *cstr = pathStr.c_str();
//	    cvNamedWindow("Example3", CV_WINDOW_AUTOSIZE );
//	    g_capture = cvCreateFileCapture(cstr);
//
//	    int frames = (int) cvGetCaptureProperty(
//	        g_capture,
//	        CV_CAP_PROP_FRAME_COUNT
//	    );
//	    if( frames!= 0 ) {
//	      cvCreateTrackbar(
//	          "Position",
//	          "Example3",
//	          &g_slider_position,
//	          frames,
//	          onTrackbarSlide
//	      );
//	    }
//	    Mat frame;
//
//	    while(1) {
//	        frame = cvQueryFrame(g_capture);
//	        if( !frame.data ) break;
//	        imshow("Example3",frame);
//	        char c = cvWaitKey(33);
//	        if( c == 27 ) break;
//	    }
//	    cvReleaseCapture( &g_capture );
//	    cvDestroyWindow( "Example3" );
//	    return(0);
//}
//
//int main() {
//	DIR *dir;
//	struct dirent *ent;
//	dir = opendir("/home/pranav/Documents/CMPE297_Video/CMPE297-1/Clips");
//
//	if (dir != NULL) {
//	  /* print all the files and directories within directory */
//	  while ((ent = readdir (dir)) != NULL) {
//	    //cout << ent->d_name<<endl;
//		  string path = "/home/pranav/Documents/CMPE297_Video/CMPE297-1/Clips/";
//		  if (!strcmp (ent->d_name, "."))
//		             continue;
//		         if (!strcmp (ent->d_name, ".."))
//		             continue;
//
//			  path.append(ent->d_name);
//			  cout << path;
//			  int a = CaptureVideo(path);
//		  }
//	  closedir (dir);
//	} else {
//	  /* could not open directory */
//	  perror ("");
//	  return EXIT_FAILURE;
//	}
//
//
//
//}
