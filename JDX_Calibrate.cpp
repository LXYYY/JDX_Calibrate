#include <iostream>
#include <opencv2/opencv.hpp>
#include "CVClass.h"

using namespace std;
using namespace cv;

#define Calibrate
#define ShowUndistorted

CVClass cvClass;

int main()
{
    cvClass.cam = VideoCapture(0);
    cvClass.camCalib();
    cvClass.errorCalc();
//    VideoCapture cap(0);
//    Mat frame;
//    while(1){
//        cap>>frame;
//        imshow("test",frame);
//        waitKey(1);
//    }
    return 0;
}

