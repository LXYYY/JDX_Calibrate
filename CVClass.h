#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


using namespace std;
using namespace cv;

class CVClass {
private:
    class CamClass {
    public:
        CamClass() {}

        ~CamClass() {}

        VideoCapture cam;
        int exposure = 90;
        Size imgSize = Size(1280, 1024);
        uchar *buffer;
        int fd;
        int fd_temp;
        char camFile[20] = "/dev/video0";

        bool openCam(int id);

        bool init_v4l2(int id);

        bool v4l2_grab(void);


        int get_Video_Parameter(void);

        Mat getImage(void);
    };

    CamClass camL;
    CamClass camR;


    enum {
        DETECTION = 0, CAPTURING = 1, CALIBRATED = 2
    };
    enum Pattern {
        CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID
    };
    Size boardSize = Size(7, 7);
    Pattern pattern = CIRCLES_GRID;
    float squareSize = 100;
    float aspectRatio = 1;
    bool writeExtrinsics = true;
    bool writePoints = true;
    bool flipVertical = false;
    bool showUndistorted = true;
    int delay = 0;

    void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f> &corners, Pattern patternType);


    double computeReprojectionErrors(
            const vector<vector<Point3f> > &objectPoints,
            const vector<vector<Point2f> > &imagePoints,
            const vector<Mat> &rvecs, const vector<Mat> &tvecs,
            const Mat &cameraMatrix, const Mat &distCoeffs,
            vector<float> &perViewErrors);

    bool runCalibration(vector<vector<Point2f> > imagePoints,
                        Size imageSize, Size boardSize, Pattern patternType,
                        float squareSize, float aspectRatio,
                        int flags, Mat &cameraMatrix, Mat &distCoeffs,
                        vector<Mat> &rvecs, vector<Mat> &tvecs,
                        vector<float> &reprojErrs,
                        double &totalAvgErr);

    void saveCameraParams(const string &filename,
                          Size imageSize, Size boardSize,
                          float squareSize, float aspectRatio, int flags,
                          const Mat &cameraMatrix, const Mat &distCoeffs,
                          const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                          const vector<float> &reprojErrs,
                          const vector<vector<Point2f> > &imagePoints,
                          double totalAvgErr);

    bool runAndSave(const string &outputFilename,
                    const vector<vector<Point2f> > &imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                    float aspectRatio, int flags, Mat &cameraMatrix,
                    Mat &distCoeffs, bool writeExtrinsics, bool writePoints);

public:
	CVClass() {};

	~CVClass() {};

    enum ColorE {
        YELLOW = 0,
        BLUE,
        GREEN
    };
    class objBoxImg {
    public:
        ColorE color;
        int id;

        objBoxImg();

        objBoxImg(vector<Point> tPts, ColorE tColor, int tId);

        ~objBoxImg();

        vector<Point> pts;

        bool print(void);

        bool sortPts(void);
    };

    class objBox3D {
    public:
        objBox3D();

        ~objBox3D();

        ColorE color;
        int id;
        vector<Point3f> pts3D;
    };

    struct CamParamS                //0 left ,1 right
    {
        Mat cameraMatrix[2];
        Mat distCoeffs[2];
        Mat R, T;
        Mat R1, R2;
        Mat P1, P2;
        Mat Q;
        Mat map1[2], map2[2];
        Size imgSize;
        Mat R2W,T2W;

    } camParam;
    Mat frame;
    Mat frameL, frameR;
    Mat rFrameL, rFrameR;

    bool worldCSInited = false;

    bool camParamInit(void);

    bool undistortFrame(void);

    bool getPoint3d(vector<Point> pt2dL, vector<Point> pt2dR, vector<Mat> &pt3d);

    bool getPointWorld(vector<Mat> &pt3d,vector<Mat> &ptW);

	vector<Mat> imgs,imgsUndistorted;

    bool camCalib(void);

	void errorCalc(void);

    bool stereoCalib(void);

    bool worldCSInit(void);

    bool camInit(bool LR);

	VideoCapture cam;

    bool getImage();

    bool showImage();

    bool showUndistortedImage();

    char waitKeyProc(int delay);

    bool findGreen(Mat &img);

    bool adjustContrast(Mat &src);

    bool getProjMats(Size imgSize);

    vector<objBox3D> computeBox3D(vector<objBoxImg> objBoxImgL, vector<objBoxImg> objBoxImgR);

    bool processSingle(Mat img, bool LorR, vector<objBoxImg> &objBoxRlt);

    vector<objBox3D> processStereo();
};

