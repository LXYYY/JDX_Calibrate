#include "CVClass.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

bool CVClass::camParamInit(void) {
    //intrinsics
    FileStorage paramL("/home/jdrobot/Desktop/JDRobot/paramL.yaml", FileStorage::READ);
    FileStorage paramR("/home/jdrobot/Desktop/JDRobot/paramR.yaml", FileStorage::READ);

    if (!paramL.isOpened()) {
        cout << "paramL.xml not available" << endl;
        getchar();
        return false;
    }
    if (!paramR.isOpened()) {
        cout << "paramR.xml not available" << endl;
        return false;
    }

    cout << (int) paramL["nframes"] << endl;

    paramL["camera_matrix"] >> camParam.cameraMatrix[0];
    paramL["distortion_coefficients"] >> camParam.distCoeffs[0];

    paramR["camera_matrix"] >> camParam.cameraMatrix[1];
    paramR["distortion_coefficients"] >> camParam.distCoeffs[1];

    paramL.release();
    paramR.release();

    getImage();
    if (frameL.size() != frameR.size()) {
        cout << "imgSizes not equal" << endl;
        return false;
    } else
        camParam.imgSize = frameL.size();

    try {
        initUndistortRectifyMap(camParam.cameraMatrix[0], camParam.distCoeffs[0], Mat(),
                                getOptimalNewCameraMatrix(camParam.cameraMatrix[0], camParam.distCoeffs[0],
                                                          camParam.imgSize, 1, camParam.imgSize, 0),
                                camParam.imgSize, CV_16SC2, camParam.map1[0], camParam.map2[0]);
        initUndistortRectifyMap(camParam.cameraMatrix[1], camParam.distCoeffs[1], Mat(),
                                getOptimalNewCameraMatrix(camParam.cameraMatrix[1], camParam.distCoeffs[1],
                                                          camParam.imgSize, 1, camParam.imgSize, 0),
                                camParam.imgSize, CV_16SC2, camParam.map1[1], camParam.map2[1]);
    }
    catch (...) {
        cout << "initUndistortedRectifyMap failed " << endl;
        return false;
    }

    //extrinsics
    FileStorage extrinsics("/home/jdrobot/Desktop/JDRobot/extrinsics.yml", FileStorage::READ);
    if (!extrinsics.isOpened()) {
        cout << "extrinsics.xml not available" << endl;
        return false;
    }

    extrinsics["R"] >> camParam.R;
    extrinsics["T"] >> camParam.T;
    extrinsics["R1"] >> camParam.R1;
    extrinsics["R2"] >> camParam.R2;
    extrinsics["P1"] >> camParam.P1;
    extrinsics["P2"] >> camParam.P2;
    extrinsics["Q"] >> camParam.Q;

    extrinsics.release();

    return true;
}

bool CVClass::undistortFrame(void) {
    try {
        remap(frameL, rFrameL, camParam.map1[0], camParam.map2[0], INTER_LINEAR);
        remap(frameR, rFrameR, camParam.map1[1], camParam.map2[1], INTER_LINEAR);
    }
    catch (...) {
        cout << "remap failed" << endl;
        return true;
    }
    return false;
}

Mat_<double> LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                                   Matx34d P,       //camera 1 matrix
                                   Point3d u1,      //homogenous image point in 2nd camera
                                   Matx34d P1       //camera 2 matrix
) {
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
              u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
              u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
              u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2)
    );
    Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)),
            -(u.y * P(2, 3) - P(1, 3)),
            -(u1.x * P1(2, 3) - P1(0, 3)),
            -(u1.y * P1(2, 3) - P1(1, 3)));

    Mat_<double> X;
    solve(A, B, X, DECOMP_SVD);

    return X;
}

#define EPSILON 0.005

Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                            Matx34d P,          //camera 1 matrix
                                            Point3d u1,         //homogenous image point in 2nd camera
                                            Matx34d P1          //camera 2 matrix
) {

    double wi = 1, wi1 = 1;
    Mat_<double> X(4, 1);

    for (int i = 0; i < 20; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
                  (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
                  (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1,
                  (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                  (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1,
                  (u1.y * P1(2, 2) - P1(1, 2)) / wi1
        );
        Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
                -(u.y * P(2, 3) - P(1, 3)) / wi,
                -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
                -(u1.y * P1(2, 3) - P1(1, 3)) / wi1
        );

        solve(A, B, X_, DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
    }
    Mat_<double> rlt(3, 1);
    rlt(0) = X(0);
    rlt(1) = X(1);
    rlt(2) = X(2);
    return rlt;
}


bool CVClass::getPoint3d(vector<Point> pts2dL, vector<Point> pts2dR, vector<Mat> &pts3d) {
    pts3d.clear();
    if (pts2dL.size() != pts2dR.size()) {
        cout << "pt2dL.size() != pt2dR.size()" << endl;
        return false;
    }
    try {
        pts3d.clear();
        size_t npts = pts2dL.size();
        vector<Mat> pts3dM(npts);
        pts3d.resize(npts);
        for (size_t i = 0; i < npts; i++) {
            Point3d ptL = Point3d(pts2dL.at(i).x, pts2dL.at(i).y, 1);
            Point3d ptR = Point3d(pts2dR.at(i).x, pts2dR.at(i).y, 1);
            pts3d.at(i) = IterativeLinearLSTriangulation(ptL, camParam.P1,
                                                         ptR, camParam.P2);
//            cout << pts3d.at(i) << endl;
        }
    }
    catch (...) {
        cout << "compute 3d points failed" << endl;
        return false;
    }
    return true;
}

void CVClass::calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f> &corners,
                                    Pattern patternType = CHESSBOARD) {
    corners.resize(0);

    switch (patternType) {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float(j * squareSize),
                                              float(i * squareSize), 0));
            break;

        case ASYMMETRIC_CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                                              float(i * squareSize), 0));
            break;

        default:
            CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

void CVClass::errorCalc(void)
{

    Mat errorImg;
    errorImg.create(Size(640,480), CV_8UC3);
    errorImg.setTo(0);
    vector<Point> centersR;
    vector<Point> centersU;
    for(int i=0;i<imgs.size();i++){
		Mat imgRaw = imgs.at(i);
		Mat imgUndistorted = imgsUndistorted.at(i);

        vector<Point> pointbufR;
        vector<Point> pointbufU;
		bool found[2];
		imshow("check1", imgRaw);
		imshow("check2", imgUndistorted);
        found[0] = findCirclesGrid(imgRaw, boardSize, pointbufR);
        found[1] = findCirclesGrid(imgUndistorted, boardSize, pointbufU);
        cout << "fuck:" << (Mat)pointbufR <<endl<<(Mat)pointbufU<< endl;

		cout << "found?" << found[0]<<ends<<found[1] << endl;
        cout<<"check"<<endl;
        if (found[0] && found[1]) {
            centersR.push_back(pointbufR.at(24));
            centersU.push_back(pointbufU.at(24));
//            vector<Point> convexR;
//            vector<Point> convexU;
//            convexHull(pointbufR, convexR);
//            convexHull(pointbufU, convexU);

//            vector<vector<Point> > convexArrayR;
//            vector<vector<Point> > convexArrayU;
//            convexArrayR.push_back(convexR);
//            convexArrayU.push_back(convexU);

//            for(int j=0;j<pointbufR.size();j++){
//                circle(errorImg,pointbufR.at(j),2,Scalar(0,0,255),-1);
//                circle(errorImg,pointbufU.at(j),2,Scalar(0,255,0),-1);
//            }
//            drawContours(errorImg, convexArrayR, -1, Scalar(0, 0, 255));
//            drawContours(errorImg, convexArrayU, -1, Scalar(0, 255, 0));



//            imshow("errorImg", errorImg);
//            waitKey(1);
        }
        else continue;
	}
    for(int i=0;i<centersR.size()-1;i++){
        line(errorImg,centersR.at(i),centersR.at(i+1),Scalar(0,0,255),2);
        line(errorImg,centersU.at(i)+centersR.at(0)-centersU.at(0),centersU.at(i+1)+centersR.at(0)-centersU.at(0),Scalar(0,255,0),2);
    }
    cout<<(centersR.at(0)-centersR.at(centersR.size()-1))-(centersU.at(0)-centersU.at(centersU.size()-1))<<endl;
    imshow("errorImg", errorImg);
    waitKey(0);
}

bool CVClass::worldCSInit(void) {
    //bool foundTagsL;
    //bool foundTagsR;
    //foundTagsL = aprilTags.processImage(frameL, aprilTags.tagsL);
    //foundTagsR = aprilTags.processImage(frameR, aprilTags.tagsR);

    //namedWindow("tags left");
    //namedWindow("tags right");

    //Mat tImgL, tImgR;
    //frameL.copyTo(tImgL);
    //frameR.copyTo(tImgR);

//    if (foundTagsL && foundTagsR) {
//        aprilTags.drawTags(tImgL, aprilTags.tagsL);
//        aprilTags.drawTags(tImgR, aprilTags.tagsR);
//    }
//
//    resize(tImgL, tImgL, Size(), 0.5, 0.5);
//    resize(tImgR, tImgR, Size(), 0.5, 0.5);
//    imshow("tags left", tImgL);
//    imshow("tags right", tImgR);
//
//    char c = waitKeyProc(1);
//
//    if (c == 'w') {
//        cout << c << endl;
//        cout << foundTagsL << " " << foundTagsR << endl;
//        if (foundTagsL && foundTagsR) {
//            try {
//                vector<Mat> pts3d;
//                vector<Point> pts2dL, pts2dR;
//
//                for (int i = 0; i < 4; i++) {
//                    pts2dL.push_back(
//                            Point(aprilTags.tagsL.at(0).p[i].first,
//                                  aprilTags.tagsL.at(0).p[i].second));
//                    pts2dR.push_back(
//                            Point(aprilTags.tagsR.at(0).p[i].first,
//                                  aprilTags.tagsR.at(0).p[i].second));
//                }
//                pts2dL.push_back(
//                        Point(aprilTags.tagsL.at(0).cxy.first,
//                              aprilTags.tagsL.at(0).cxy.second));
//                pts2dR.push_back(
//                        Point(aprilTags.tagsR.at(0).cxy.first,
//                              aprilTags.tagsR.at(0).cxy.second));
//
//                cout << (Mat) pts2dL << endl;
//                getPoint3d(pts2dL, pts2dR, pts3d);
//                for (size_t i = 0; i < pts3d.size(); i++) {
//                    cout << pts3d.at(i) << endl;
//                }
//
//                Mat_<double> worldMat(3, 3);
//                Mat_<double> ptsMat(3, 3);
//                Mat_<double> deltaMat(3, 3);
//                Mat_<double> p5Mat;
//
//                pts3d.at(0).copyTo(ptsMat.col(0));
//                pts3d.at(2).copyTo(ptsMat.col(1));
////                cout<<"test1"<<pts3d.at(0)<<endl<<pts3d.at(1)<<endl<<pts3d.at(2)<<endl
////                        <<pts3d.at(0)-pts3d.at(1)<<endl<<pts3d.at(2)-pts3d.at(1)<<endl;
////                cout<<"pts3d.at(0).cross(pts3d.at(1)"<<(pts3d.at(0)-pts3d.at(2)).cross(pts3d.at(1)-pts3d.at(4))<<endl;
//                p5Mat = (pts3d.at(0) - pts3d.at(1)).cross(pts3d.at(2) - pts3d.at(1)) + pts3d.at(1);
//                p5Mat.copyTo(ptsMat.col(2));
//
//                pts3d.at(1).copyTo(deltaMat.col(0));
//                pts3d.at(1).copyTo(deltaMat.col(1));
//                pts3d.at(1).copyTo(deltaMat.col(2));
//
//                worldMat.at<double>(0, 0) = -150.f;
//                worldMat.at<double>(1, 0) = 0.f;
//                worldMat.at<double>(2, 0) = 0.f;
//                worldMat.at<double>(0, 1) = 0.f;
//                worldMat.at<double>(1, 1) = 150.f;
//                worldMat.at<double>(2, 1) = 0.f;
//                Mat tmp;
//                tmp = worldMat.col(0).cross(worldMat.col(1));
//                tmp.copyTo(worldMat.col(2));
////                worldMat.at<double>(0, 2) = 0;
////                worldMat.at<double>(1, 2) = 0;
////                worldMat.at<double>(2, 2) = -22500;
//
////                worldMat/=norm(worldMat.col(0));
//
//                cout << "ptsMat:" << ptsMat << endl;
//                cout << "deltaMat:" << deltaMat << endl;
//                cout << "ptsMat - deltaMat" << (ptsMat - deltaMat) << endl;
//                cout << "worldMat" << worldMat << endl;
//                cout << "worldMat.inv" << worldMat.inv() << endl;
//                camParam.R2W = (ptsMat - deltaMat) ;
//                camParam.R2W.col(0) /= -norm(camParam.R2W.col(0));
//                camParam.R2W.col(1) /= norm(camParam.R2W.col(1));
//                camParam.R2W.col(2) /= -norm(camParam.R2W.col(2));
//                cout << "R2W" << camParam.R2W << endl;
//                cout<<"r.det="<<determinant(camParam.R2W)<<endl;
//                cout << camParam.R2W.col(0).mul(camParam.R2W.col(1)).mul(camParam.R2W.col(2)) << endl;
//                cout << "pts3d.at(4)" << pts3d.at(1) << endl;
//                camParam.T2W = pts3d.at(1);
//                cout << "w2W" << camParam.T2W << endl;
//
//                Mat tmp1;
//                tmp1=camParam.R2W.col(0).mul(camParam.R2W.col(1)).mul(camParam.R2W.col(2));
//                if(tmp1.at<double>(0)+tmp1.at<double>(1)+tmp1.at<double>(2)<0.1){
//                    return true;
//                }
//            }
//            catch (...) {
//                cout << "world CS compute failed" << endl;
//                return false;
//            }
////            worldCSInited = true;
//        }
//    }
    return false;
}

double CVClass::computeReprojectionErrors(
        const vector<vector<Point3f> > &objectPoints,
        const vector<vector<Point2f> > &imagePoints,
        const vector<Mat> &rvecs, const vector<Mat> &tvecs,
        const Mat &cameraMatrix, const Mat &distCoeffs,
        vector<float> &perViewErrors) {
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int) objectPoints.size(); i++) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int) objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

bool CVClass::runCalibration(vector<vector<Point2f> > imagePoints,
                             Size imageSize, Size boardSize, Pattern patternType,
                             float squareSize, float aspectRatio,
                             int flags, Mat &cameraMatrix, Mat &distCoeffs,
                             vector<Mat> &rvecs, vector<Mat> &tvecs,
                             vector<float> &reprojErrs,
                             double &totalAvgErr) {
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, flags/* | CALIB_FIX_K4 | CALIB_FIX_K5*/);
    ///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

void CVClass::saveCameraParams(const string &filename,
                               Size imageSize, Size boardSize,
                               float squareSize, float aspectRatio, int flags,
                               const Mat &cameraMatrix, const Mat &distCoeffs,
                               const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                               const vector<float> &reprojErrs,
                               const vector<vector<Point2f> > &imagePoints,
                               double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nframes" << (int) std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if (flags & CALIB_FIX_ASPECT_RATIO)
        fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty()) {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int) rvecs.size(), 6, rvecs[0].type());
        for (int i = 0; i < (int) rvecs.size(); i++) {
            Mat r = bigmat(Range(i, i + 1), Range(0, 3));
            Mat t = bigmat(Range(i, i + 1), Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if (!imagePoints.empty()) {
        Mat imagePtMat((int) imagePoints.size(), (int) imagePoints[0].size(), CV_32FC2);
        for (int i = 0; i < (int) imagePoints.size(); i++) {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
}

bool CVClass::runAndSave(const string &outputFilename,
                         const vector<vector<Point2f> > &imagePoints,
                         Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                         float aspectRatio, int flags, Mat &cameraMatrix,
                         Mat &distCoeffs, bool writeExtrinsics, bool writePoints) {
    cout << "running" << endl;
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                             aspectRatio, flags, cameraMatrix, distCoeffs,
                             rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if (ok)
        saveCameraParams(outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         writeExtrinsics ? rvecs : vector<Mat>(),
                         writeExtrinsics ? tvecs : vector<Mat>(),
                         writeExtrinsics ? reprojErrs : vector<float>(),
                         writePoints ? imagePoints : vector<vector<Point2f> >(),
                         totalAvgErr);
    return ok;
}

bool CVClass::getImage(void)
{
	cam >> frame;
	return true;
}

bool CVClass::camCalib(void) {
#if 0
    char imgL[10] = "";
    char imgR[10] = "";
    int cnt = 0;
    Mat tFrameL, tFrameR;

    while (1) {
        getImage();
        char c = waitKey(1);
        if (c == 's') {
            sprintf(imgL, "imgL%d.jpg", cnt);
            imwrite(imgL, frame);
            cout << cnt << endl;
            cnt++;
        } else if (c == 'q')
            break;
        vector<Point2f> pointbuf;
        bool found=false;
        found = findChessboardCorners(frame, boardSize, pointbuf);
		if (found) {

			drawChessboardCorners(frame, boardSize, Mat(pointbuf), found);
		}
        resize(frame, tFrameL, Size(640, 480));
        imshow("camL", tFrameL);
    }
#endif
    Size imageSize;
    Mat cameraMatrix, distCoeffs;
    string outputFilenameL = "paramL.xml";
    string inputFilename = "";

    int i;
    size_t nframes;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;

    mode = CAPTURING;

    namedWindow("Image View", 1);

    char imgName[20] = "";
    char outputFilename[20] = "";
    imgs.clear();
	int lr = 0;
   /* for (int lr = 0; lr > 0; lr--)*/ {
        imgs.clear();
        imagePoints.clear();
        mode = CAPTURING;
        nframes = 0;
        for (int i = 0;; i++) {
            if (lr == 0)
                sprintf(imgName, "imgL%d.jpg", i);
            else if (lr == 1)
                sprintf(imgName, "imgR%d.jpg", i);
			cout << imgName << endl;

            Mat tImg;
            tImg = imread(imgName, 1);
            if (!tImg.empty()) {
                imgs.push_back(tImg);
            } else {
                cout << "empty" << endl;
                break;
            }
        }
        nframes = imgs.size();

        if (lr == 0)
            sprintf(outputFilename, "paramL.yaml");
        else if (lr == 1)
            sprintf(outputFilename, "paramR.yaml");
        //calibration
        for (i = 0;; i++) {
            cout << "imagePoints.size()=" << imagePoints.size()
                 << " mode=" << mode << " nframes=" << nframes << " lr=" << lr << " " << "i=" << i << endl;
            Mat view, viewGray;
            bool blink = false;

            if (i == nframes) {
                if (imagePoints.size() > 0) {
                    runAndSave(outputFilename, imagePoints, imageSize,
                               boardSize, pattern, squareSize, aspectRatio,
                               flags, cameraMatrix, distCoeffs,
                               writeExtrinsics, writePoints);
                }
                break;
            }


			imgs.at(i).copyTo(view);;
            imageSize = view.size();

            if (flipVertical)
                flip(view, view, 0);

            vector<Point2f> pointbuf;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);

            /////////////////////pattern/////////////////////
            pattern = CIRCLES_GRID;
            bool found;
            switch (pattern) {
                case CHESSBOARD:
                    found = findChessboardCorners(view, boardSize, pointbuf,
                                                  CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK |
                                                  CALIB_CB_NORMALIZE_IMAGE);
                    break;
                case CIRCLES_GRID:
                    found = findCirclesGrid(view, boardSize, pointbuf);
                    break;
                case ASYMMETRIC_CIRCLES_GRID:
                    found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
                    break;
                default:
                    return fprintf(stderr, "Unknown pattern type\n"), -1;
            }
            ////////////////////////////////////////////////////

            // improve the found corners' coordinate accuracy
			if (pattern == CHESSBOARD && found)
			{
				Mat viewGray;
				cvtColor(frame, viewGray, COLOR_BGR2GRAY);
				cornerSubPix(viewGray, pointbuf, Size(11, 11),
					Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			}


            if (mode == CAPTURING && found &&
                (!capture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC)) {
                imagePoints.push_back(pointbuf);
                prevTimestamp = clock();
                blink = capture.isOpened();
            }
            cout << pointbuf.size() << endl;
            if (found)
                drawChessboardCorners(view, boardSize, Mat(pointbuf), found);


            string msg = mode == CAPTURING ? "100/100" :
                         mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
            int baseLine = 0;
            Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
            Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

            if (mode == CAPTURING) {
                if (undistortImage)
                    msg = format("%d/%d Undist", (int) imagePoints.size(), nframes);
                else
                    msg = format("%d/%d", (int) imagePoints.size(), nframes);
            }

            putText(view, msg, textOrigin, 1, 1,
                    mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

            if (blink)
                bitwise_not(view, view);

            if (mode == CALIBRATED && undistortImage) {
                Mat temp = view.clone();
                undistort(temp, view, cameraMatrix, distCoeffs);
            }

            imshow("Image View", view);
            char key = (char) waitKey(capture.isOpened() ? 1 : 1);

            if (key == 27)
                break;

            if (key == 'u' && mode == CALIBRATED)
                undistortImage = !undistortImage;

            if (capture.isOpened() && key == 'g') {
                mode = CAPTURING;
                imagePoints.clear();
            }

            if (mode == CAPTURING && imagePoints.size() >= (unsigned) nframes) {
                if (runAndSave(outputFilename, imagePoints, imageSize,
                               boardSize, pattern, squareSize, aspectRatio,
                               flags, cameraMatrix, distCoeffs,
                               writeExtrinsics, writePoints))
                    mode = CALIBRATED;
                else
                    mode = DETECTION;
                if (!capture.isOpened())
                    break;
            }
        }  //calibration

        //show undistorted
        if (!capture.isOpened() && showUndistorted) {
            imgsUndistorted.clear();
            cout << "undistorted  ";
            Mat view, map1, map2;
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                    imageSize, CV_16SC2, map1, map2);

			for (int i = 0;i < imgs.size();i++) {
                Mat rview;
				cout << "check:" << i << endl;
				view = imgs.at(i);
				imshow("fuck", view);
				//undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
				remap(view, rview, map1, map2, INTER_LINEAR);
				imgsUndistorted.push_back(rview);
                imshow("imgName", rview);
                char c = (char)waitKey(1);
				if (c == 27 || c == 'q' || c == 'Q')
					return 0;
			}
        }

        cout << lr << endl;

    }
    waitKey(0);
    return false;
}

