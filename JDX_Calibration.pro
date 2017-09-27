QT += core
QT -= gui

CONFIG += c++11

TARGET = JDX_Calibration
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    CVCalib.cpp \
    JDX_Calibrate.cpp

INCLUDEPATH+= /home/lxy-ubuntu/opencv-3.2.0/include \
              /home/lxy-ubuntu/opencv-3.2.0/include/opencv\
              /home/lxy-ubuntu/opencv-3.2.0/include/opencv2

 LIBS+=-L/usr/local/lib\
        -lopencv_calib3d\
        -lopencv_core\
        -lopencv_features2d\
        -lopencv_flann\
        -lopencv_highgui\
        -lopencv_imgcodecs\
        -lopencv_imgproc\
        -lopencv_ml\
        -lopencv_objdetect\
        -lopencv_videostab\
        -lopencv_video\
        -lopencv_videoio\
        -lopencv_superres\
        -lopencv_stitching\
        -lopencv_shape\
        -lopencv_photo\
        -lpthread\

HEADERS += \
    CVClass.h
