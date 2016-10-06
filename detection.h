#ifndef DETECTION_H
#define DETECTION_H

#include <string.h>
#include <vector>

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

using namespace std;
using namespace cv;

class DetectPlates{
    public:
        vector<Rect> DetectPlate(Mat input, CvSVM * svmClassifier);
        void readPCAPlane();
        string pcafilename;
        PCA pca_;
        void setFilename(string f);
        bool saveRegions;
        bool showSteps;
    //private:
    //    bool verifySizes(RotatedRect mr);
};

#endif
