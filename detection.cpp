#include "detection.h"

void DetectPlates::setFilename(string s) {
        filename=s;
}


vector<Rect> DetectPlates::DetectPlate(Mat input, CvSVM svmClassifier){
    vector<Rect> plates;
    int i, response;
    Mat p;
    Rect roi;

    for(i=0; i < 10; i++)
    {

        response = (int) svmClassifier.predict( p );
        if(response == 1)
            plates.push_back(roi);
    }


    return plates;
}
