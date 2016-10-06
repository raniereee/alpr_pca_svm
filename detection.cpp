#include "detection.h"

#define NUM_PRINCIPAL_COMP 12

void DetectPlates::readPCAPlane()
{
    FileStorage fs(pcafilename.c_str(),FileStorage::READ);
    fs["vectors"]    >> pca_.eigenvectors ;
    fs["values"]     >> pca_.eigenvalues ;
    fs["mean"]       >> pca_.mean ;
    fs.release();

    #if 0
    // Mean face:
    printf("Carregou o PCA: %s\n\n", pcafilename.c_str());
    namedWindow("avg", 1);

    imshow("avg", pca_.mean.reshape(1, 150));
    waitKey(0);
    #endif
}

void DetectPlates::setFilename(string s) {
    pcafilename=s;

    readPCAPlane();
}

vector<Rect> DetectPlates::DetectPlate(Mat input, CvSVM * svmClassifier){
    vector<Rect> plates;
    int i, response;
    Mat p;
    Rect roi;

    imshow ("original", input);

    Mat xx(1, input.rows * input.cols, CV_32FC1);
    Mat img_row = input.reshape(1, 1);
    img_row.convertTo(xx, CV_32FC1, 1/255.);

    Mat dataprojected(1,  NUM_PRINCIPAL_COMP, CV_32FC1);
    pca_.project(xx, dataprojected);

    waitKey(0);


    /*
    for(i=0; i < 10; i++)
    {

        response = (int) svmClassifier->predict( p );
        if(response == 1)
            plates.push_back(roi);
    }
    */
    return plates;
}
