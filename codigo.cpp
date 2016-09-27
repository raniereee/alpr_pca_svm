#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

Mat normalize(const Mat& src) {
    Mat srcnorm;
    normalize(src, srcnorm, 0, 255, NORM_MINMAX, CV_8UC1);
    return srcnorm;
}

int main(int argc, char *argv[]) {
    vector<Mat> db;

    // load greyscale images (these are from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
    db.push_back(imread("placas/0.jpeg",0));
    db.push_back(imread("placas/1.jpeg",0));
    db.push_back(imread("placas/2.jpeg",0));
    db.push_back(imread("placas/3.jpeg",0));
    db.push_back(imread("placas/4.jpeg",0));
    db.push_back(imread("placas/5.jpeg",0));
    db.push_back(imread("placas/6.jpeg",0));
    db.push_back(imread("placas/7.jpeg",0));
    db.push_back(imread("placas/8.jpeg",0));
    db.push_back(imread("placas/9.jpeg",0));
    db.push_back(imread("placas/10.jpeg",0));
    db.push_back(imread("placas/11.jpeg",0));
    db.push_back(imread("placas/12.jpeg",0));
    db.push_back(imread("placas/13.jpeg",0));
    db.push_back(imread("placas/14.jpeg",0));
    db.push_back(imread("placas/15.jpeg",0));

    int total = db[0].rows * db[0].cols;

    // build matrix (column)
    //Mat mat(total, db.size(), CV_32FC1);
    Mat mat(db.size(), total, CV_32FC1);
    for(int i = 0; i < db.size(); i++) {
        Mat X = mat.row(i);
        db[i].reshape(1, 1).row(0).convertTo(X, CV_32FC1, 1/255.);
    }

    // Change to the number of principal components you want:
    int numPrincipalComponents = 5;

    // Do the PCA:
    PCA pca(mat, Mat(), CV_PCA_DATA_AS_ROW, numPrincipalComponents);

    // Create the Windows:
    namedWindow("avg", 1);
    namedWindow("pc1", 1);
    namedWindow("pc2", 1);
    namedWindow("pc3", 1);

    // Mean face:
    imshow("avg", pca.mean.reshape(1, db[0].rows));

    // First three eigenfaces:
    imshow("pc1", normalize(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", normalize(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", normalize(pca.eigenvectors.row(2)).reshape(1, db[0].rows));

    #if 0
    Mat dataprojected(mat.rows, numPrincipalComponents, CV_32FC1);
    for(int i=0; i < db.size(); i++)
    {
        pca.project(mat.row(i), dataprojected.row(i));
    }

    //Backproject to reconstruct images
    Mat datareconstructed (mat.rows, mat.cols, mat.type());
    for(int i=0; i<db.size(); i++)
    {
        pca.backProject(dataprojected.row(i), datareconstructed.row(i) );
    }

    for(int i=0; i<db.size(); i++)
    {
        imshow ("reconstruido", datareconstructed.row(i).reshape(1, db[0].rows) );
        waitKey();
    }
    #endif

    //Mat img_test = imread("placas/0.jpeg",0);
    //Mat img_test = imread("teste/0.jpeg",0);
    Mat img_test = imread("teste/1.jpeg",0);
    //Mat img_test = imread("teste/n1.jpeg",0);
    //Mat img_test = imread("teste/n2.jpeg",0);
    imshow ("original", img_test);

    Mat xx(1, img_test.rows * img_test.cols, CV_32FC1);
    Mat img_row = img_test.reshape(1, 1);
    img_row.convertTo(xx, CV_32FC1, 1/255.);

    Mat dataprojected(1,  numPrincipalComponents, CV_32FC1);
    pca.project(xx, dataprojected);

    Mat datareconstructed (1, img_test.cols, img_test.type());
    pca.backProject(dataprojected, datareconstructed);
    imshow ("reconstruido", datareconstructed.reshape(1, img_test.rows) );

    waitKey(0);
}

