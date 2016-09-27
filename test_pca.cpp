#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

// Change to the number of principal components you want:
#define NUM_PRINCIPAL_COMP 12
#define DEBUG 1

Mat normalize(const Mat& src) {
    Mat srcnorm;
    normalize(src, srcnorm, 0, 255, NORM_MINMAX, CV_8UC1);
    return srcnorm;
}

vector<Mat> read_noplates_images()
{
    vector<Mat> db;

    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/0.jpg",0));
    db.push_back(imread("noplates/1.jpeg",0));
    db.push_back(imread("noplates/1.jpeg",0));
    db.push_back(imread("noplates/1.jpeg",0));
    db.push_back(imread("noplates/1.jpeg",0));
    db.push_back(imread("noplates/1.jpeg",0));
    db.push_back(imread("noplates/1.jpeg",0));

    return db;
}


vector<Mat> read_plates_images()
{
    vector<Mat> db;

    db.push_back(imread("plates/0.jpeg",0));
    db.push_back(imread("plates/1.jpeg",0));
    db.push_back(imread("plates/2.jpeg",0));
    db.push_back(imread("plates/3.jpeg",0));
    db.push_back(imread("plates/4.jpeg",0));
    db.push_back(imread("plates/5.jpeg",0));
    db.push_back(imread("plates/6.jpeg",0));
    db.push_back(imread("plates/7.jpeg",0));
    db.push_back(imread("plates/8.jpeg",0));
    db.push_back(imread("plates/9.jpeg",0));
    db.push_back(imread("plates/10.jpeg",0));
    db.push_back(imread("plates/11.jpeg",0));
    db.push_back(imread("plates/12.jpeg",0));
    db.push_back(imread("plates/13.jpeg",0));
    db.push_back(imread("plates/14.jpeg",0));
    db.push_back(imread("plates/15.jpeg",0));

    return db;
}

Mat get_matrix_row_image(vector<Mat> db)
{
    int total = db[0].rows * db[0].cols;

    // build matrix (column)
    Mat mat(db.size(), total, CV_32FC1);
    for(int i = 0; i < db.size(); i++) {
        Mat X = mat.row(i);
        db[i].reshape(1, 1).row(0).convertTo(X, CV_32FC1, 1/255.);
    }

    return mat;
}

PCA do_pca(vector<Mat> db)
{
    Mat mat = get_matrix_row_image(db);

    // Do the PCA:
    PCA pca(mat, Mat(), CV_PCA_DATA_AS_ROW, NUM_PRINCIPAL_COMP);

    #if DEBUG
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
    waitKey(0);
    #endif

    return pca;
}

int main(int argc, char *argv[]) {
    
    vector<Mat> db;
    PCA pca_plates, pca_noplates;

    /* Plates set processing */
    db = read_plates_images();
    pca_plates = do_pca(db);

    Mat mat = get_matrix_row_image(db);

    Mat trainingImages;
    vector<int> trainingLabels;
    Mat features_projected(mat.rows, NUM_PRINCIPAL_COMP, CV_32FC1);
    for(int i = 0; i < NUM_PRINCIPAL_COMP; i++)
    {
        pca_plates.project(mat.row(i), features_projected.row(i));
        trainingImages.push_back(features_projected.row(i));
	trainingLabels.push_back(1);
    }

    /* No plates set processing */
    db = read_noplates_images();
    pca_noplates = do_pca(db);

    mat = get_matrix_row_image(db);
    Mat features_noplate_projected(mat.rows, NUM_PRINCIPAL_COMP, CV_32FC1);
    for(int i = 0; i < NUM_PRINCIPAL_COMP; i++)
    {
        pca_plates.project(mat.row(i), features_noplate_projected.row(i));
        trainingImages.push_back(features_noplate_projected.row(i));
	trainingLabels.push_back(0);
    }


    #if 0
    Mat dataprojected(mat.rows, NUM_PRINCIPAL_COMP, CV_32FC1);
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

    //Mat img_test = imread("plates/0.jpeg",0);
    //Mat img_test = imread("teste/0.jpeg",0);
    Mat img_test = imread("test/1.jpeg",0);
    //Mat img_test = imread("teste/n1.jpeg",0);
    //Mat img_test = imread("teste/n2.jpeg",0);
    imshow ("original", img_test);

    Mat xx(1, img_test.rows * img_test.cols, CV_32FC1);
    Mat img_row = img_test.reshape(1, 1);
    img_row.convertTo(xx, CV_32FC1, 1/255.);

    Mat dataprojected(1,  NUM_PRINCIPAL_COMP, CV_32FC1);
    pca_plates.project(xx, dataprojected);

    Mat datareconstructed (1, img_test.cols, img_test.type());
    pca_plates.backProject(dataprojected, datareconstructed);
    imshow ("reconstruido", datareconstructed.reshape(1, img_test.rows) );

    waitKey(0);
}

