#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

// Change to the number of principal components you want:
#define NUM_PRINCIPAL_COMP 2
#define DEBUG 0

Mat normalize(const Mat& src) {
    Mat srcnorm;
    normalize(src, srcnorm, 0, 255, NORM_MINMAX, CV_8UC1);
    return srcnorm;
}

vector<Mat> read_noplates_images()
{
    vector<Mat> db;

    Mat image;
    int i = 0;
    while(1)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << "../noplates/" << i++ << ".jpeg";
        //cout << "Image: " << ss.str() << "\n";
        image = imread(ss.str(), 0);
        if(!image.data )  break;

        db.push_back(imread(ss.str(), 0));
    }

    printf("Number of no plates: %d\n", i);
    return db;
}


vector<Mat> read_plates_images()
{
    vector<Mat> db;

    Mat image;
    int i = 0;
    while(1)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << "../plates/" << i++ << ".jpeg";
        //cout << "Image: " << ss.str() << "\n";
        image = imread(ss.str(), 0);
        if(!image.data )  break;

        db.push_back(imread(ss.str(), 0));
    }

    printf("Number of plates: %d\n", i);
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


void PCA_write(FileStorage& fs, PCA pca)
{
    CV_Assert( fs.isOpened() );

    //fs << "name"    << "PCA";
    fs << "vectors" << pca.eigenvectors;
    fs << "values"  << pca.eigenvalues;
    fs << "mean"    << pca.mean;
}

int main(int argc, char *argv[]) {
    
    vector<Mat> db;
    PCA pca_plates, pca_noplates;

    (void) argc;
    (void) argv;

    /* Plates set processing */
    db = read_plates_images();
    pca_plates = do_pca(db);

    /* Save the PCA base */
    FileStorage fs("PCA_PLANE.xml", FileStorage::WRITE);
    PCA_write(fs, pca_plates);
    fs.release();

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
    //pca_noplates = do_pca(db);

    mat = get_matrix_row_image(db);
    Mat features_noplate_projected(mat.rows, NUM_PRINCIPAL_COMP, CV_32FC1);
    for(int i = 0; i < NUM_PRINCIPAL_COMP; i++)
    {
        pca_plates.project(mat.row(i), features_noplate_projected.row(i));
        trainingImages.push_back(features_noplate_projected.row(i));
	trainingLabels.push_back(0);
    }

    Mat classes;
    Mat trainingData;

    Mat(trainingImages).copyTo(trainingData);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);

    FileStorage fss("PCA4SVM.xml", FileStorage::WRITE);
    fss << "TrainingData" << trainingData;
    fss << "classes" << classes;
    fss.release();
}

