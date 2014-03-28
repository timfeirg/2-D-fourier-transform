//
//  main.cpp
//  DFT
//
//  Created by Tim Feirg on 3/22/14.
//  Copyright (c) 2014 Tim Feirg. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void quadrantShift(Mat& src, Mat& dst);

static void help(const char* progName) {
    
    cout << "------------------------------------------"<< endl
    <<  "This program demonstrated the use of DFT, and some basic frequency domain filtering. " << endl
    <<  "Usage:"                                                                      << endl
    << progName << " [option] [optional filter specification] [image_name -- default lena.jpg] " << endl
    << "options:" << endl
    << "\t--visual" << "\tvisualized DFT spectrum"<< endl
    << "\t-gaussian" << "\tuse gaussian filter"<< endl
    << "------------------------------------------"<< endl;
}

int main(int argc, const char * argv[])
{
    void visualDFT( Mat& dft_result, Mat& dst );
    void DFT( Mat& src, Mat& dst );
    void inverseDFT( Mat& dft_result, Mat& dst );
    void onGaussianTrackbar( int, void*);
    Mat visualDFT( Mat& dft_result );
    Mat createGaussianFilter( Size size_of_filter, double sigma );
    
    help(argv[0]);
    
    // Load user specified image, if none, use lena as default target
    Mat src = (argc>=2) ? imread(argv[ argc - 1 ], CV_LOAD_IMAGE_GRAYSCALE) :
    imread("/Users/timfeirg/Google Drive/OpenCV/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif",
           CV_LOAD_IMAGE_GRAYSCALE);
    Mat dst;
    
    if (strcmp(argv[1], "--visual") == 0) {
        
        DFT(src, dst);
        visualDFT(dst, dst);
        
    }
    
    else if (strcmp(argv[1], "--gaussian") == 0) {
        Mat dft_container;
        // computes the fourier transformation of source image
        DFT(src, dft_container);
        // before filtering
        imshow("spectrum before filtering", visualDFT(dft_container));
        // user specify sigma for creating the filter
        double sigma;
        cout<<"specify sigma:"<<endl;
        cin>>sigma;

        Mat gaussian_filter = createGaussianFilter(src.size(), sigma);
        
        if (strcmp(argv[2], "-h") == 0 ) {
            gaussian_filter = 1.0 - gaussian_filter;
        }
        
        // multiply the frequency spectrum with gaussian filter, pixel wise, sort of
        mulSpectrums(dft_container, gaussian_filter, dst, DFT_ROWS);
        imshow("spectrum after filtering", visualDFT(dst));
        inverseDFT(dst, dst);
    }
    imshow("original image", src);
    imshow("output", dst);
    waitKey();
    return 0;
}

void onGaussianTrackbar( int, void* ) {
    
}

Mat createGaussianFilter( Size size_of_filter, double sigma ) {
    
    Mat gaussian_filter = Mat(size_of_filter, CV_32F),
    filter_x = getGaussianKernel(size_of_filter.height, sigma, CV_32F),
    filter_y = getGaussianKernel(size_of_filter.width, sigma, CV_32F);
    // this will create filter as Mat object of which size is x*y
    gaussian_filter = filter_x * filter_y.t();
    normalize(gaussian_filter, gaussian_filter, 0, 1, CV_MINMAX);
    
    Mat to_merge[] = {gaussian_filter, gaussian_filter};
    merge(to_merge, 2, gaussian_filter);
    // the filter is used to process spetrums before quadrant shift, so:
    quadrantShift(gaussian_filter, gaussian_filter);
    return gaussian_filter;
    
}

void inverseDFT(Mat& dft_result, Mat& dst) {
    
    dft(dft_result, dst, DFT_INVERSE|DFT_REAL_OUTPUT);
    normalize(dst, dst, 0, 1, CV_MINMAX);
    
}

void DFT(Mat& src, Mat& dst) {
    
    // expand the source image to optimal size for dft
    copyMakeBorder(src, dst,
                   0, getOptimalDFTSize( src.rows ) - src.rows,
                   0, getOptimalDFTSize( src.cols ) - src.cols,
                   BORDER_ISOLATED);
    
    // create a plane containing 2 mat layer to form a 2-channel mat object
    Mat planes[] = {Mat_<float>(dst), Mat::zeros(dst.size(), CV_32F)};
    // this is the 2-channel object that I was talking about
    merge(planes, 2, dst);
    // dft result will be stored in dst, in which two channels holds separately real and imaginary components
    dft(dst, dst);
    
}

void visualDFT(  Mat& dft_result, Mat& dst ) {
    
    // create a plane containing 2 mat layer to form a 2-channel mat object
    Mat planes[2];
    // in order to calculate the magnitude, we'll have to split the image by channel in order to obtain each component
    split(dft_result, planes);
    magnitude(planes[0], planes[1], dst);
    
    // switch to logarithmic scale
    dst += Scalar::all(1);
    log(dst, dst);
    
    normalize(dst, dst,0,1,CV_MINMAX);
    //    cout<<dst<<"dst end";
    quadrantShift(dst, dst);
    
}

Mat visualDFT(  Mat& dft_result ) {
    Mat dst;
    // create a plane containing 2 mat layer to form a 2-channel mat object
    Mat planes[2];
    // in order to calculate the magnitude, we'll have to split the image by channel in order to obtain each component
    split(dft_result, planes);
    magnitude(planes[0], planes[1], dst);
    
    // switch to logarithmic scale
    dst += Scalar::all(1);
    log(dst, dst);
    
    normalize(dst, dst,0,1,CV_MINMAX);
    //    cout<<dst<<"dst end";
    quadrantShift(dst, dst);
    return dst;
}

void quadrantShift( Mat& src, Mat& dst) {
    
    dst = src;
    src = src(Rect(0, 0, src.cols & -2, src.rows & -2));
    uint cx = src.cols/2;
    uint cy = src.rows/2;
    
    Rect q0 = Rect(0, 0, cx, cy);   // Top-Left - Create a ROI per quadrant
    Rect q1 = Rect(cx, 0, cx, cy);  // Top-Right
    Rect q2 = Rect(0, cy, cx, cy);  // Bottom-Left
    Rect q3 = Rect(cx, cy, cx, cy); // Bottom-Right
    
    Mat temp;   // creating a temporary Mat object to protect the quadrant, in order to handle the situation where src = dst
    
    src(q0).copyTo(temp);   // preserve q0 section
    src(q3).copyTo(dst(q0));
    temp.copyTo(dst(q3));   // swap q0 and q3
    
    src(q1).copyTo(temp);
    src(q2).copyTo(dst(q1));
    temp.copyTo(dst(q2));
    
}
