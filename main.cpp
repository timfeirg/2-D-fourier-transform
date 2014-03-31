//
//  main.cpp
//  DFT
//
//  Created by Tim Feirg on 3/22/14.
//  Copyright (c) 2014 Tim Feirg. All rights reserved.
//

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

void quadrantShift(Mat& src, Mat& dst);
//Mat quadrantShift(Mat& src);

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
    Mat visualDFT( Mat& dft_result );
    Mat createGaussianFilter( Size size_of_filter, double sigma, bool highpass_flag);
    Mat createIdealFilter( Size size_of_filter, float threthold, bool highpass_flag );
    
    help(argv[0]);
    
    // Load user specified image, if none, use lena as default target
    Mat src = (argc>=2) ? imread(argv[ argc - 1 ], CV_LOAD_IMAGE_GRAYSCALE) :
    imread("/Users/timfeirg/Google Drive/OpenCV/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif",
           CV_LOAD_IMAGE_GRAYSCALE);
    Mat dft_container,result;
    
    DFT(src, dft_container);
    
    // compute the DFT of image and visualize
    imshow("visualized frequency spectrum (before filtering)", visualDFT(dft_container));
    
    if (strcmp(argv[1], "--visual") == 0) return 1;
    
    // apply gaussian filter with user-specified sigma to the image, visualize & show the frequency spectrum
    else if (strcmp(argv[1], "--gaussian") == 0) {
        
        Mat gaussian_filter;
        double sigma;
        // user specify sigma for creating the filter
        cout<<"specify sigma, any non-positive value signals the program to use default sigma:"<<endl;
        cin>>sigma;
        

        gaussian_filter = (strcmp(argv[2], "--highpass") == 0 ) ?
        createGaussianFilter(dft_container.size(), sigma, true) :
        createGaussianFilter(dft_container.size(), sigma, false);
        
        // multiply the frequency spectrum with gaussian filter, pixel wise, sort of
        mulSpectrums(dft_container, gaussian_filter, dft_container, DFT_ROWS);
        inverseDFT(dft_container, result);
        
        // visualize things
        imshow("spectrum after ideal filtering", visualDFT(dft_container));
        imshow("image after ideal filtering", result);
        waitKey();
        
    }
    
    // implement ideal high/low pass filtering
    else if (strcmp(argv[1], "--ideal") == 0) {
        
        Mat filter = (strcmp(argv[2], "--highpass") == 0) ?
        createIdealFilter(dft_container.size(), 0, 1) : createIdealFilter(dft_container.size(), 0, 0);
        
        cout<<dft_container.type()<<endl;
        mulSpectrums(dft_container, filter, dft_container, DFT_ROWS);
        
        // perform inverse dft to observe the result of such filtering
        inverseDFT(dft_container, result);
        
        // visualize things
        imshow("spectrum after filtering", visualDFT(dft_container));
        imshow("after filtering", result);
        waitKey();
        
    }
    
    return 0;
}

Mat createIdealFilter( Size size_of_filter, float threthold, bool highpass_flag ) {
    
    Mat filter(size_of_filter, CV_32F, Scalar::all(0));
    
    Point ideal_center = Point_<uint>(size_of_filter.width / 2, size_of_filter.height / 2);
    int idean_radius = ( size_of_filter.height >= size_of_filter.width ) ?
    size_of_filter.height / 4 : size_of_filter.width / 4;
    
    circle(filter, ideal_center, idean_radius, Scalar::all(1), -1);

    Mat to_merge[] = { filter, filter };
    merge(to_merge, 2, filter);
    
    // if user specified ideal highpass filter, then we'd invert the filter
    if (highpass_flag) {
        filter = 255 - filter;
    }
    
    quadrantShift(filter, filter);
    return filter;
}

Mat createGaussianFilter( Size size_of_filter, double sigma, bool highpass_flag ) {
    
    Mat gaussian_filter = Mat(size_of_filter, CV_32F),
    filter_x = getGaussianKernel(size_of_filter.height, sigma, CV_32F),
    filter_y = getGaussianKernel(size_of_filter.width, sigma, CV_32F);
    
    // this will create filter as Mat object of which size is x*y
    gaussian_filter = filter_x * filter_y.t();
    normalize(gaussian_filter, gaussian_filter, 0, 1, CV_MINMAX);
    
    if (highpass_flag == true) {
        gaussian_filter = 1 - gaussian_filter;
    }
    
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
