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
    << "\t-v" << "\tvisualized DFT spectrum"<< endl
    << "\t-f" << "\tfilter" <<endl
    << "filter;" <<endl
    << "------------------------------------------"<< endl;
}

int main(int argc, const char * argv[])
{
    void visualDFT( Mat& dft_result, Mat& dst );
    void DFT( Mat& src, Mat& dst );
    void inverseDFT( Mat& dft_result, Mat& dst );
    
    help(argv[0]);
    
    // Load user specified image, if none, use lena as default target
    Mat src = (argc>=2) ? imread(argv[ argc - 1 ]) : imread(
                                                   "/Users/timfeirg/Google Drive/OpenCV/DIP3E_Original_Images_CH02/Fig0222(a)(face).tif",
                                                   CV_LOAD_IMAGE_GRAYSCALE),
    dst;
    
    imshow("target", dst);
    waitKey();
    
    return 0;
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

void quadrantShift( Mat& src, Mat& dst) {
    
    dst = src;
    src = src(Rect(0, 0, src.cols & -2, src.rows & -2));
    int cx = src.cols/2;
    int cy = src.rows/2;
    
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
