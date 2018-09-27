#include <jni.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

void detect(Mat& frame);
Mat mask;// = imread("/storage/emulated/0/data/5.jpg");
Mat putMask(Mat src,Point center,Size face_size);

extern "C" JNIEXPORT void

JNICALL
Java_com_karbyshev_my4_MainActivity_stringFromJNI(
        JNIEnv *env,
        jclass ,
        jlong addrRgba/* this */) {

    Mat& frame = *(Mat*)addrRgba;

    mask = imread("/storage/emulated/0/data/5.jpg");

    detect(frame);
}

void detect(Mat& frame) {

    String face_cascade_name = "/storage/emulated/0/data/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return ; };

    std::vector<Rect> faces;
//    Mat frame_gray;
//
//    cvtColor( frame, frame_gray, CV_BGR2GRAY );
//    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        frame = putMask(frame, center, Size( faces[i].width, faces[i].height));
//        Mat faceROI = frame_gray( faces[i] );
    }
}

Mat putMask(Mat src,Point center,Size face_size) {
    Mat mask1, src1;
    resize(mask, mask1, face_size);

    // ROI selection
    Rect roi(center.x - face_size.width / 2, center.y - face_size.width / 2, face_size.width,
             face_size.width);
    src(roi).copyTo(src1);

    // to make the white region transparent
    Mat mask2, m, m1;
    cvtColor(mask1, mask2, CV_BGR2GRAY);
    threshold(mask2, mask2, 230, 255, CV_THRESH_BINARY_INV);

    std::vector<Mat> maskChannels(3), result_mask(3);
    split(mask1, maskChannels);
    bitwise_and(maskChannels[0], mask2, result_mask[0]);
    bitwise_and(maskChannels[1], mask2, result_mask[1]);
    bitwise_and(maskChannels[2], mask2, result_mask[2]);
    merge(result_mask, m);         //    imshow("m",m);

    mask2 = 255 - mask2;
    std::vector<Mat> srcChannels(3);
    split(src1, srcChannels);
    bitwise_and(srcChannels[0], mask2, result_mask[0]);
    bitwise_and(srcChannels[1], mask2, result_mask[1]);
    bitwise_and(srcChannels[2], mask2, result_mask[2]);
    merge(result_mask, m1);        //    imshow("m1",m1);

    addWeighted(m, 1, m1, 1, 0, m1);    //    imshow("m2",m1);

    m1.copyTo(src(roi));

    return src;
}
