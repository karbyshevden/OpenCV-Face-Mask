package com.karbyshev.my4;

import android.content.Context;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;

import static org.opencv.core.Core.addWeighted;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.split;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2GRAY;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private JavaCameraView javaCameraView;
    private Mat mask, testMat;
    private CascadeClassifier cascadeClassifier;
    private int mWidth;
    private int mHeight;
    private int mAbsoluteFaceSize = 0;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
//                    Log.i(TAG, "OpenCV loaded successfully");
                    javaCameraView.enableView();
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                    FileOutputStream os = null;
                    try {
                        os = new FileOutputStream(mCascadeFile);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();
                    } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }


                    cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = (JavaCameraView) findViewById(R.id.cameraView);
        javaCameraView.setVisibility(View.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        try {
            File file = new File(Environment.getExternalStorageDirectory(), "five.jpg");

            mask = Imgcodecs.imread(file.getPath(), -1);

        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, "File NOT find");
        }

        long llong= 11l;
        testMat = new Mat(llong);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV loaded");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCV NOT loaded!!!");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        if (mWidth != width || mHeight != height) {
            mWidth = width;
            mHeight = height;
        }
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mRgba = inputFrame.rgba();
        Mat mGrey = inputFrame.gray();
        MatOfRect faces = new MatOfRect();
        int height = mGrey.rows();
        if (Math.round(height * 0.2) > 0) {
            mAbsoluteFaceSize = (int) Math.round(height * 0.2);
        }
        cascadeClassifier.detectMultiScale(mGrey, faces, 1.1, 2, 2,//Objdetect.CASCADE_SCALE_IMAGE,
                new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Point center = new Point(facesArray[i].x + facesArray[i].width / 2, facesArray[i].y + facesArray[i].height / 2);
            mRgba = putMask(mRgba, center, new Size(facesArray[i].width, facesArray[i].height));
//            rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }

        return mRgba;
    }

    private Mat putMask(Mat src, Point center, Size face_size) {

        Log.e(TAG, "src at the START of putMask: " + src.toString());
        //mask : masque chargé depuis l'image
        Mat mask_resized = new Mat(); //masque resizé
        Mat src_roi = new Mat(); //ROI du visage croppé depuis la preview
        Mat roi_gray = new Mat();

        resize(mask, mask_resized, face_size);

        // ROI selection
        Rect roi = new Rect((int) (center.x - face_size.width / 2), (int) (center.y - face_size.height / 2), (int) face_size.width, (int) face_size.height);

        src.submat(roi).copyTo(src_roi);

        Log.d(TAG, "MASK SRC1 :" + src_roi.size());

        // to make the white region transparent
        Mat mask_grey = new Mat(); //greymask
        Mat roi_rgb = new Mat();
        cvtColor(mask_resized, mask_grey, COLOR_BGRA2GRAY);
        threshold(mask_grey, mask_grey, 230, 255, THRESH_BINARY_INV);

        ArrayList<Mat> maskChannels = new ArrayList<>(3);
        ArrayList<Mat> result_mask = new ArrayList<>(3);
        result_mask.add(new Mat());
        result_mask.add(new Mat());
        result_mask.add(new Mat());
//        result_mask.add(new Mat());

        split(mask_resized, maskChannels);

        bitwise_and(maskChannels.get(0), mask_grey, result_mask.get(0));
        bitwise_and(maskChannels.get(1), mask_grey, result_mask.get(1));
        bitwise_and(maskChannels.get(2), mask_grey, result_mask.get(2));
//        bitwise_and(maskChannels.get(3), mask_grey, result_mask.get(3));

        merge(result_mask, roi_gray);

        bitwise_not(mask_grey, mask_grey);

        ArrayList<Mat> srcChannels = new ArrayList<>(3);
        split(src_roi, srcChannels);
        bitwise_and(srcChannels.get(0), mask_grey, result_mask.get(0));
        bitwise_and(srcChannels.get(1), mask_grey, result_mask.get(1));
        bitwise_and(srcChannels.get(2), mask_grey, result_mask.get(2));
//        bitwise_and(srcChannels.get(3), mask_grey, result_mask.get(3));

        merge(result_mask, roi_rgb);

        addWeighted(roi_gray, 1, roi_rgb, 1, 0, roi_rgb);

//        roi_rgb.copyTo(new Mat(src, roi));
        src = new Mat(src, roi);
        roi_rgb.copyTo(src);

        Log.i(TAG, "src at the END of putMask: " + src.toString() + "_______________________");
        return src;
    }
}
