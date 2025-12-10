#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


int mixer(int src1, int src2, float k) {
    return k * src1 + (1 - k) * src2;
}

int main() {
    cv::Mat img1 = cv::imread("images/gtr_r34.jpg");
    cv::Mat img2 = cv::imread("images/wrx_sti.jpg");
    cv::resize(img1, img1, cv::Size(img2.cols, img2.rows), cv::INTER_LINEAR);

    float k = 0.5;

    cv::Mat blend_img;
    blend_img.create(img2.rows, img2.cols, CV_8UC3);

    for (int rows = 0; rows < img2.rows; rows++)
        for (int cols = 0; cols < img2.cols; cols++) {
            blend_img.at<cv::Vec3b>(rows, cols)[0] = mixer(img1.at<cv::Vec3b>(rows, cols)[0], img2.at<cv::Vec3b>(rows, cols)[0], k);
            blend_img.at<cv::Vec3b>(rows, cols)[1] = mixer(img1.at<cv::Vec3b>(rows, cols)[1], img2.at<cv::Vec3b>(rows, cols)[1], k);
            blend_img.at<cv::Vec3b>(rows, cols)[2] = mixer(img1.at<cv::Vec3b>(rows, cols)[2], img2.at<cv::Vec3b>(rows, cols)[2], k);
        }

    cv::imwrite("images/result.png", blend_img);
    cv::imshow("blended", blend_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}