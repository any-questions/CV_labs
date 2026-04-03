#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main() {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::Mat markerImage;
    //generateImageMarker вместо drawMarker
    cv::aruco::generateImageMarker(dictionary, 23, 400, markerImage, 1);

    cv::putText(markerImage, "ArUco Marker ID: 23",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(128), 2);

    cv::imshow("ArUco Marker", markerImage);
    std::cout << "Press any key to close..." << std::endl;
    cv::waitKey(0);

    cv::imwrite("aruco_marker_23.png", markerImage);
    std::cout << "Saved as aruco_marker_23.png" << std::endl;

    return 0;
}