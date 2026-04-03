#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <csignal>

cv::VideoCapture* globalCap = nullptr;

void signalHandler(int) {
    if (globalCap) globalCap->release();
    cv::destroyAllWindows();
    exit(0);
}

bool loadCalibration(const std::string& path, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs.release();
    return true;
}

cv::Point2f projectPoint(const cv::Point3f& p3d,
                          const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                          const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    std::vector<cv::Point3f> pts = {p3d};
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs, pts2d);
    return pts2d[0];
}

void drawCube(cv::Mat& image, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
              const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, float markerSize) {
    float s = markerSize;
    float h = markerSize;

    std::vector<cv::Point3f> pts3d = {
        {-s/2,  s/2, 0}, { s/2,  s/2, 0}, { s/2, -s/2, 0}, {-s/2, -s/2, 0}, // основание
        {-s/2,  s/2, -h}, { s/2,  s/2, -h}, { s/2, -s/2, -h}, {-s/2, -s/2, -h}  // верх
    };

    std::vector<cv::Point2f> pts2d(8);
    for (int i = 0; i < 8; i++) {
        pts2d[i] = projectPoint(pts3d[i], rvec, tvec, cameraMatrix, distCoeffs);
    }

    int t = 2;

    // Нижнее основание — зелёное
    cv::line(image, pts2d[0], pts2d[1], cv::Scalar(0, 255, 0), t);
    cv::line(image, pts2d[1], pts2d[2], cv::Scalar(0, 255, 0), t);
    cv::line(image, pts2d[2], pts2d[3], cv::Scalar(0, 255, 0), t);
    cv::line(image, pts2d[3], pts2d[0], cv::Scalar(0, 255, 0), t);

    // Верхнее основание — синее
    cv::line(image, pts2d[4], pts2d[5], cv::Scalar(255, 0, 0), t);
    cv::line(image, pts2d[5], pts2d[6], cv::Scalar(255, 0, 0), t);
    cv::line(image, pts2d[6], pts2d[7], cv::Scalar(255, 0, 0), t);
    cv::line(image, pts2d[7], pts2d[4], cv::Scalar(255, 0, 0), t);

    // Вертикальные рёбра — красные
    cv::line(image, pts2d[0], pts2d[4], cv::Scalar(0, 0, 255), t);
    cv::line(image, pts2d[1], pts2d[5], cv::Scalar(0, 0, 255), t);
    cv::line(image, pts2d[2], pts2d[6], cv::Scalar(0, 0, 255), t);
    cv::line(image, pts2d[3], pts2d[7], cv::Scalar(0, 0, 255), t);
}

int main() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    cv::Mat cameraMatrix, distCoeffs;
    if (!loadCalibration("/home/kosmos/Desktop/lab_5/camera_calibration.yml", cameraMatrix, distCoeffs)) {
        std::cout << "Не удалось загрузить калибровку!" << std::endl;
        return -1;
    }
    std::cout << "Калибровка загружена успешно" << std::endl;

    cv::VideoCapture cap(0);
    globalCap = &cap;

    if (!cap.isOpened()) {
        std::cout << "Не удалось открыть камеру! Попробуй индекс 1 или 2" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    cv::aruco::DetectorParameters params;
    cv::aruco::ArucoDetector detector(dictionary, params);

    float markerSize = 0.035f;

    // Сглаживание для каждого маркера (по ID)
    std::map<int, cv::Vec3d> smoothRvecs, smoothTvecs;
    double alpha = 0.5;

    std::cout << "Нажми Q для выхода" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        detector.detectMarkers(frame, corners, ids, rejected);

        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(frame, corners, ids);

            for (size_t i = 0; i < ids.size(); i++) {
                int id = ids[i];
                cv::Vec3d rvec, tvec;

                std::vector<cv::Point3f> objPoints = {
                    {-markerSize/2,  markerSize/2, 0},
                    { markerSize/2,  markerSize/2, 0},
                    { markerSize/2, -markerSize/2, 0},
                    {-markerSize/2, -markerSize/2, 0}
                };
                cv::solvePnP(objPoints, corners[i], cameraMatrix, distCoeffs, rvec, tvec);

                // Инициализация сглаживания для нового маркера
                if (smoothRvecs.find(id) == smoothRvecs.end()) {
                    smoothRvecs[id] = rvec;
                    smoothTvecs[id] = tvec;
                }

                // Экспоненциальное сглаживание
                smoothRvecs[id] = alpha * rvec + (1.0 - alpha) * smoothRvecs[id];
                smoothTvecs[id] = alpha * tvec + (1.0 - alpha) * smoothTvecs[id];

                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs,
                                  smoothRvecs[id], smoothTvecs[id], markerSize * 0.5f);
                drawCube(frame, smoothRvecs[id], smoothTvecs[id],
                         cameraMatrix, distCoeffs, markerSize);

                cv::putText(frame, "ID: " + std::to_string(id),
                            corners[i][0], cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(255, 255, 0), 2);
            }
        }

        cv::imshow("ArUco Cube", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    cv::waitKey(1);
    return 0;
}