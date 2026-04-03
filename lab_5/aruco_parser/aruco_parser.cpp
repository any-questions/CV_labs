#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

// Загрузка калибровки
bool loadCalibration(const std::string& path, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs.release();
    return true;
}

// Проецируем 3D точку на 2D
cv::Point2f projectPoint(const cv::Point3f& p3d,
                          const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                          const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    std::vector<cv::Point3f> pts = {p3d};
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs, pts2d);
    return pts2d[0];
}

// Рисуем куб поверх маркера
void drawCube(cv::Mat& image, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
              const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, float markerSize) {
    float s = markerSize;
    float h = markerSize; // высота куба = размер маркера

    // 8 вершин куба
    // Основание (z=0, лежит на маркере)
    std::vector<cv::Point3f> pts3d = {
        {0, 0, 0}, {s, 0, 0}, {s, s, 0}, {0, s, 0},  // нижние 4 точки
        {0, 0, -h}, {s, 0, -h}, {s, s, -h}, {0, s, -h} // верхние 4 точки
    };

    // Центрируем куб по маркеру
    for (auto& p : pts3d) {
        p.x -= s / 2;
        p.y -= s / 2;
    }

    // Проецируем все точки
    std::vector<cv::Point2f> pts2d(8);
    for (int i = 0; i < 8; i++) {
        pts2d[i] = projectPoint(pts3d[i], rvec, tvec, cameraMatrix, distCoeffs);
    }

    int thickness = 2;

    // Нижнее основание (зелёное) — лежит на маркере
    cv::line(image, pts2d[0], pts2d[1], cv::Scalar(0, 255, 0), thickness);
    cv::line(image, pts2d[1], pts2d[2], cv::Scalar(0, 255, 0), thickness);
    cv::line(image, pts2d[2], pts2d[3], cv::Scalar(0, 255, 0), thickness);
    cv::line(image, pts2d[3], pts2d[0], cv::Scalar(0, 255, 0), thickness);

    // Верхнее основание (синее)
    cv::line(image, pts2d[4], pts2d[5], cv::Scalar(255, 0, 0), thickness);
    cv::line(image, pts2d[5], pts2d[6], cv::Scalar(255, 0, 0), thickness);
    cv::line(image, pts2d[6], pts2d[7], cv::Scalar(255, 0, 0), thickness);
    cv::line(image, pts2d[7], pts2d[4], cv::Scalar(255, 0, 0), thickness);

    // Вертикальные рёбра (красные)
    cv::line(image, pts2d[0], pts2d[4], cv::Scalar(0, 0, 255), thickness);
    cv::line(image, pts2d[1], pts2d[5], cv::Scalar(0, 0, 255), thickness);
    cv::line(image, pts2d[2], pts2d[6], cv::Scalar(0, 0, 255), thickness);
    cv::line(image, pts2d[3], pts2d[7], cv::Scalar(0, 0, 255), thickness);
}

int main() {
    cv::Mat cameraMatrix, distCoeffs;
    if (!loadCalibration("/home/kosmos/Desktop/lab_5/camera_calibration.yml", cameraMatrix, distCoeffs)) {
        std::cout << "Не удалось загрузить калибровку!" << std::endl;
        return -1;
    }
    std::cout << "Калибровка загружена успешно" << std::endl;

    cv::VideoCapture cap(0); // 0 = первая веб-камера
    if (!cap.isOpened()) {
        std::cout << "Не удалось открыть камеру!" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    cv::aruco::DetectorParameters params;
    cv::aruco::ArucoDetector detector(dictionary, params);

    float markerSize = 0.035f; // реальный размер маркера в метрах — подправь под свой

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
            // Рисуем контуры маркеров
            cv::aruco::drawDetectedMarkers(frame, corners, ids);

            // Для каждого найденного маркера рисуем куб
            for (size_t i = 0; i < ids.size(); i++) {
                cv::Vec3d rvec, tvec;

                // Определяем позицию маркера в пространстве
                std::vector<cv::Point3f> objPoints = {
                    {-markerSize/2,  markerSize/2, 0},
                    { markerSize/2,  markerSize/2, 0},
                    { markerSize/2, -markerSize/2, 0},
                    {-markerSize/2, -markerSize/2, 0}
                };
                cv::solvePnP(objPoints, corners[i], cameraMatrix, distCoeffs, rvec, tvec);

                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, markerSize * 0.5f);

                drawCube(frame, rvec, tvec, cameraMatrix, distCoeffs, markerSize);

                cv::Point2f center = corners[i][0];
                cv::putText(frame, "ID: " + std::to_string(ids[i]),
                            center, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 0), 2);
            }
        }

        cv::imshow("ArUco Cube", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}