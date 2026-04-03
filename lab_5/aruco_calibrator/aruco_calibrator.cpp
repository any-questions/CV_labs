#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <iostream>
#include <vector>
#include <glob.h>

int main() {
    int squaresX = 5;
    int squaresY = 7;
    float squareLength = 0.035f;
    float markerLength = 0.017f;

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

    cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(
        cv::Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    cv::aruco::CharucoDetector detector(*board);

    // Загружаем все BMP файлы
    std::vector<std::string> imagePaths;
    glob_t glob_result;
    glob("/home/kosmos/Desktop/lab_5/Camera_Calibration_Aruco/*.bmp", GLOB_TILDE, NULL, &glob_result);
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        imagePaths.push_back(glob_result.gl_pathv[i]);
    }
    globfree(&glob_result);

    std::cout << "Найдено фотографий: " << imagePaths.size() << std::endl;

    std::vector<std::vector<cv::Point2f>> allCharucoCorners;
    std::vector<std::vector<int>> allCharucoIds;
    cv::Size imageSize;

    for (const auto& path : imagePaths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cout << "Не удалось загрузить: " << path << std::endl;
            continue;
        }
        imageSize = image.size();

        std::vector<int> charucoIds;
        std::vector<cv::Point2f> charucoCorners;
        detector.detectBoard(image, charucoCorners, charucoIds);

        std::cout << path << " — найдено углов: " << charucoIds.size() << std::endl;

        cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds);
        cv::imshow("Calibration", image);
        cv::waitKey(300);

        if (charucoIds.size() >= 6) {
            allCharucoCorners.push_back(charucoCorners);
            allCharucoIds.push_back(charucoIds);
        }
    }
    cv::destroyAllWindows();

    if (allCharucoCorners.size() < 3) {
        std::cout << "Недостаточно фотографий для калибровки!" << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    double reprojError = cv::aruco::calibrateCameraCharuco(
        allCharucoCorners, allCharucoIds, board,
        imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
        cv::noArray(), cv::noArray(), cv::noArray()
    );

    std::cout << "\n=== Результаты калибровки ===" << std::endl;
    std::cout << "Ошибка репроекции: " << reprojError << std::endl;
    std::cout << "\nМатрица камеры:\n" << cameraMatrix << std::endl;
    std::cout << "\nКоэффициенты дисторсии:\n" << distCoeffs << std::endl;

    cv::FileStorage fs("/home/kosmos/Desktop/lab_5/camera_calibration.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs << "reprojection_error" << reprojError;
    fs.release();

    std::cout << "\nКалибровка сохранена в camera_calibration.yml" << std::endl;
    return 0;
}