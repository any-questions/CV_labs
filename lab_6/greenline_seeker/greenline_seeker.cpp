#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
using namespace cv;
using namespace std;

void laser(Mat &frame, Mat& mask) {
    cvtColor(frame, mask, COLOR_BGR2HSV);
    inRange(mask, Scalar(0, 0, 130), Scalar(90, 115, 255), mask);
}

string stemName(const string& path) {
    size_t slash = path.find_last_of("/\\");
    string fname = (slash == string::npos) ? path : path.substr(slash + 1);
    size_t dot = fname.find_last_of('.');
    return (dot == string::npos) ? fname : fname.substr(0, dot);
}

string uniqueOutput(const string& stem) {
    string candidate = stem + "_map.avi";
    struct stat st;
    int idx = 1;
    while (stat(candidate.c_str(), &st) == 0) {
        ostringstream oss;
        oss << stem << "_map_" << idx++ << ".avi";
        candidate = oss.str();
    }
    return candidate;
}

// Сетка с подписями расстояний
void drawGrid(Mat& img, int originX, int originY, double mmPerPixel) {
    int W = img.cols, H = img.rows;

    int stepMinor = (int)round(100.0 / mmPerPixel);
    for (int x = originX % stepMinor; x < W; x += stepMinor)
        line(img, Point(x, 0), Point(x, H), Scalar(28, 28, 28), 1);
    for (int y = originY % stepMinor; y < H; y += stepMinor)
        line(img, Point(0, y), Point(W, y), Scalar(28, 28, 28), 1);

    int stepMajor = (int)round(500.0 / mmPerPixel);
    for (int x = 0; x < W; x += stepMajor) {
        line(img, Point(x, 0), Point(x, H), Scalar(55, 55, 55), 1);
        int cm = (int)round((x - originX) * mmPerPixel / 10.0);
        string lbl = (cm >= 0 ? "+" : "") + to_string(cm) + "cm";
        putText(img, lbl, Point(x + 2, H - 4),
                FONT_HERSHEY_PLAIN, 0.75, Scalar(90, 90, 90), 1);
    }
    for (int y = 0; y < H; y += stepMajor) {
        line(img, Point(0, y), Point(W, y), Scalar(55, 55, 55), 1);
        int cm = (int)round((originY - y) * mmPerPixel / 10.0);
        if (cm <= 0) continue;
        string lbl = to_string(cm) + "cm";
        putText(img, lbl, Point(3, y - 3),
                FONT_HERSHEY_PLAIN, 0.75, Scalar(90, 90, 90), 1);
    }

    // Осевые линии через позицию телеги
    line(img, Point(0, originY), Point(W, originY), Scalar(50, 50, 90), 1);
    line(img, Point(originX, 0), Point(originX, H), Scalar(50, 50, 90), 1);
}

int main(int argc, char** argv) {
    bool debug = true;

    string videoPath = (argc > 1) ? argv[1] : "/home/kosmos/Desktop/lab_6/vids/3.avi";
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Could not open video: " << videoPath << endl;
        return -1;
    }

    int frameWidth  = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    double frameRate = cap.get(CAP_PROP_FPS);

    int cent_x = frameWidth / 2;
    int cent_y = frameHeight / 2;

    int camAngle = 74;
    double camAngleX = camAngle * CV_PI / 180;
    double focusX = (frameWidth / 2.0) / tan(camAngleX / 2.0);
    double focusY = focusX;

    double laserD = -145.0;

    int mapW = frameWidth;
    int mapH = frameHeight;
    int mapOriginX = mapW / 2;
    int mapOriginY = mapH - 1;
    double mmPerPixel = 5.0; // параметр для настройки масштаба карты

    string outName = uniqueOutput(stemName(videoPath));
    cout << "Video:  " << frameWidth << "x" << frameHeight << " fps=" << frameRate << endl;
    cout << "focusX=" << focusX << "  laserD=" << laserD << " mm" << endl;
    cout << "Output: " << outName << endl;

    VideoWriter mapperVid(outName,
                          VideoWriter::fourcc('M','J','P','G'),
                          frameRate, Size(mapW, mapH));

    Mat frame, mask, out;

    while (cap.read(frame)) {
        laser(frame, mask);

        out = Mat::zeros(mapH, mapW, CV_8UC3);

        drawGrid(out, mapOriginX, mapOriginY, mmPerPixel);

        // Проецируем лазерные пиксели на карту
        for (int y = 0; y < frameHeight; y++) {
            for (int x = 0; x < frameWidth; x++) {
                if (mask.at<uchar>(y, x) != 255) continue;

                double ex = (x - cent_x) / focusX;
                double ey = (y - cent_y) / focusY;
                double ez = 1.0;
                double eLen = sqrt(ex*ex + ey*ey + ez*ez);
                ex /= eLen; ey /= eLen; ez /= eLen;

                if (fabs(ey) < 1e-6) continue;
                double k = -laserD / ey;
                if (k <= 0) continue;

                double X = k * ex;
                double Z = k * ez;

                int mapX = mapOriginX + (int)round(X / mmPerPixel);
                int mapY = mapOriginY - (int)round(Z / mmPerPixel);

                if (mapX < 0 || mapX >= mapW) continue;
                if (mapY < 0 || mapY >= mapH) continue;

                circle(out, Point(mapX, mapY), 1, Scalar(0, 255, 0), -1);
            }
        }

        // Значок телеги
        circle(out, Point(mapOriginX, mapOriginY - 5), 5, Scalar(0, 165, 255), -1);

        mapperVid.write(out);

        if (debug) {
            imshow("video", frame);
            imshow("laser mask", mask);
            imshow("map", out);
            int key = waitKey(30);
            if (key == 27) break;
            if (key == 32) waitKey(0);
        }
    }

    mapperVid.release();
    cout << "Saved: " << outName << endl;
    return 0;
}