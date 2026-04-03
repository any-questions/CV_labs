#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

// =====================================================================
// ---- Zhang-Suen (из задания 1) ------
// =====================================================================

int countTransitions(const uchar* n) {
    int count = 0;
    for (int i = 0; i < 8; i++) {
        int cur  = (n[i]         > 0) ? 1 : 0;
        int next = (n[(i+1) % 8] > 0) ? 1 : 0;
        if (cur == 0 && next == 1) count++;
    }
    return count;
}

int countNeighbors(const uchar* n) {
    int count = 0;
    for (int i = 0; i < 8; i++) if (n[i] > 0) count++;
    return count;
}

void getNeighbors(const Mat& img, int r, int c, uchar* n) {
    n[0] = img.at<uchar>(r-1, c  ); // P2
    n[1] = img.at<uchar>(r-1, c+1); // P3
    n[2] = img.at<uchar>(r  , c+1); // P4
    n[3] = img.at<uchar>(r+1, c+1); // P5
    n[4] = img.at<uchar>(r+1, c  ); // P6
    n[5] = img.at<uchar>(r+1, c-1); // P7
    n[6] = img.at<uchar>(r  , c-1); // P8
    n[7] = img.at<uchar>(r-1, c-1); // P9
}

Mat zhangSuenThinning(const Mat& inputImg) {
    Mat img;
    inputImg.copyTo(img);
    img /= 255;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int step = 1; step <= 2; step++) {
            Mat toDelete = Mat::zeros(img.size(), CV_8UC1);
            for (int r = 1; r < img.rows - 1; r++) {
                for (int c = 1; c < img.cols - 1; c++) {
                    if (img.at<uchar>(r, c) == 0) continue;
                    uchar nb[8];
                    getNeighbors(img, r, c, nb);
                    int B = countNeighbors(nb);
                    int A = countTransitions(nb);
                    uchar P2 = nb[0], P4 = nb[2], P6 = nb[4], P8 = nb[6];
                    bool cond3, cond4;
                    if (step == 1) {
                        cond3 = (P2 == 0 || P4 == 0 || P6 == 0);
                        cond4 = (P4 == 0 || P6 == 0 || P8 == 0);
                    } else {
                        cond3 = (P2 == 0 || P4 == 0 || P8 == 0);
                        cond4 = (P2 == 0 || P6 == 0 || P8 == 0);
                    }
                    if (B >= 2 && B <= 6 && A == 1 && cond3 && cond4)
                        toDelete.at<uchar>(r, c) = 1;
                }
            }
            for (int r = 0; r < img.rows; r++)
                for (int c = 0; c < img.cols; c++)
                    if (toDelete.at<uchar>(r, c)) {
                        img.at<uchar>(r, c) = 0;
                        changed = true;
                    }
        }
    }
    img *= 255;
    return img;
}

// =====================================================================
// Предобработка кадра: бинаризация белой линии на тёмном полу
// =====================================================================
Mat preprocessFrame(const Mat& frame) {
    Mat gray, blurred, binary;

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Размытие для устранения шума
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // Адаптивная бинаризация — работает лучше при неравномерном освещении
    // Белая линия (светлая) на тёмном фоне
    adaptiveThreshold(blurred, binary, 255,
                      ADAPTIVE_THRESH_GAUSSIAN_C,
                      THRESH_BINARY, 31, -10);

    // Морфология: убираем мелкий шум
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN,  kernel);
    morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1,-1), 2);

    return binary;
}

// =====================================================================
// Расстояние между двумя точками
// =====================================================================
double dist(Point a, Point b) {
    return sqrt((double)(a.x-b.x)*(a.x-b.x) + (double)(a.y-b.y)*(a.y-b.y));
}

// =====================================================================
// Объединить отрезки в одну ломаную vector<Point>
// Алгоритм: жадный — всегда берём ближайший незадействованный конец
// =====================================================================
vector<Point> buildPolyline(const vector<Vec4i>& lines) {
    if (lines.empty()) return {};

    // Представим каждую линию как пару точек
    // Каждый отрезок можно пройти в любом направлении
    int n = (int)lines.size();
    vector<bool> used(n, false);

    // Стартуем с первого отрезка
    vector<Point> poly;
    used[0] = true;
    Point cur_start(lines[0][0], lines[0][1]);
    Point cur_end  (lines[0][2], lines[0][3]);
    poly.push_back(cur_start);
    poly.push_back(cur_end);

    for (int iter = 0; iter < n - 1; iter++) {
        double best_dist = numeric_limits<double>::max();
        int    best_idx  = -1;
        bool   best_reverse = false;

        // Ищем ближайший конец среди неиспользованных отрезков
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            Point p1(lines[i][0], lines[i][1]);
            Point p2(lines[i][2], lines[i][3]);

            double d1 = dist(cur_end, p1); // присоединяем начало p1
            double d2 = dist(cur_end, p2); // присоединяем конец p2 (разворот)

            if (d1 < best_dist) { best_dist = d1; best_idx = i; best_reverse = false; }
            if (d2 < best_dist) { best_dist = d2; best_idx = i; best_reverse = true;  }
        }

        if (best_idx == -1) break;

        used[best_idx] = true;
        Point next_start(lines[best_idx][0], lines[best_idx][1]);
        Point next_end  (lines[best_idx][2], lines[best_idx][3]);

        if (!best_reverse) {
            poly.push_back(next_start);
            poly.push_back(next_end);
            cur_end = next_end;
        } else {
            poly.push_back(next_end);
            poly.push_back(next_start);
            cur_end = next_start;
        }
    }

    return poly;
}

// =====================================================================
// Обработка одного кадра: бинаризация → скелет → линии → ломаная
// Возвращает: кадр с нарисованной ломаной
// =====================================================================
Mat processFrame(const Mat& frame, bool drawDebug = false) {
    Mat result = frame.clone();

    // 1. Предобработка
    Mat binary = preprocessFrame(frame);

    // 2. Скелетизация (Zhang-Suen)
    Mat skeleton = zhangSuenThinning(binary);

    // 3. Поиск линий через вероятностное преобразование Хафа
    vector<Vec4i> lines;
    HoughLinesP(skeleton, lines, 1, CV_PI / 180,
                /*threshold=*/30,
                /*minLineLength=*/40,
                /*maxLineGap=*/20);

    if (lines.empty()) {
        putText(result, "No lines found", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255), 2);
        return result;
    }

    // 4. Строим ломаную из найденных отрезков
    vector<Point> polyline = buildPolyline(lines);

    // 5. Рисуем отладочную информацию (опционально)
    if (drawDebug) {
        // Показываем отдельные линии Хафа (синим)
        for (auto& l : lines) {
            line(result, Point(l[0],l[1]), Point(l[2],l[3]),
                 Scalar(255, 100, 0), 1, LINE_AA);
        }
    }

    // 6. Рисуем итоговую ломаную (зелёным, толстой линией)
    if (polyline.size() >= 2) {
        // polylines ожидает vector<vector<Point>>
        vector<vector<Point>> pts = { polyline };
        polylines(result, pts, false, Scalar(0, 255, 0), 2, LINE_AA);

        // Рисуем точки излома (красными кружками)
        for (const auto& p : polyline) {
            circle(result, p, 3, Scalar(0, 0, 255), -1);
        }
    }

    // Небольшая подсказка на кадре
    string info = "Lines: " + to_string(lines.size()) +
                  "  Pts: " + to_string(polyline.size());
    putText(result, info, Point(10, 25),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,0), 1);

    return result;
}

// =====================================================================
// Обработка статичных изображений
// =====================================================================
void processImages(const string& dir) {
    for (int i = 0; i <= 3; i++) {
        string path = dir + to_string(i) + ".jpg";
        Mat frame = imread(path);
        if (frame.empty()) {
            // Попробуем .png
            path = dir + to_string(i) + ".png";
            frame = imread(path);
        }
        if (frame.empty()) {
            cout << "Изображение не найдено: " << path << endl;
            continue;
        }

        cout << "Обрабатываем изображение: " << path << endl;

        Mat out = processFrame(frame, /*drawDebug=*/true);

        // Сохраняем
        string savePath = dir + "result_" + to_string(i) + ".png";
        imwrite(savePath, out);

        imshow("Image " + to_string(i), out);
    }
    waitKey(0);
    destroyAllWindows();
}

// =====================================================================
// Обработка видеофайлов
// =====================================================================
void processVideos(const string& dir) {
    for (int i = 0; i <= 3; i++) {
        string path = dir + to_string(i) + ".avi";
        VideoCapture cap(path);

        if (!cap.isOpened()) {
            cerr << "Не удалось открыть видео: " << path << endl;
            continue;
        }

        cout << "Обрабатываем видео: " << path << endl;

        // Параметры для записи выходного видео
        int fourcc = VideoWriter::fourcc('M','J','P','G');
        double fps = cap.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 25.0;
        Size frameSize((int)cap.get(CAP_PROP_FRAME_WIDTH),
                       (int)cap.get(CAP_PROP_FRAME_HEIGHT));

        string outPath = dir + "result_simple_" + to_string(i) + ".avi";
        VideoWriter writer(outPath, fourcc, fps, frameSize);

        string winName = "Video " + to_string(i);

        for (;;) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            Mat out = processFrame(frame, /*drawDebug=*/false);

            writer.write(out);
            imshow(winName, out);

            // q или ESC — следующее видео
            int key = waitKey(1);
            if (key == 'q' || key == 27) break;
        }

        cap.release();
        writer.release();
        destroyWindow(winName);
        cout << "Сохранено: " << outPath << endl;
    }
}

// =====================================================================
// main
// =====================================================================
int main() {
    string videoDir = "/home/kosmos/Desktop/lab_7/line_vids/";
    string imageDir = "/home/kosmos/Desktop/lab_7/line_imgs/"; // если есть папка с изображениями

    cout << "=== Обработка видеофайлов ===" << endl;
    processVideos(videoDir);

    // Раскомментируй, если есть отдельная папка с изображениями разметки:
    // cout << "\n=== Обработка изображений ===" << endl;
    // processImages(imageDir);

    return 0;
}