#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

// =====================================================================
// ---- Zhang-Suen с распараллеливанием ------
// =====================================================================
int countB(const int p[8]) {
    int B = 0;
    for (int k = 0; k < 8; k++) {
        if (p[k] == 255) B++;
    }
    return B;
}

int countA(const int p[8]) {
    int A = 0;
    for (int k = 0; k < 8; k++) {
        int cur = p[k];
        int nxt = p[(k + 1) % 8];
        if (cur == 0 && nxt == 255) A++;
    }
    return A;
}

// Параллельная обработка строк для скелетизации
void processRowsStep1(const Mat& img1, Mat& toDelete, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 1; j < img1.cols - 1; j++) {
            if (img1.at<uchar>(i, j) != 255) continue;
            
            int p[8] = {
                img1.at<uchar>(i-1, j),      // P2
                img1.at<uchar>(i-1, j+1),    // P3
                img1.at<uchar>(i, j+1),      // P4
                img1.at<uchar>(i+1, j+1),    // P5
                img1.at<uchar>(i+1, j),      // P6
                img1.at<uchar>(i+1, j-1),    // P7
                img1.at<uchar>(i, j-1),      // P8
                img1.at<uchar>(i-1, j-1)     // P9
            };
            
            int B = countB(p);
            if (B < 2 || B > 6) continue;
            
            int A = countA(p);
            if (A != 1) continue;
            
            if ((p[0] == 255) && (p[2] == 255) && (p[4] == 255)) continue;
            if ((p[2] == 255) && (p[4] == 255) && (p[6] == 255)) continue;
            
            toDelete.at<uchar>(i, j) = 1;
        }
    }
}

void processRowsStep2(const Mat& img1, Mat& toDelete, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 1; j < img1.cols - 1; j++) {
            if (img1.at<uchar>(i, j) != 255) continue;
            
            int p[8] = {
                img1.at<uchar>(i-1, j),      // P2
                img1.at<uchar>(i-1, j+1),    // P3
                img1.at<uchar>(i, j+1),      // P4
                img1.at<uchar>(i+1, j+1),    // P5
                img1.at<uchar>(i+1, j),      // P6
                img1.at<uchar>(i+1, j-1),    // P7
                img1.at<uchar>(i, j-1),      // P8
                img1.at<uchar>(i-1, j-1)     // P9
            };
            
            int B = countB(p);
            if (B < 2 || B > 6) continue;
            
            int A = countA(p);
            if (A != 1) continue;
            
            if ((p[0] == 255) && (p[2] == 255) && (p[6] == 255)) continue;
            if ((p[0] == 255) && (p[4] == 255) && (p[6] == 255)) continue;
            
            toDelete.at<uchar>(i, j) = 1;
        }
    }
}

Mat zhangSuenThinningParallel(const Mat img) {
    Mat img1 = img.clone();
    Mat img2 = img.clone();
    
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    while (true) {
        img2 = img1.clone();
        
        // Шаг 1 - параллельно
        Mat toDelete1 = Mat::zeros(img1.size(), CV_8UC1);
        vector<thread> threads;
        int rowsPerThread = (img1.rows - 2) / numThreads;
        
        for (int t = 0; t < numThreads; t++) {
            int startRow = 1 + t * rowsPerThread;
            int endRow = (t == numThreads - 1) ? img1.rows - 1 : startRow + rowsPerThread;
            threads.emplace_back(processRowsStep1, ref(img1), ref(toDelete1), startRow, endRow);
        }
        for (auto& th : threads) th.join();
        
        // Применяем удаление
        for (int i = 0; i < img1.rows; i++) {
            for (int j = 0; j < img1.cols; j++) {
                if (toDelete1.at<uchar>(i, j)) {
                    img2.at<uchar>(i, j) = 0;
                }
            }
        }
        
        if (countNonZero(img1 != img2) == 0) break;
        img1 = img2.clone();
        
        img2 = img1.clone();
        
        // Шаг 2 - параллельно
        Mat toDelete2 = Mat::zeros(img1.size(), CV_8UC1);
        threads.clear();
        
        for (int t = 0; t < numThreads; t++) {
            int startRow = 1 + t * rowsPerThread;
            int endRow = (t == numThreads - 1) ? img1.rows - 1 : startRow + rowsPerThread;
            threads.emplace_back(processRowsStep2, ref(img1), ref(toDelete2), startRow, endRow);
        }
        for (auto& th : threads) th.join();
        
        // Применяем удаление
        for (int i = 0; i < img1.rows; i++) {
            for (int j = 0; j < img1.cols; j++) {
                if (toDelete2.at<uchar>(i, j)) {
                    img2.at<uchar>(i, j) = 0;
                }
            }
        }
        
        if (countNonZero(img1 != img2) == 0) break;
        img1 = img2.clone();
    }
    
    return img1;
}

// =====================================================================
// Поиск белых областей (оптимизированная версия)
// =====================================================================
Mat findWhiteRegions(const Mat& frame) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    vector<Mat> ch(3);
    split(hsv, ch);
    
    double maxV;
    minMaxLoc(ch[2], nullptr, &maxV);
    int lowerV = (int)round(maxV * 0.4);
    int maxS = 120;
    
    Mat maskV, maskS, binary;
    threshold(ch[2], maskV, lowerV, 255, THRESH_BINARY);
    threshold(ch[1], maskS, maxS, 255, THRESH_BINARY_INV);
    bitwise_and(maskV, maskS, binary);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1, -1), 2);
    
    return binary;
}

// =====================================================================
// Расстояние между точками
// =====================================================================
inline double ptDist(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

// =====================================================================
// Улучшенное соединение отрезков с учетом Y-координаты
// =====================================================================
vector<Point> connectNearestEndpointsImproved(const vector<Vec4i>& lines, float maxJoinDistPx = 90.0f) {
    vector<Point> poly;
    if (lines.empty()) return poly;
    
    struct Seg {
        Point2f a, b;
        bool used = false;
        float length;
    };
    
    vector<Seg> segs;
    segs.reserve(lines.size());
    for (const auto& l : lines) {
        Point2f a((float)l[0], (float)l[1]);
        Point2f b((float)l[2], (float)l[3]);
        float len = sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
        segs.push_back({a, b, false, len});
    }
    
    auto d2 = [](const Point2f& p, const Point2f& q) -> float {
        float dx = p.x - q.x;
        float dy = p.y - q.y;
        return dx*dx + dy*dy;
    };
    
    const float maxD2 = maxJoinDistPx * maxJoinDistPx;
    
    // Начинаем с самого длинного отрезка, который находится ближе к низу кадра
    int startIdx = 0;
    float bestScore = -1.0f;
    for (int i = 0; i < (int)segs.size(); i++) {
        float yAvg = (segs[i].a.y + segs[i].b.y) / 2.0f;
        float score = segs[i].length + yAvg * 0.1f; // Приоритет длинным и нижним отрезкам
        if (score > bestScore) {
            bestScore = score;
            startIdx = i;
        }
    }
    
    segs[startIdx].used = true;
    poly.push_back(Point((int)lround(segs[startIdx].a.x), (int)lround(segs[startIdx].a.y)));
    poly.push_back(Point((int)lround(segs[startIdx].b.x), (int)lround(segs[startIdx].b.y)));
    
    // Сортируем точки по Y (нижняя первая)
    if (poly[0].y < poly[1].y) {
        swap(poly[0], poly[1]);
    }
    
    bool extended;
    do {
        extended = false;
        Point2f head((float)poly.front().x, (float)poly.front().y);
        Point2f tail((float)poly.back().x, (float)poly.back().y);
        
        int bestIdx = -1;
        int bestMode = -1;
        float bestDist2 = maxD2;
        
        for (int i = 0; i < (int)segs.size(); i++) {
            if (segs[i].used) continue;
            
            // Проверяем соединение с хвостом (продолжение вверх)
            float t_a = d2(tail, segs[i].a);
            if (t_a < bestDist2 && segs[i].a.y < tail.y + 20) { // Предпочитаем движение вверх
                bestDist2 = t_a;
                bestIdx = i;
                bestMode = 2;
            }
            
            float t_b = d2(tail, segs[i].b);
            if (t_b < bestDist2 && segs[i].b.y < tail.y + 20) {
                bestDist2 = t_b;
                bestIdx = i;
                bestMode = 3;
            }
            
            // Проверяем соединение с головой (вниз, но реже)
            if (poly.size() < 3) { // Только в начале позволяем расширяться вниз
                float h_a = d2(head, segs[i].a);
                if (h_a < bestDist2 && segs[i].a.y > head.y - 20) {
                    bestDist2 = h_a;
                    bestIdx = i;
                    bestMode = 0;
                }
                
                float h_b = d2(head, segs[i].b);
                if (h_b < bestDist2 && segs[i].b.y > head.y - 20) {
                    bestDist2 = h_b;
                    bestIdx = i;
                    bestMode = 1;
                }
            }
        }
        
        if (bestIdx != -1) {
            segs[bestIdx].used = true;
            const auto& s = segs[bestIdx];
            
            if (bestMode == 0) {
                poly.insert(poly.begin(), Point((int)lround(s.b.x), (int)lround(s.b.y)));
            } else if (bestMode == 1) {
                poly.insert(poly.begin(), Point((int)lround(s.a.x), (int)lround(s.a.y)));
            } else if (bestMode == 2) {
                poly.push_back(Point((int)lround(s.b.x), (int)lround(s.b.y)));
            } else if (bestMode == 3) {
                poly.push_back(Point((int)lround(s.a.x), (int)lround(s.a.y)));
            }
            extended = true;
        }
    } while (extended);
    
    // Добавляем недостающие отрезки внизу, пытаясь соединить с ближайшими
    if (poly.size() >= 2) {
        Point lowestPoint = poly[0];
        for (const auto& p : poly) {
            if (p.y > lowestPoint.y) lowestPoint = p;
        }
        
        // Ищем отрезки ниже самой нижней точки
        float minY = lowestPoint.y;
        for (int i = 0; i < (int)segs.size(); i++) {
            if (segs[i].used) continue;
            
            float segMinY = min(segs[i].a.y, segs[i].b.y);
            if (segMinY > minY) {
                float distToA = ptDist(lowestPoint, Point((int)segs[i].a.x, (int)segs[i].a.y));
                float distToB = ptDist(lowestPoint, Point((int)segs[i].b.x, (int)segs[i].b.y));
                
                if (distToA < maxJoinDistPx) {
                    poly.push_back(Point((int)segs[i].a.x, (int)segs[i].a.y));
                    poly.push_back(Point((int)segs[i].b.x, (int)segs[i].b.y));
                    segs[i].used = true;
                } else if (distToB < maxJoinDistPx) {
                    poly.push_back(Point((int)segs[i].b.x, (int)segs[i].b.y));
                    poly.push_back(Point((int)segs[i].a.x, (int)segs[i].a.y));
                    segs[i].used = true;
                }
            }
        }
    }
    
    return poly;
}

// =====================================================================
// Обработка кадра с улучшенным детектированием низа
// =====================================================================
Mat processFrame(const Mat& frame, bool showDebug = false) {
    Mat binary = findWhiteRegions(frame);
    
    // Добавляем дополнительную обработку нижней части кадра
    int bottomHeight = frame.rows / 4;
    Mat bottomROI = binary(Rect(0, frame.rows - bottomHeight, frame.cols, bottomHeight));
    
    // Усиливаем линии в нижней части
    Mat kernelBottom = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(bottomROI, bottomROI, kernelBottom);
    erode(bottomROI, bottomROI, kernelBottom);
    
    Mat skeleton = zhangSuenThinningParallel(binary);
    
    vector<Vec4i> lines;
    HoughLinesP(skeleton, lines, 1, CV_PI / 180, 45, 25, 60);
    
    Mat result = frame.clone();
    
    // Рисуем линии Хафа
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
    }
    
    vector<Point> poly = connectNearestEndpointsImproved(lines, 100.0f);
    
    if (poly.size() >= 2) {
        vector<vector<Point>> pts = { poly };
        polylines(result, pts, false, Scalar(0, 255, 0), 3, LINE_AA);
        
        for (const auto& p : poly) {
            circle(result, p, 5, Scalar(255, 0, 0), -1);
        }
    }
    
    if (showDebug) {
        imshow("[Debug] Binary", binary);
        imshow("[Debug] Skeleton", skeleton);
        
        string info = "Lines: " + to_string(lines.size()) + 
                      "  Points: " + to_string(poly.size());
        putText(result, info, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
    }
    
    return result;
}

// =====================================================================
// main
// =====================================================================
int main() {
    string dir = "/home/kosmos/Desktop/lab_7/line_vids/";
    bool showDebug = true;
    
    cout << "Количество ядер CPU: " << thread::hardware_concurrency() << endl;
    
    for (int i = 0; i <= 3; i++) {
        string path = dir + to_string(i) + ".avi";
        VideoCapture cap(path);
        if (!cap.isOpened()) {
            cerr << "Не удалось открыть: " << path << endl;
            continue;
        }
        
        cout << "Обрабатываем: " << path << endl;
        
        double fps = cap.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 25.0;
        
        Size sz((int)cap.get(CAP_PROP_FRAME_WIDTH),
                (int)cap.get(CAP_PROP_FRAME_HEIGHT));
        
        VideoWriter writer(dir + "result_" + to_string(i) + ".avi",
                           VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        
        string win = "Video " + to_string(i);
        namedWindow(win, WINDOW_NORMAL);
        
        int frameCount = 0;
        auto startTime = chrono::steady_clock::now();
        
        for (;;) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) break;
            
            Mat out = processFrame(frame, showDebug);
            writer.write(out);
            imshow(win, out);
            
            int key = waitKey(1);
            if (key == 27 || key == 'q') break;
            if (key == 'd') {
                showDebug = !showDebug;
                if (!showDebug) {
                    destroyWindow("[Debug] Binary");
                    destroyWindow("[Debug] Skeleton");
                }
            }
            
            frameCount++;
            
            // Вывод статистики каждые 100 кадров
            if (frameCount % 100 == 0) {
                auto now = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - startTime).count();
                cout << "Frame " << frameCount << ", FPS: " << (frameCount * 1000.0 / elapsed) << endl;
            }
        }
        
        cap.release();
        writer.release();
        destroyWindow(win);
        cout << "Сохранено: " << dir + "result_" + to_string(i) + ".avi" << endl;
        cout << "Обработано кадров: " << frameCount << endl;
    }
    
    destroyAllWindows();
    return 0;
}