#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ===== ЧИСТАЯ НАСЫЩЕННОСТЬ =====
double mean_S_clean(const Mat& img) {
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    vector<Mat> ch;
    split(hsv, ch);
    Mat S = ch[1];
    Mat V = ch[2];

    Mat maskV, maskS;
    inRange(V, 50, 220, maskV);
    inRange(S, 30, 255, maskS);

    Mat mask = maskV & maskS;

    if (countNonZero(mask) < 50)
        return mean(S)[0];

    return mean(S, mask)[0];
}

// ===== S В КРУГЕ =====
double mean_S_in_circle(const Mat& img, Point center, int radius) {
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    vector<Mat> ch;
    split(hsv, ch);
    Mat S = ch[1];
    Mat V = ch[2];

    Mat maskCircle = Mat::zeros(img.size(), CV_8UC1);
    circle(maskCircle, center, radius * 0.7, Scalar(255), -1);

    Mat maskV, maskS;
    inRange(V, 50, 220, maskV);
    inRange(S, 20, 255, maskS);

    Mat mask = maskCircle & maskV & maskS;

    if (countNonZero(mask) < 50)
        mask = maskCircle;

    return mean(S, mask)[0];
}

// ===== КЛАССИФИКАЦИЯ =====
string classify(double coinS, double nickelS, double latunS) {
    double k = 2.0; // <-- твой коэффициент

    double dN = abs(coinS - nickelS);
    double dL = abs(coinS - k * latunS);

    return (dL < dN) ? "LATUN" : "NICKEL";
}

// ===== УМНОЕ РАЗМЕЩЕНИЕ ТЕКСТА =====
Point getLabelPosition(Point center, int radius, int index) {
    int offset = radius + 20;

    // разные направления (чтобы не пересекались)
    switch (index % 4) {
        case 0: return Point(center.x + offset, center.y);
        case 1: return Point(center.x - offset, center.y);
        case 2: return Point(center.x, center.y + offset);
        default:return Point(center.x, center.y - offset);
    }
}

vector<Vec3f> sort_circles_reading_order(vector<Vec3f> circles) {
    // сортируем по Y
    sort(circles.begin(), circles.end(), [](const Vec3f& a, const Vec3f& b) {
        return a[1] < b[1];
    });

    vector<vector<Vec3f>> rows;
    const int rowThreshold = 40; // допуск по высоте

    for (auto& c : circles) {
        bool added = false;

        for (auto& row : rows) {
            if (abs(c[1] - row[0][1]) < rowThreshold) {
                row.push_back(c);
                added = true;
                break;
            }
        }

        if (!added) {
            rows.push_back({c});
        }
    }

    // сортировка внутри строк по X
    for (auto& row : rows) {
        sort(row.begin(), row.end(), [](const Vec3f& a, const Vec3f& b) {
            return a[0] < b[0];
        });
    }

    // собираем обратно
    vector<Vec3f> sorted;
    for (auto& row : rows) {
        for (auto& c : row) {
            sorted.push_back(c);
        }
    }

    return sorted;
}

int main() {
    string base = "/home/kosmos/Desktop/lab_7/coins_img/";

    Mat coins = imread(base + "coins.jpg");
    Mat copper = imread(base + "copper.jpg");
    Mat nickel = imread(base + "nikel.jpg");

    if (coins.empty() || copper.empty() || nickel.empty()) {
        cout << "Ошибка загрузки изображений!" << endl;
        return -1;
    }

    // ===== ЭТАЛОНЫ =====
    double latunS = mean_S_clean(copper);
    double nickelS = mean_S_clean(nickel);

    // ===== ПОИСК КРУГОВ =====
    Mat gray;
    cvtColor(coins, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(9, 9), 2);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT,
                 1, 40,
                 100, 40,
                 10, 120);

    circles = sort_circles_reading_order(circles);
    
    vector<string> labels;

    int latunCount = 0;
    int nickelCount = 0;

    // ===== ОБРАБОТКА МОНЕТ =====
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        double coinS = mean_S_in_circle(coins, center, radius);
        string type = classify(coinS, nickelS, latunS);

        if (type == "LATUN") latunCount++;
        else nickelCount++;

        // Рисуем круг
        circle(coins, center, radius, Scalar(0, 255, 0), 2);

        // НОМЕР НА МОНЕТЕ (маленький)
        string id = to_string(i + 1);
        putText(coins, id,
                center,
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(255, 255, 255),
                2);

        // СОХРАНЯЕМ ДЛЯ ВЕРХНЕЙ ПАНЕЛИ
        char buf[100];
        sprintf(buf, "#%d: %s (S=%.0f)", (int)i + 1, type.c_str(), coinS);
        labels.push_back(string(buf));
    }

    // ===== СОЗДАЁМ ВЕРХНЮЮ ПАНЕЛЬ =====
    int panelHeight = 30 + labels.size() * 25;

    Mat panel(panelHeight, coins.cols, CV_8UC3, Scalar(40, 40, 40));

    // Заголовок
    string header = "Latun: " + to_string(latunCount) +
                    "   Nickel: " + to_string(nickelCount);

    putText(panel, header,
            Point(20, 25),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 255, 255),
            2);

    // Список монет
    for (size_t i = 0; i < labels.size(); i++) {
        putText(panel, labels[i],
                Point(20, 50 + i * 25),
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(255, 255, 255),
                1);
    }

    // ===== ОБЪЕДИНЯЕМ =====
    Mat result;
    vconcat(panel, coins, result);

    // ===== СОХРАНЕНИЕ =====
    string outPath = base + "result.jpg";
    imwrite(outPath, result);

    cout << "Saved to: " << outPath << endl;

    // ===== ПОКАЗ =====
    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", result);
    waitKey(0);

    return 0;
}