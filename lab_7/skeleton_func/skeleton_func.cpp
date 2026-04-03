#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Функция A(P1) — количество переходов 0->1 в последовательности
// P2, P3, P4, P5, P6, P7, P8, P9, P2 (по кругу)
int countTransitions(const uchar* neighbors) {
    // neighbors[0..7] = P2..P9
    int count = 0;
    for (int i = 0; i < 8; i++) {
        // переход от 0 к 1: текущий=0, следующий=1
        int cur  = (neighbors[i]     > 0) ? 1 : 0;
        int next = (neighbors[(i+1) % 8] > 0) ? 1 : 0;
        if (cur == 0 && next == 1)
            count++;
    }
    return count;
}


int countNeighbors(const uchar* neighbors) {
    int count = 0;
    for (int i = 0; i < 8; i++) {
        if (neighbors[i] > 0) count++;
    }
    return count;
}


// Заполнить массив соседей для пикселя (r, c)
// Порядок: P2..P9 по часовой стрелке, начиная сверху:
//   P9 P2 P3
//   P8 P1 P4
//   P7 P6 P5
// Индексы в массиве neighbors[]: 0=P2, 1=P3, 2=P4, 3=P5, 4=P6, 5=P7, 6=P8, 7=P9
void getNeighbors(const Mat& img, int r, int c, uchar* neighbors) {
    neighbors[0] = img.at<uchar>(r-1, c  ); // P2 — сверху
    neighbors[1] = img.at<uchar>(r-1, c+1); // P3 — сверху-справа
    neighbors[2] = img.at<uchar>(r  , c+1); // P4 — справа
    neighbors[3] = img.at<uchar>(r+1, c+1); // P5 — снизу-справа
    neighbors[4] = img.at<uchar>(r+1, c  ); // P6 — снизу
    neighbors[5] = img.at<uchar>(r+1, c-1); // P7 — снизу-слева
    neighbors[6] = img.at<uchar>(r  , c-1); // P8 — слева
    neighbors[7] = img.at<uchar>(r-1, c-1); // P9 — сверху-слева
}


Mat zhangSuenThinning(const Mat& inputImg) {
    // Работаем с копией, нормализуем к 0/1 для удобства
    Mat img;
    inputImg.copyTo(img);

    // Приводим к 0/1 (алгоритм описан для значений 0 и 1)
    img /= 255;

    bool changed = true;

    while (changed) {
        changed = false;

        // --- Шаг 1 ---
        Mat toDelete = Mat::zeros(img.size(), CV_8UC1);

        for (int r = 1; r < img.rows - 1; r++) {
            for (int c = 1; c < img.cols - 1; c++) {

                // (0) Пиксель должен быть белым
                if (img.at<uchar>(r, c) == 0) continue;

                uchar neighbors[8];
                getNeighbors(img, r, c, neighbors);

                int B = countNeighbors(neighbors);   // B(P1)
                int A = countTransitions(neighbors); // A(P1)

                // P2, P4, P6, P8 в нашем массиве = индексы 0, 2, 4, 6
                uchar P2 = neighbors[0];
                uchar P4 = neighbors[2];
                uchar P6 = neighbors[4];
                uchar P8 = neighbors[6];

                // Условия шага 1:
                // (1): 2 <= B <= 6
                // (2): A = 1
                // (3): хотя бы один из P2, P4, P6 == 0
                // (4): хотя бы один из P4, P6, P8 == 0
                if (B >= 2 && B <= 6 &&
                    A == 1 &&
                    (P2 == 0 || P4 == 0 || P6 == 0) &&
                    (P4 == 0 || P6 == 0 || P8 == 0))
                {
                    toDelete.at<uchar>(r, c) = 1;
                }
            }
        }

        // Применяем удаление после полного прохода шага 1
        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                if (toDelete.at<uchar>(r, c)) {
                    img.at<uchar>(r, c) = 0;
                    changed = true;
                }
            }
        }

        // --- Шаг 2 ---
        toDelete = Mat::zeros(img.size(), CV_8UC1);

        for (int r = 1; r < img.rows - 1; r++) {
            for (int c = 1; c < img.cols - 1; c++) {

                if (img.at<uchar>(r, c) == 0) continue;

                uchar neighbors[8];
                getNeighbors(img, r, c, neighbors);

                int B = countNeighbors(neighbors);
                int A = countTransitions(neighbors);

                uchar P2 = neighbors[0];
                uchar P4 = neighbors[2];
                uchar P6 = neighbors[4];
                uchar P8 = neighbors[6];

                // Условия шага 2:
                // (1): 2 <= B <= 6
                // (2): A = 1
                // (3): хотя бы один из P2, P4, P8 == 0
                // (4): хотя бы один из P2, P6, P8 == 0
                if (B >= 2 && B <= 6 &&
                    A == 1 &&
                    (P2 == 0 || P4 == 0 || P8 == 0) &&
                    (P2 == 0 || P6 == 0 || P8 == 0))
                {
                    toDelete.at<uchar>(r, c) = 1;
                }
            }
        }

        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                if (toDelete.at<uchar>(r, c)) {
                    img.at<uchar>(r, c) = 0;
                    changed = true;
                }
            }
        }
    }

    img *= 255;
    return img;
}


int main() {
    string inputDir  = "/home/kosmos/Desktop/lab_7/skeleton_img/";
    string outputDir = "/home/kosmos/Desktop/lab_7/skeleton_img/";

    for (int i = 1; i <= 8; i++) {
        string filename = inputDir + to_string(i) + ".jpg";

        // Загружаем в оттенках серого
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Не удалось загрузить: " << filename << endl;
            continue;
        }

        // Бинаризация (порог Otsu)
        Mat binary;
        threshold(img, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // Скелетизация
        Mat skeleton = zhangSuenThinning(binary);

        // Сохраняем результат
        string outName = outputDir + "skeleton_" + to_string(i) + ".png";
        imwrite(outName, skeleton);
        cout << "Сохранено: " << outName << endl;

        // Показываем исходное и скелет рядом
        Mat combined;
        hconcat(binary, skeleton, combined);

        string windowName = "Original (binarized) | Skeleton — " + to_string(i) + ".jpg";
        imshow(windowName, combined);
    }

    cout << "\nНажмите любую клавишу для выхода..." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}