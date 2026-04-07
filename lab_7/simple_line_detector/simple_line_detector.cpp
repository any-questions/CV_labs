#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;

// ── Параметры ───────────────────────────────────────────────
struct Params {
    int roi_top_frac   = 50;   // Верхняя граница ROI (% от высоты кадра)
    int morph_ksize    = 5;    // Размер ядра морфологии
    int min_area       = 200;  // Минимальная площадь контура в окне (px^2)
    int n_windows      = 10;   // Количество горизонтальных окон sliding window
    int win_width      = 80;   // Полуширина окна sliding window (px)
    int max_fill_pct   = 40;   // Максимальная доля белых пикселей в маске (0-100 %)
};

// ── Вычислить долю белых пикселей в маске (0.0 – 1.0) ───────
float fillRatio(const Mat& mask) {
    int total = mask.rows * mask.cols;
    if (total == 0) return 0.f;
    return (float)countNonZero(mask) / (float)total;
}

// ── Получить маску с коррекцией полярности ──────────────────
Mat correctPolarity(const Mat& raw_mask, float max_fill) {
    if (fillRatio(raw_mask) > max_fill) {
        Mat inv;
        bitwise_not(raw_mask, inv);
        return inv;
    }
    return raw_mask.clone();
}

// ── Бинарная маска по каналу V (HSV) с коррекцией полярности ─
Mat getMaskFromV(const Mat& bgr, float max_fill) {
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    vector<Mat> ch;
    split(hsv, ch);

    Mat v_norm;
    normalize(ch[2], v_norm, 0, 255, NORM_MINMAX);

    Mat mask;
    threshold(v_norm, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return correctPolarity(mask, max_fill);
}

// ── Бинарная маска по каналу L (LAB) с коррекцией полярности ─
Mat getMaskFromL(const Mat& bgr, float max_fill) {
    Mat lab;
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    vector<Mat> ch;
    split(lab, ch);

    Mat l_norm;
    normalize(ch[0], l_norm, 0, 255, NORM_MINMAX);

    Mat mask;
    threshold(l_norm, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return correctPolarity(mask, max_fill);
}

// ── Морфологическая очистка ──────────────────────────────────
Mat morphClean(const Mat& mask, int ksize) {
    int ks = (ksize % 2 == 0) ? ksize + 1 : ksize;
    ks = max(3, ks);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ks, ks));
    Mat result;
    morphologyEx(mask, result, MORPH_CLOSE, kernel);
    morphologyEx(result, result, MORPH_OPEN,  kernel);
    return result;
}

// ── Комитентный детектор: AND двух масок ────────────────────
Mat committeeMask(const Mat& mv, const Mat& ml) {
    Mat combined;
    bitwise_and(mv, ml, combined);
    return combined;
}

// ── Финальная проверка площади ───────────────────────────────
Mat areaGuard(const Mat& mask, float max_fill) {
    if (fillRatio(mask) > max_fill) {
        return Mat::zeros(mask.size(), mask.type());
    }
    return mask.clone();
}

// ── Sliding Window ───────────────────────────────────────────
struct LineResult {
    vector<Point> centers;
    int   cx_bottom;
    double angle_deg;
    bool  found;
};

LineResult slidingWindow(const Mat& mask, const Params& p) {
    LineResult res;
    res.found     = false;
    res.cx_bottom = mask.cols / 2;
    res.angle_deg = 0.0;

    int h     = mask.rows;
    int w     = mask.cols;
    int win_h = max(1, h / p.n_windows);

    Mat bottom = mask(Rect(0, h * 2/3, w, h - h * 2/3));
    Mat hist;
    reduce(bottom, hist, 0, REDUCE_SUM, CV_32S);
    Point maxLoc;
    minMaxLoc(hist, nullptr, nullptr, nullptr, &maxLoc);
    int cx = maxLoc.x;

    vector<Point> centers;

    for (int i = p.n_windows - 1; i >= 0; i--) {
        int y_top = i * win_h;

        int x_left  = max(0, cx - p.win_width / 2);
        int x_right = min(w, cx + p.win_width / 2);
        if (x_right <= x_left) {
            centers.push_back(Point(cx, y_top + win_h/2));
            continue;
        }

        Rect win_rect(x_left, y_top, x_right - x_left, win_h);
        Mat  win = mask(win_rect);

        Moments mom = moments(win, true);
        if (mom.m00 > p.min_area) {
            cx = x_left + (int)(mom.m10 / mom.m00);
            int cy = y_top + (int)(mom.m01 / mom.m00);
            centers.push_back(Point(cx, cy));
            res.found = true;
        } else {
            centers.push_back(Point(cx, y_top + win_h / 2));
        }
    }

    res.centers = centers;

    if (!centers.empty()) {
        res.cx_bottom = centers.back().x;

        if ((int)centers.size() >= 2) {
            vector<float> xs, ys;
            for (auto& pt : centers) {
                xs.push_back((float)pt.x);
                ys.push_back((float)pt.y);
            }
            int n = (int)xs.size();
            float sx  = accumulate(xs.begin(), xs.end(), 0.f);
            float sy  = accumulate(ys.begin(), ys.end(), 0.f);
            float sxy = 0, syy = 0;
            for (int i = 0; i < n; i++) {
                sxy += xs[i] * ys[i];
                syy += ys[i] * ys[i];
            }
            float slope = (n*sxy - sx*sy) / (n*syy - sy*sy + 1e-6f);
            res.angle_deg = atan((double)slope) * 180.0 / CV_PI;
        }
    }

    return res;
}

// ── Текст с непрозрачным фоном ──────────────────────────────
static void putTextBg(Mat& img, const string& text,
                      Point org, double scale, Scalar fg, int thick = 1) {
    int baseline = 0;
    Size sz = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thick, &baseline);
    rectangle(img,
              org + Point(0, baseline),
              org + Point(sz.width, -sz.height - 2),
              Scalar(0,0,0), FILLED);
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, fg, thick, LINE_AA);
}

// ── Функция для изменения размера окна с сохранением пропорций ──
Mat resizeWindowKeepAspect(const Mat& src, int target_width, int target_height) {
    Mat dst;
    double aspect = (double)src.cols / src.rows;
    int new_width, new_height;
    
    if (aspect > 1.0) {
        new_width = target_width;
        new_height = (int)(target_width / aspect);
    } else {
        new_height = target_height;
        new_width = (int)(target_height * aspect);
    }
    
    resize(src, dst, Size(new_width, new_height));
    
    // Добавляем черные поля до нужного размера
    Mat result(target_height, target_width, src.type(), Scalar(0,0,0));
    int x_offset = (target_width - new_width) / 2;
    int y_offset = (target_height - new_height) / 2;
    dst.copyTo(result(Rect(x_offset, y_offset, new_width, new_height)));
    
    return result;
}

// ── Визуализация на основном окне ─────────────────────────────
void drawResult(Mat& frame, const Mat& roi, const LineResult& lr,
                int roi_y, float fill_v, float fill_l, float fill_f,
                float max_fill) {
    int w = frame.cols;

    // Зелёный оверлей финальной маски
    Mat overlay = roi.clone();
    Mat final_mask_for_overlay = areaGuard(committeeMask(
        morphClean(getMaskFromV(roi, max_fill), 5),
        morphClean(getMaskFromL(roi, max_fill), 5)
    ), max_fill);
    
    overlay.setTo(Scalar(0, 180, 0), final_mask_for_overlay);
    addWeighted(frame(Rect(0, roi_y, roi.cols, roi.rows)),
                0.7, overlay, 0.3, 0,
                frame(Rect(0, roi_y, roi.cols, roi.rows)));

    // Центры sliding window
    for (auto& pt : lr.centers)
        circle(frame, Point(pt.x, pt.y + roi_y), 4, Scalar(0, 255, 255), -1);

    // Центральная вертикаль
    line(frame, Point(w/2, roi_y), Point(w/2, frame.rows),
         Scalar(255, 255, 0), 1, LINE_AA);

    // Аппроксимированная линия
    if (lr.found && (int)lr.centers.size() >= 2) {
        line(frame,
             Point(lr.centers.front().x, lr.centers.front().y + roi_y),
             Point(lr.centers.back().x,  lr.centers.back().y  + roi_y),
             Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Статус и метрики
    int dev = lr.cx_bottom - w / 2;
    Scalar col = lr.found ? Scalar(0,255,0) : Scalar(0,80,255);
    putTextBg(frame, lr.found ? "LINE: FOUND" : "LINE: LOST",
              Point(10, 28), 0.7, col, 2);
    putTextBg(frame, "Dev:   " + to_string(dev) + " px",
              Point(10, 54), 0.55, Scalar(220,220,220));
    putTextBg(frame, "Angle: " + to_string((int)lr.angle_deg) + " deg",
              Point(10, 76), 0.55, Scalar(220,220,220));

    // Информация о заполнении масок
    putTextBg(frame, "V fill: " + to_string((int)(fill_v * 100)) + "%",
              Point(10, 100), 0.45, Scalar(200,200,200));
    putTextBg(frame, "L fill: " + to_string((int)(fill_l * 100)) + "%",
              Point(10, 120), 0.45, Scalar(200,200,200));
    putTextBg(frame, "AND fill: " + to_string((int)(fill_f * 100)) + "%",
              Point(10, 140), 0.45, Scalar(200,200,200));

    // Граница ROI
    line(frame, Point(0, roi_y), Point(w, roi_y),
         Scalar(80, 80, 255), 1, LINE_AA);
}

// ── Отображение маски в отдельном окне с информацией ──────
void showMaskWindow(const Mat& mask, const string& windowName, 
                    const string& title, float fill_ratio, float max_fill,
                    int window_width, int window_height) {
    if (mask.empty()) return;
    
    Mat colored_mask;
    cvtColor(mask, colored_mask, COLOR_GRAY2BGR);
    
    // Добавляем информационный текст
    string info = title + " - Fill: " + to_string((int)(fill_ratio * 100)) + "%";
    if (fill_ratio > max_fill) info += " (OVER)";
    
    putTextBg(colored_mask, info, Point(10, 30), 0.6, 
              fill_ratio > max_fill ? Scalar(0,0,255) : Scalar(0,255,0), 2);
    
    // Приводим к единому размеру
    Mat resized = resizeWindowKeepAspect(colored_mask, window_width, window_height);
    imshow(windowName, resized);
}

// ── Функция для создания окна настроек ──────────────────────
void createSettingsWindow(Params& p, int window_width, int window_height) {
    namedWindow("Settings Panel", WINDOW_NORMAL);
    resizeWindow("Settings Panel", window_width, window_height);
    moveWindow("Settings Panel", 50, 600);
    
    // Создаем трекбары в отдельном окне
    createTrackbar("ROI top %", "Settings Panel", &p.roi_top_frac, 90);
    createTrackbar("Morph ksize", "Settings Panel", &p.morph_ksize, 21);
    createTrackbar("Min area", "Settings Panel", &p.min_area, 2000);
    createTrackbar("N windows", "Settings Panel", &p.n_windows, 20);
    createTrackbar("Max fill %", "Settings Panel", &p.max_fill_pct, 90);
}

int main(int argc, char** argv) {

    int VIDEO_NUMBER = 2;
    
    string vid_base = "/home/kosmos/Desktop/lab_7/line_vids/";
    string vid_path = vid_base + to_string(VIDEO_NUMBER) + ".avi";
    cout << "Открываю видео: " << vid_path << "\n";

    VideoCapture cap(vid_path);
    if (!cap.isOpened()) {
        cerr << "Не удалось открыть: " << vid_path << "\n";
        cerr << "Проверьте путь к файлам и номер видео\n";
        return -1;
    }

    Params p;

    // Получаем размеры видео
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    // Задаем единый размер для всех окон с масками
    int MASK_WINDOW_WIDTH = 480;
    int MASK_WINDOW_HEIGHT = 360;
    
    // Создаем все окна
    namedWindow("Main View", WINDOW_NORMAL);
    resizeWindow("Main View", frame_width, frame_height);
    moveWindow("Main View", 50, 50);
    
    namedWindow("V Channel Mask", WINDOW_NORMAL);
    resizeWindow("V Channel Mask", MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);
    moveWindow("V Channel Mask", frame_width + 70, 50);
    
    namedWindow("L Channel Mask", WINDOW_NORMAL);
    resizeWindow("L Channel Mask", MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);
    moveWindow("L Channel Mask", frame_width + 70, MASK_WINDOW_HEIGHT + 70);
    
    namedWindow("Final AND Mask", WINDOW_NORMAL);
    resizeWindow("Final AND Mask", MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);
    moveWindow("Final AND Mask", frame_width + 70, 2 * MASK_WINDOW_HEIGHT + 90);
    
    // Создаем окно настроек
    createSettingsWindow(p, MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);

    double fps = cap.get(CAP_PROP_FPS);
    if (fps < 1) fps = 25;
    int wait_ms = max(1, (int)(1000.0 / fps));

    cout << "FPS: " << fps << "\n";
    cout << "Размер основного окна: " << frame_width << "x" << frame_height << "\n";
    cout << "Размер окон масок: " << MASK_WINDOW_WIDTH << "x" << MASK_WINDOW_HEIGHT << "\n";
    cout << "Управление: [SPACE] пауза  [Q/Esc] выход\n";
    cout << "Настройки в окне 'Settings Panel'\n";

    bool paused = false;
    Mat  frame;

    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                cap.set(CAP_PROP_POS_FRAMES, 0);
                continue;
            }
        }

        int h = frame.rows;
        int w = frame.cols;
        float max_fill = p.max_fill_pct / 100.f;

        // ROI
        int roi_y = h * p.roi_top_frac / 100;
        Mat roi = frame(Rect(0, roi_y, w, h - roi_y)).clone();

        // Маски с автоматической коррекцией полярности
        Mat mask_v = getMaskFromV(roi, max_fill);
        Mat mask_l = getMaskFromL(roi, max_fill);

        // Морфология
        Mat mv_clean = morphClean(mask_v, p.morph_ksize);
        Mat ml_clean = morphClean(mask_l, p.morph_ksize);

        // Комитет AND
        Mat mask_final = committeeMask(mv_clean, ml_clean);

        // Финальный фильтр площади
        float fill_v = fillRatio(mv_clean);
        float fill_l = fillRatio(ml_clean);
        mask_final   = areaGuard(mask_final, max_fill);
        float fill_f = fillRatio(mask_final);

        // Sliding Window
        p.n_windows = max(2, p.n_windows);
        LineResult lr = slidingWindow(mask_final, p);

        // Отображение в основном окне
        Mat display_frame = frame.clone();
        drawResult(display_frame, roi, lr, roi_y,
                   fill_v, fill_l, fill_f, max_fill);
        imshow("Main View", display_frame);
        
        // Отображение масок в отдельных окнах (всегда показываем)
        showMaskWindow(mv_clean, "V Channel Mask", "V Channel", 
                      fill_v, max_fill, MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);
        
        showMaskWindow(ml_clean, "L Channel Mask", "L Channel", 
                      fill_l, max_fill, MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);
        
        showMaskWindow(mask_final, "Final AND Mask", "Final AND", 
                      fill_f, max_fill, MASK_WINDOW_WIDTH, MASK_WINDOW_HEIGHT);

        // Обновляем информационную панель настроек
        Mat settings_info(MASK_WINDOW_HEIGHT, MASK_WINDOW_WIDTH, CV_8UC3, Scalar(50, 50, 50));
        putText(settings_info, "CURRENT SETTINGS:", Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        putText(settings_info, ("ROI top: " + to_string(p.roi_top_frac) + "%").c_str(), 
                Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        putText(settings_info, ("Morph kernel: " + to_string(p.morph_ksize)).c_str(), 
                Point(10, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        putText(settings_info, ("Min area: " + to_string(p.min_area)).c_str(), 
                Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        putText(settings_info, ("Windows: " + to_string(p.n_windows)).c_str(), 
                Point(10, 145), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        putText(settings_info, ("Max fill: " + to_string(p.max_fill_pct) + "%").c_str(), 
                Point(10, 170), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
        
        putText(settings_info, "CONTROLS:", Point(10, 220), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        putText(settings_info, "SPACE - Pause", Point(10, 250), 
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);
        putText(settings_info, "Q/Esc - Exit", Point(10, 275), 
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);
        putText(settings_info, "VIDEO NUMBER: " + to_string(VIDEO_NUMBER), 
                Point(10, 310), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        putText(settings_info, "To change video, edit VIDEO_NUMBER", 
                Point(10, 335), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 150, 150), 1);
        putText(settings_info, "in the code and recompile", 
                Point(10, 355), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 150, 150), 1);
        
        imshow("Settings Panel", settings_info);

        int key = waitKey(wait_ms) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == ' ') paused = !paused;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}