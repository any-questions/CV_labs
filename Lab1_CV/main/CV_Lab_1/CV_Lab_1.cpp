#include <opencv2/opencv.hpp>
#include <cmath>

int main() {
    // Параметры
    double amplitude = 100;
    double freq = 0.03;

    // Загружаем изображения
    cv::Mat background = cv::imread("images/medoed.jpg");
    cv::Mat car = cv::imread("images/gtr_r34.png", cv::IMREAD_UNCHANGED);

    // Размеры
    int width = background.cols;   // 1200
    int height = background.rows;  // 628

    // Уменьшаем машину
    cv::resize(car, car, cv::Size(80, 160));

    // Если есть прозрачность - берем маску
    cv::Mat mask;
    if (car.channels() == 4) {
        std::vector<cv::Mat> channels(4);
        cv::split(car, channels);
        mask = channels[3];
        channels.pop_back();
        cv::merge(channels, car);
    }

    // Основной цикл
    cv::Mat frame;
    for (int x = 0; x < width; x++) {
        // Копируем чистый фон каждый кадр
        background.copyTo(frame);

        // Синусоида
        int y = amplitude * cos(freq * x) + height / 2;
        y = std::max(40, std::min(y, height - 40));

        // Вставляем машину с маской
        cv::Rect roi_rect(x - 40, y - 20, 80, 160);
        if (roi_rect.x >= 0 && roi_rect.x + 80 < width) {
            cv::Mat roi = frame(roi_rect);
            car.copyTo(roi, mask);
        }

        // Рисуем траекторию (только позади машины)
        for (int i = 0; i < x; i++) {
            int ty = amplitude * cos(freq * i) + height / 2;
            cv::circle(frame, cv::Point(i, ty+60), 1, cv::Scalar(255, 0, 0), 1);
        }

        // Сохраняем в середине
        if (x == width / 2) cv::imwrite("middle.jpg", frame);

        // Показываем
        cv::imshow("Window", frame);
        if (cv::waitKey(10) == 27) break;
    }

    return 0;
}