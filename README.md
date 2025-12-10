# CV_labs
Repo for CV labs

# Краткая инструкция для винды
1. Скачивание OpenCV

    Перейдите на официальный сайт OpenCV
    Скачайте последнюю версию для Windows (например, opencv-4.x.x-windows.exe)
    Запустите скачанный exe-файл и укажите путь для распаковки (например, C:\opencv)

2. Настройка переменных среды

    Откройте "Параметры Windows" → "Система" → "О системе" → "Дополнительные параметры системы"
    Нажмите "Переменные среды"
    В системных переменнах найдите Path и добавьте:

    C:\opencv\build\x64\vc16\bin

3. Настройка проекта в Visual Studio

    Добавьте include-директории:

        ПКМ на проекте → Свойства

        VC++ Directories → Include Directories → Добавить:
        text

    C:\opencv\build\include
    C:\opencv\build\include\opencv2

Добавьте library-директории:

    VC++ Directories → Library Directories → Добавить:
    text

C:\opencv\build\x64\vc16\lib

Добавьте библиотеки:

    Linker → Input → Additional Dependencies → Добавить:
    text

opencv_world4xxd.lib

(xx - номер версии для release, например, opencv_world455.lib)
(xxd - для debug версии)
