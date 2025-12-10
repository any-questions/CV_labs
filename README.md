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
    text

C:\opencv\build\x64\vc16\bin

    (путь может отличаться в зависимости от версии Visual Studio)

3. Настройка проекта в Visual Studio
Для нового проекта:

    Создайте проект:

        Файл → Создать → Проект

        Выберите "Консольное приложение"

    Настройте конфигурацию:

        В верхней панели выберите "x64" (важно!)

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

    (vc14 для VS 2015, vc15 для VS 2017, vc16 для VS 2019/2022)

Добавьте библиотеки:

    Linker → Input → Additional Dependencies → Добавить:
    text

opencv_world4xx.lib

(xx - номер версии, например, opencv_world455.lib)
