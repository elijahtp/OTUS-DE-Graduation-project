# Выпускной проект по курсу OTUS Data Engineer -
# Тема: 
Система автоматической обработки, хранения и прогнозирования данных об онлайн продажах для сети супермаркетов

# О проекте:
Данный репозиторий содержит коды исполняемых файлов дипломного проект.
Код решает задачу автоматической обработки данных поступающих из внешнего API при помощи Spark и обеспечивает возможность строить дэшборды и предоставлять отчетность бизнес-пользователям.

# Цели проекта:
  - Создать систему способную выполнять реальные задачи
  - Все инструменты развернуть на serverless решениях
  - Использовать соврменные инструменты для работы с большими данными 
  - Использовать машинное обучение для построения прогнозной модели

# Архитектура
![Image alt](https://github.com/elijahtp/OTUS-DE-Graduation-project/blob/b80b61abad1e90f9b7c9099143053ecc37028d76/scheme.png)

# Используемые инструменты: 
  - Spark
  - Airflow
  - Superset
  - Postgress
  - Docker

# Структура
Источники данных: <br />
  За основу взят датасет "Superstore Sales Dataset", в котором содержится детальная информация об онлайн-заказах крупного сетевого ретейлера за 4 года. Перед загрузкой в слой с сырыми данными(STG) с датасетом были произведены некотрые трансформации: 
- Все даты изменены на актуальные даты (диапазон 2015-2018 заменен на 2020-2023).
- Перед созданием raw слоя датасет был разделен на две части: на архивную(3,5 года) и будущую (0,5 года). Архивная была сохранена в csv и загружена в слой STG Postgresql, вторая часть самостоятельно загружена в api с помощью сервиса https://retool.com (генерация api из csv)

Postgresql: <br />
На основе базы данных Postgresql производится создание слоев DWH (STG,ODS,Datamart). Реализуется в виде отдельных баз данных.

Apache Spark: <br />
С помощью Spark производится запуск пайплайнов ETL, выполняющих следующие задачи:
  - выгрузка данных из источника данных (csv) в слой STG Postgresql.
  - преобразование данных и загрузка из слоя STG в слой ODS Postgresql
  - создание слоя витрин (datamart) из слоя ODS 
  - проведение разведочного анализа данных (EDA), анализа сезонности и трендов и применение моделей ML(ARIMA,SARIMA) для создания прогноза продаж на 7 дней. Создание соответствующей витрины в слое datamart

Apache Airflow: <br />
С помощью Airflow производится запуск пайплайна ETL, который:
  - производит получение свежих данных о продажах и подключается к api и производит дальнейшее преобразование и загрузку данных в оперативный слой ODS Postgress. Загрузка производится в автоматическом режиме по расписанию (1 раз в сутки)  

Superset: <br />
С помощью BI инструмента Superset производится подключение к слою datamart, создание и отображение дашбордов со всеми интересующими бизнес пользователей графиками 

Cron: <br />
С помощью скриптов CRON задается расписание на выполнение всех пайплайнов в Spark в автоматическом режиме (1 раз в сутки - после получения свежих данных из api и обновления слоя ODS)

Serverless: <br />
Все инструменты развернуты на базе облачной инфраструктуры Yandex Cloud и представляют собой виртуальные машины с установленными операционными системами Linux Ubuntu

Docker: <br />
Для развертывания всех программных инструментов использовалась технология контейнеризации, позволяющая запускать приложения изолированно от операционной системы

# Описание пайплайнов: <br />
    spark/ <br />
      0_final_spark_stg.py: загружает данные в raw слой STG PostgreSQL (jdbc драйвер)
      1_final_spark_ods.py: преобразует данные и загружает из слоя STG в слой ODS PostgreSQL (jdbc драйвер)
      2_final_spark_datamart.py: cоздает слой витрин (datamart) из слоя ODS PostgreSQL (jdbc драйвер)
      3_final_spark_forecast.py: создание прогноза продаж на 7 дней и загрузка в витрину PostgreSQL (jdbc драйвер)
    airflow/ <br />
      final_airflow_api_to_postgress.py - task подключения к api и загрузки актуальных данных о продажах за день в оперативный слой ODS Postgress (PythonOperator,           PostgresOperator)
# Скриншоты, демонстрирующие функционал


