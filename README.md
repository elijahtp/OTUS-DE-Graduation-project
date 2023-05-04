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
  - Использовать внедрение машинного обучения и построение прогнозной модели
  
# Используемые технологии: <br />
  - Spark
  - Airflow
  - Superset
  - Postgress
  - Docker

# Структура
Источники данных:
  За основу взят датасет "Superstore Sales Dataset", в котором содержится детальная информация об онлайн-заказах крупного сетевого ретейлера за 4 года. Перед загрузкой в слой с сырыми данными(STG) с датасетом были произведены некотрые трансформации: 
- Все даты изменены на актуальные даты (диапазон 2015-2018 заменен на 2020-2023).
- Перед созданием raw слоя датасет был разделен на две части: на архивную(3,5 года) и будущую (0,5 года). Архивная была сохранена в csv и загружена в слой STG Postgresql, вторая часть самостоятельно загружена в api с помощью сервиса https://retool.com (генерация api из csv)

Apache Spark:
С помощью Spark производится запуск пайплайнов ETL, выполняющих следующие задачи:
  - выгрузка данных из источника данных (csv) в слой STG Postgresql.
  - преобразование данных и загрузка из слоя STG в слой ODS Postgresql
  - создание слоя витрин (datamart) из слоя ODS 
  - проведение разведочного анализа данных (EDA), анализа сезонности и трендов и применение моделей ML(ARIMA,SARIMA) для создания предсказания о продажах в течение 7 ближайших дней. Создание соответсвующей витрины в слое datamart

Apache Airflow:
С помощью Airflow производится запуск пайплайна ETL, который:
  - производит получение свежих данных о продажах и подключается к api и производит дальнейшее преобразование и загрузку данных в оперативный слой ODS Postgress

Superset:
Дальнейшее подключение к слою datamart с помощью BI инструмента Superset



    task_load_files_into_db: загружает данные в PostgreSQL (PythonOperator, SQLalchemy)
    task_transform_data_in_db: преобразовывает данные для результирующей витрины (BashOperator, dbt)
    task_test_data_in_db: тестирует данные на отсутствие дублей и null (BashOperator, dbt)

airflow/

test_gobike_tripdata.py - DAG

test_raw_gobike_tripdata.py - task загрузки CSV в RAW

test_ods_gobike_tripdata.py - task загрузки в ODS

test_dm_station_gobike_tripdata.py - task загрузки измерения в DM

test_dm_trips_gobike_tripdata.py - task расчета агрегата в DM

test_dm_teradata_gobike_tripdata.py - task копирования витрины в Teradata

jupyter/

test_gobike_tripdata.ipynb - Jupyter Notebook для тестовой визуализации данных и лога загрузок


# Архитектура
![Image alt](https://github.com/elijahtp/OTUS-DE-Graduation-project/blob/b80b61abad1e90f9b7c9099143053ecc37028d76/scheme.png)
