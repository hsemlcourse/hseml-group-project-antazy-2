# ML Project - Прогнозирование скорости "усыновления" животных из приютов

**Студент:** Суслова Анастасия Алексеевна  
**Группа:** БИВ231

## Описание задачи

Цель проекта - предсказать класс `AdoptionSpeed` для животных из приютов.

- **Датасет:** [PetFinder.my Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction/data)
- **Тип задачи:** мультиклассовая классификация
- **Основная метрика:** Macro F1  
  Macro F1 выбрана как основная, так как она учитывает качество по каждому
  классу независимо и лучше отражает баланс качества, чем accuracy.

## Структура репозитория

```text
.
├── data
│   ├── processed               # Предобработанные наборы
│   └── raw                     # Сырые данные из Kaggle
├── models                      # Сохраненные модели (joblib)
├── notebooks
│   ├── 01_download_data.py  # Скачивание и распаковка данных
│   ├── 02_preprocessing.ipynb  # EDA, визуализация, очистка и split
│   └── 03_modeling.ipynb       # Модели и метрики
├── presentation                # Материалы для защиты
├── report
│   └── report.md               # Финальный отчет
├── tests
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
# 1) Создать и активировать виртуальное окружение
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# 2) Установить зависимости
pip install -r requirements.txt
```

## Работа с данными

1. Положить `kaggle.json` в корень проекта.
2. Скачать и распаковать данные:

```bash
python notebooks/01_download_data.py
```

После выполнения:
- сырые данные лежат в `data/raw/`,
- zip-архив также сохраняется в `data/raw/`.

## Запуск моделирования

Запустить предобработку:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.executed.ipynb
```

Запустить базовый цикл обучения и сравнения моделей:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --output 03_modeling.executed.ipynb
```

Что делает ноутбук `03_modeling.ipynb`:
- читает данные из `data/raw/train/train.csv`,
- выполняет очистку и split на train/val/test,
- обучает набор baseline/классических моделей,
- считает `accuracy` и `macro_f1`,
- сохраняет лучшую модель в `models/best_model.joblib`.


## Критерии выполнения проекта

### 1) Обработка и подготовка данных (13 баллов)

- описание источника и структуры данных;
- обработка пропусков, дублей, выбросов и типов;
- feature engineering и обоснование новых признаков;
- визуализации и анализ зависимостей;
- корректный split и защита от data leakage.

### 2) Моделирование и эксперименты (7 баллов)

- baseline-модель;
- не менее 4-5 экспериментов с разными моделями;
- таблица экспериментов с гипотезами, параметрами и метриками;
- обоснованный выбор финальной модели.

### 3) Качество кода и воспроизводимость (5 баллов)

- понятная структура проекта;
- фиксированный `random_state` в экспериментах;
- зависимости в `requirements.txt`;
- ноутбуки запускаются и показывают повторяемый результат;

## Результаты

Итоговые метрики и сравнение моделей фиксируются:
- в ноутбуке `notebooks/03_modeling.ipynb`,
- в отчете `report/report.md`,
- в выводе выполненного ноутбука.

## Отчет

Финальный отчет: [`report/report.md`](report/report.md)
