# RetailRocket Recommender System

## Описание
Это pet-проект по построению рекомендательной системы (Recommender System) на основе датасета Retailrocket от Kaggle. Проект демонстрирует полный ML-пайплайн: от EDA и предобработки данных до моделирования, оценки и deployment. Цель — подготовка к стажировкам в компаниях вроде VK (рекомендации контента), Сбера (персонализация продуктов) и МТС (рекомендации услуг), где recsys играют ключевую роль.

Датасет включает ~2.7M событий взаимодействия пользователей с товарами (просмотры, добавления в корзину, покупки) плюс метаданные items. Мы фокусируемся на implicit feedback для hybrid моделей (collaborative + content-based).

## Цели проекта
- Построить production-ready систему с метриками Precision@K, Recall@K, NDCG.
- Решить проблемы sparsity, cold start, temporal splits.
- Развернуть API для рекомендаций.

## Стек технологий
- **Язык**: Python 3.12
- **Библиотеки**: Pandas, NumPy, Surprise, LightFM, FastAPI, Matplotlib/Seaborn
- **Инструменты**: Jupyter, Git, Conda (для env), VS Code
- **Deployment**: FastAPI + Uvicorn (Docker-ready)

## Структура проекта
```
retailrocket-recsys/
├── src/                  # Основной код (модули)
│   ├── data_processing/  # EDA, preprocessing
│   ├── models/           # Модели (CF, hybrid)
│   ├── evaluation/       # Метрики, tuning
│   └── deployment/       # API endpoints
├── notebooks/            # Эксперименты (EDA.ipynb)
├── data/                 # Raw и processed данные
├── models/               # Сохранённые модели
├── configs/              # YAML с параметрами
├── tests/                # Unit-тесты
├── requirements.txt      # Зависимости
├── README.md            # Это!
└── .gitignore
```

## Запуск
1. `conda create -n retail-rocket-recsys-venv python=3.12 && conda activate retail-rocket-recsys-venv`
2. `conda install -c conda-forge -r requirements.txt`
3. Скачай данные в `data/raw/` с Kaggle.
4. `jupyter notebook notebooks/eda.ipynb` для старта.
5. Для API: `uvicorn src.deployment.app:app --reload`
