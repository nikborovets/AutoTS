#!/usr/bin/env python3
"""
Интеграция данных из Prometheus с AutoTS для автоматического выбора модели
"""

import pandas as pd
import numpy as np
from autots import AutoTS
from autots.data_loader import fetch_frame
import matplotlib.pyplot as plt
import warnings
import argparse
import logging
import os
import io
import sys
from contextlib import contextmanager
warnings.filterwarnings('ignore')

@contextmanager
def capture_to_logging(logger):
    """
    Контекстный менеджер для перехвата stdout и stderr и направления их в логгер.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class LoggingWriter(io.TextIOBase):
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ''

        def write(self, string):
            # Буферизуем строки до нахождения переноса строки
            self.buffer += string
            lines = self.buffer.split('\\n')
            for line in lines[:-1]:
                if line.strip():
                    self.logger.log(self.level, line.strip())
            self.buffer = lines[-1]
            return len(string)

        def flush(self):
            if self.buffer.strip():
                self.logger.log(self.level, self.buffer.strip())
            self.buffer = ''

    log_stdout = LoggingWriter(logger, logging.INFO)
    log_stderr = LoggingWriter(logger, logging.ERROR)

    sys.stdout = log_stdout
    sys.stderr = log_stderr
    try:
        yield
    finally:
        log_stdout.flush()
        log_stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'autots_run.log')
    
    # Удаляем предыдущие хендлеры, чтобы избежать дублирования при повторных запусках
    # в интерактивных средах (например, Jupyter)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Дублируем вывод в консоль
        ]
    )
    logging.info("Система логирования настроена.")


def run_prometheus_autots_prediction(
    days=7,
    start_date="2025-04-27 18:00:00", 
    end_date="2025-05-13 11:41:00",
    forecast_length=24,  # количество периодов для прогноза (в единицах step)
    use_cache=True,
    verbose=True
):
    """
    Основная функция для запуска AutoTS на данных из Prometheus
    
    Args:
        days: количество дней данных для загрузки
        start_date: начальная дата
        end_date: конечная дата  
        forecast_length: длина прогноза в периодах
        use_cache: использовать кэш
        verbose: детальный вывод
    
    Returns:
        Кортеж (model, prediction, forecast_df)
    """
    
    logging.info("🔄 Загружаю данные из Prometheus...")
    
    # Загружаем данные
    df = fetch_frame(
        days=days,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        verbose=verbose
    )
    
    if df.empty:
        raise ValueError("Не удалось загрузить данные из Prometheus")
    
    logging.info(f"✅ Загружено {len(df)} записей с {len(df.columns)} метриками")
    logging.info(f"📊 Период данных: {df.index.min()} - {df.index.max()}")
    logging.info(f"📈 Метрики: {list(df.columns)}")
    
    # Базовая очистка данных
    # Удаляем колонки с слишком большим количеством NaN
    df_clean = df.dropna(axis=1, thresh=len(df) * 0.7)  # удаляем колонки с >30% NaN
    
    if df_clean.empty:
        raise ValueError("После очистки данные пусты")
    
    logging.info(f"🧹 После очистки осталось {len(df_clean.columns)} метрик")
    
    # Настройки AutoTS для автоматического выбора модели
    model_config = {
        'forecast_length': forecast_length,
        'frequency': 'infer',  # AutoTS автоматически определит частоту
        'prediction_interval': 0.9,
        # 'max_generations': 50,  # количество поколений генетического алгоритма
        'max_generations': 5,  # количество поколений генетического алгоритма
        'num_validations': 1,   # количество кросс-валидаций
        # 'num_validations': 3,   # количество кросс-валидаций
        'validation_method': 'backwards',
        # 'models_to_validate': 0.2,  # валидировать топ 20% моделей
        'models_to_validate': 0.1,  # валидировать топ 20% моделей
        # 'model_list': 'fast',   # используем быстрые модели
        'model_list': 'superfast',   # используем быстрые модели
        # 'transformer_list': 'fast',
        'transformer_list': 'superfast',
        # 'ensemble': [
        #     'horizontal-max',   # горизонтальный ансамбль
        #     'mosaic-weighted-0-20',  # мозаичный ансамбль
        # ],
        'ensemble': None,
        'metric_weighting': {
            'smape_weighting': 5,      # SMAPE - основная метрика
            'mae_weighting': 2,        # MAE
            'rmse_weighting': 1,       # RMSE
            'spl_weighting': 3,        # Scaled Pinball Loss
            'containment_weighting': 0.1,  # покрытие интервалов
            'runtime_weighting': 0.05,     # время выполнения
        },
        'n_jobs': 'auto',
        'verbose': 1 if verbose else 0,
        'random_seed': 2024,
        'no_negatives': True,  # исключаем отрицательные прогнозы (для метрик)
    }
    
    logging.info("🤖 Инициализирую AutoTS для автоматического выбора модели...")
    
    # Создаем модель AutoTS
    model = AutoTS(**model_config)
    
    logging.info("🏋️ Начинаю обучение (это может занять время)...")
    
    # Обучаем модель
    with capture_to_logging(logging.getLogger()):
        model = model.fit(df_clean)
    
    logging.info("✅ Обучение завершено!")
    logging.info(f"🎯 Лучшая модель: {model.best_model_name}")
    logging.info(f"📋 Параметры модели: {model.best_model_params}")
    
    # Делаем прогноз
    logging.info("🔮 Делаю прогноз...")
    with capture_to_logging(logging.getLogger()):
        prediction = model.predict()
    
    # Получаем результаты
    forecast_df = prediction.forecast
    upper_forecast = prediction.upper_forecast  
    lower_forecast = prediction.lower_forecast
    
    logging.info(f"✅ Прогноз готов на {len(forecast_df)} периодов")
    
    # Выводим статистику модели
    logging.info("📊 Статистика лучшей модели:")
    results = model.results()
    best_model_results = results[results['ID'] == model.best_model_id]
    
    if not best_model_results.empty:
        metrics = ['smape', 'mae', 'rmse', 'spl']
        for metric in metrics:
            if metric in best_model_results.columns:
                value = best_model_results[metric].iloc[0]
                logging.info(f"  {metric.upper()}: {value:.4f}")
    
    return model, prediction, forecast_df


def plot_results(df_historical, forecast_df, upper_forecast, lower_forecast, 
                metric_name=None, save_plot=False):
    """
    Построение графика исторических данных и прогноза
    """
    if metric_name is None:
        metric_name = df_historical.columns[0]
    
    plt.figure(figsize=(15, 8))
    
    # Исторические данные
    plt.plot(df_historical.index, df_historical[metric_name], 
             label='Исторические данные', color='blue', linewidth=1.5)
    
    # Прогноз
    plt.plot(forecast_df.index, forecast_df[metric_name], 
             label='Прогноз', color='red', linewidth=2)
    
    # Доверительные интервалы
    plt.fill_between(forecast_df.index, 
                     lower_forecast[metric_name], 
                     upper_forecast[metric_name],
                     alpha=0.3, color='red', label='Доверительный интервал')
    
    plt.title(f'Прогноз для метрики: {metric_name}')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'forecast_{metric_name}.png', dpi=300, bbox_inches='tight')
        logging.info(f"📊 График сохранен как forecast_{metric_name}.png")
    
    plt.show()


def main():
    """
    Основная функция для запуска прогнозирования
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Запуск AutoTS для данных Prometheus.")
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'initial', 'finetune'],
        help="Этап выполнения: 'initial' - только первичный поиск, 'finetune' - только тонкая настройка, 'all' - оба этапа."
    )
    parser.add_argument(
        '--template-path',
        type=str,
        default='prometheus_autots_template.csv',
        help="Путь к файлу шаблона для тонкой настройки."
    )
    args = parser.parse_args()

    try:
        if args.step in ['initial', 'all']:
            logging.info("🚀 Этап 1: Первичный поиск лучшей модели...")
            # Запускаем прогнозирование
            model, prediction, forecast_df = run_prometheus_autots_prediction(
                days=7,
                start_date="2025-04-27 18:00:00",
                end_date="2025-05-13 11:41:00",
                forecast_length=48,
                use_cache=True,
                verbose=True
            )

            # Получаем исторические данные для графика
            df_historical = fetch_frame(
                days=7,
                start_date="2025-04-27 18:00:00",
                end_date="2025-05-13 11:41:00",
                use_cache=True,
                verbose=False
            )

            # Строим график для первой метрики
            if not forecast_df.empty and not df_historical.empty:
                metric_name = forecast_df.columns[0]
                plot_results(
                    df_historical,
                    prediction.forecast,
                    prediction.upper_forecast,
                    prediction.lower_forecast,
                    metric_name=metric_name,
                    save_plot=True
                )

            # Сохраняем результаты
            forecast_df.to_csv('prometheus_forecast.csv')
            logging.info("💾 Прогноз сохранен в prometheus_forecast.csv")

            # Сохраняем шаблон модели для будущего использования
            model.export_template(args.template_path, models='best', n=5)
            logging.info(f"💾 Шаблон модели сохранен в {args.template_path}")

            logging.info("🎉 Этап 1 (первичный поиск) завершен успешно!")

        if args.step in ['finetune', 'all']:
            if args.step == 'finetune':
                logging.info("🚀 Этап 2: Тонкая настройка моделей из шаблона...")
            
            # Запускаем тонкую настройку на основе полученного шаблона
            run_finetuning(
                template_path=args.template_path,
                days=7,
                start_date="2025-04-27 18:00:00",
                end_date="2025-05-13 11:41:00",
                forecast_length=48
            )

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}", exc_info=True)
        raise


def run_finetuning(
    template_path='prometheus_autots_template.csv',
    days=7,
    start_date="2025-04-27 18:00:00",
    end_date="2025-05-13 11:41:00",
    forecast_length=48,
    use_cache=True,
    verbose=True
):
    """
    Запускает тонкую настройку моделей из шаблона.
    
    Args:
        template_path: Путь к файлу шаблона с лучшими моделями.
        days, start_date, end_date: Параметры для загрузки данных.
        forecast_length: Длина прогноза.
        use_cache: Использовать кэш.
        verbose: Детальный вывод.
        
    Returns:
        Кортеж (model, prediction, forecast_df)
    """
    logging.info("\n\n" + "="*50)
    logging.info("⚙️  Запускаю тонкую настройку лучших моделей из шаблона...")
    logging.info(f"📄 Шаблон: {template_path}")
    logging.info("="*50 + "\n")

    # Загружаем данные так же, как и в основной функции
    df = fetch_frame(
        days=days,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        verbose=verbose
    )
    
    if df.empty:
        raise ValueError("Не удалось загрузить данные из Prometheus для тюнинга")
    
    df_clean = df.dropna(axis=1, thresh=len(df) * 0.7)
    if df_clean.empty:
        raise ValueError("После очистки данные для тюнинга пусты")

    logging.info(f"✅ Для тюнинга загружено {len(df_clean)} записей")

    # Конфигурация для тонкой настройки
    # Мы увеличиваем max_generations и num_validations для более тщательного поиска
    model_config = {
        'forecast_length': forecast_length,
        'frequency': 'infer',
        'prediction_interval': 0.9,
        'max_generations': 20,  # Больше поколений
        'num_validations': 2,   # Больше валидаций
        'validation_method': 'backwards',
        'models_to_validate': 0.3, # Валидируем больше моделей из топа
        'ensemble': ['horizontal-max', 'mosaic-weighted-0-20'], # Попробуем ансамбли
        'metric_weighting': {
            'smape_weighting': 5,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'spl_weighting': 3,
            'containment_weighting': 0.1,
            'runtime_weighting': 0.05, # Меньше смотрим на время
        },
        'n_jobs': 'auto',
        'verbose': 1 if verbose else 0,
        'random_seed': 2024,
        'no_negatives': True,
    }

    logging.info("🤖 Инициализирую AutoTS для тонкой настройки...")
    
    model = AutoTS(**model_config)
    
    # Заменяем список моделей для поиска на те, что в шаблоне (`method='only'`).
    # Это позволяет сфокусироваться на тонкой настройке лучших кандидатов.
    logging.info(f"Импортирую модели из {template_path} для тонкой настройки...")
    model_after_import = model.import_template(template_path, method='only')

    # Проверяем, не вернул ли import_template ошибку вместо объекта модели
    if not isinstance(model_after_import, AutoTS):
        logging.error("❌ Не удалось импортировать шаблон. Похоже, метод import_template вернул ошибку.")
        logging.error("Пожалуйста, проверьте файл prometheus_autots_template.csv на отсутствие пустых строк в конце.")
        if isinstance(model_after_import, Exception):
            raise model_after_import
        raise TypeError(f"import_template вернул неожиданный тип: {type(model_after_import)}")
    
    model = model_after_import

    logging.info("🏋️ Начинаю обучение (это может занять больше времени, чем первый прогон)...")
    
    # Обучаем модель
    with capture_to_logging(logging.getLogger()):
        model = model.fit(df_clean)
    
    logging.info("✅ Тонкая настройка завершена!")
    logging.info(f"🎯 Лучшая модель после тюнинга: {model.best_model_name}")
    logging.info(f"📋 Параметры модели: {model.best_model_params}")
    
    # Делаем прогноз
    logging.info("🔮 Делаю прогноз после тюнинга...")
    with capture_to_logging(logging.getLogger()):
        prediction = model.predict()
    
    # Получаем результаты
    forecast_df = prediction.forecast
    
    logging.info(f"✅ Прогноз готов на {len(forecast_df)} периодов")
    
    # Выводим статистику модели
    logging.info("📊 Статистика лучшей модели после тюнинга:")
    results = model.results()
    best_model_results = results[results['ID'] == model.best_model_id]
    
    if not best_model_results.empty:
        metrics = ['smape', 'mae', 'rmse', 'spl']
        for metric in metrics:
            if metric in best_model_results.columns:
                value = best_model_results[metric].iloc[0]
                logging.info(f"  {metric.upper()}: {value:.4f}")

    # Строим график для первой метрики
    if not forecast_df.empty and not df_clean.empty:
        metric_name = forecast_df.columns[0]
        plot_results(
            df_clean, 
            prediction.forecast,
            prediction.upper_forecast,
            prediction.lower_forecast,
            metric_name=metric_name,
            save_plot=True
        )

    # Сохраняем результаты
    forecast_df.to_csv('prometheus_forecast_tuned.csv')
    logging.info("💾 Прогноз после тюнинга сохранен в prometheus_forecast_tuned.csv")
    
    # Сохраняем шаблон модели для будущего использования
    model.export_template('prometheus_autots_template_tuned.csv', models='best', n=5)
    logging.info("💾 Шаблон тюнингованной модели сохранен в prometheus_autots_template_tuned.csv")
    
    logging.info("🎉 Тонкая настройка завершена успешно!")
    
    return model, prediction, forecast_df


if __name__ == "__main__":
    # Запускаем прогнозирование
    main() 