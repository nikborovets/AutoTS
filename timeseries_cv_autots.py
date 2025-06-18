#!/usr/bin/env python3
"""
Скрипт для кросс-валидации модели AutoTS с использованием TimeSeriesSplit.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from autots import AutoTS
from autots.data_loader import fetch_frame
# Для расчета метрик будем использовать встроенные в AutoTS функции,
# но их можно импортировать и отдельно из autots.evaluator.metrics
import logging
import os
import sys
import io
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager

from autots.viz import plot_history_forecast

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
    
    # Имя лог-файла с отметкой времени, чтобы не перезаписывать предыдущие логи
    log_file = os.path.join(
        log_dir,
        f"autots_cv_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),  # по умолчанию 'a', файл уникален по времени
            logging.StreamHandler()
        ]
    )
    logging.info("Система логирования настроена.")

def run_timeseries_cv(
    template_path='prometheus_autots_template_tuned.csv',
    n_splits=5,
    forecast_length=48,
    days=7,
    start_date="2025-04-27 18:00:00",
    end_date="2025-05-13 11:41:00"
):
    """
    Выполняет кросс-валидацию для модели AutoTS на основе шаблона.
    """
    if not os.path.exists(template_path):
        logging.error(f"❌ Шаблон модели '{template_path}' не найден.")
        logging.error("Пожалуйста, сначала запустите 'prometheus_autots_integration.py' для его создания.")
        return

    # 1. Загрузка данных
    logging.info("🔄 Загружаю данные из Prometheus...")
    df = fetch_frame(
        days=days,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
        verbose=False
    )
    df_clean = df.dropna(axis=1, thresh=len(df) * 0.7)
    logging.info(f"✅ Загружено {len(df_clean)} записей с {len(df_clean.columns)} метриками: {list(df_clean.columns)}")
    logging.info(f"🗓️ Период данных: {df_clean.index.min()} - {df_clean.index.max()}")

    # 2. Настройка TimeSeriesSplit
    # Размер тестовой выборки (test_size) для каждой складки равен горизонту прогноза.
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=forecast_length)
    logging.info(f"⚙️ Настроена кросс-валидация TimeSeriesSplit с {n_splits} сплитами и размером теста {forecast_length}.")

    all_results = []
    
    # Создаём директорию под графики
    plots_dir = Path("plots_cv")
    plots_dir.mkdir(exist_ok=True)

    # 3. Цикл по сплитам
    for i, (train_index, test_index) in enumerate(tscv.split(df_clean)):
        train_df = df_clean.iloc[train_index]
        test_df = df_clean.iloc[test_index]

        logging.info("\n" + "="*50)
        logging.info(f"🚀 Сплит {i + 1}/{n_splits}")
        logging.info(f"   Обучение: {len(train_df)} точек ({train_df.index.min()} -> {train_df.index.max()})")
        logging.info(f"   Тест:     {len(test_df)} точек ({test_df.index.min()} -> {test_df.index.max()})")
        logging.info("="*50)

        # 4. Настройка и обучение модели AutoTS
        # Используем параметры из скрипта тонкой настройки, но без поиска новых моделей (max_generations=0)
        model_config = {
            'forecast_length': forecast_length,
            'frequency': 'infer',
            'prediction_interval': 0.9,
            'max_generations': 0,  # Не ищем новые модели, используем шаблон
            'num_validations': 0,  # Внутренняя валидация не нужна, т.к. делаем внешнюю CV
            'ensemble': None,
            'n_jobs': 'auto',
            'verbose': 1,  # Детальный вывод модели
            'random_seed': 2024 + i, # Меняем seed для разнообразия, если в моделях есть случайность
            'no_negatives': True,
        }
        
        model = AutoTS(**model_config)
        
        logging.info(f"📄 Импорт моделей из шаблона: {template_path}")
        model = model.import_template(template_path, method='only')
        
        # на случай, если AutoTS всё-таки полезет собирать горизонтальный ансамбль
        model.ensemble_templates = pd.DataFrame()
        
        try:
            logging.info("🏋️ Обучаю модель на данных для обучения...")
            with capture_to_logging(logging.getLogger()):
                model.fit(train_df, future_regressor=None)

            # чекпоинт шаблона после обучения
            model.export_template(str(plots_dir / f"template_split_{i+1}.csv"), models='best', n=50)

            logging.info("🔮 Делаю прогноз на тестовый период...")
            with capture_to_logging(logging.getLogger()):
                prediction = model.predict(future_regressor=None)

            # 5. Оценка результатов
            logging.info("📈 Оцениваю точность прогноза...")
            evaluation = prediction.evaluate(actual=test_df)
            score_df = evaluation.per_series_metrics
            score_df['split'] = i + 1
            all_results.append(score_df)
        except Exception as e:
            logging.error(f"❌ Ошибка в сплите {i+1}: {e}", exc_info=True)
            continue

        # 6. Строим графики прогноза vs факта для каждой серии
        ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for col in prediction.forecast.columns:
            y_pred_series = prediction.forecast[col]
            y_true_series = test_df[col]
            # Покажем последние 30 минут истории перед началом прогноза
            hist_start = y_pred_series.index[0] - pd.Timedelta(minutes=30)
            history_series = train_df.loc[hist_start : y_pred_series.index[0], col]

            if 'prediction' in locals():
                plot_history_forecast(
                    history=history_series,
                    forecast=y_pred_series,
                    actual=y_true_series,
                    title=f"Fold {i + 1} — {col}",
                    filename=str(plots_dir / f"cv_fold_{i + 1}_{col}_{ts_stamp}.png"),
                )

    # 6. Агрегация и вывод итоговых результатов
    logging.info("\n" + "="*50)
    logging.info("🏆 Итоговые результаты кросс-валидации")
    logging.info("="*50)

    if not all_results:
        logging.error("Нет валидных результатов ни для одного сплита.")
        return

    final_results_df = pd.concat(all_results, ignore_index=False)
    
    # Считаем средние метрики и стандартное отклонение по всем сплитам для каждой серии
    avg_metrics = final_results_df.groupby(final_results_df.index).mean().drop('split', axis=1)
    std_metrics = final_results_df.groupby(final_results_df.index).std().drop('split', axis=1)

    logging.info("\nСредние метрики по сериям (AVG ± STD):\n")
    
    # Красивый вывод с AVG ± STD
    for series_name in avg_metrics.index:
        logging.info(f"➡️  {series_name}:")
        for metric in avg_metrics.columns:
            avg = avg_metrics.loc[series_name, metric]
            std = std_metrics.loc[series_name, metric]
            logging.info(f"    {metric:<10}: {avg:.2f} ± {std:.2f}")

    # Общие средние метрики по всем сериям и сплитам
    overall_avg = avg_metrics.mean()
    logging.info("\n" + "-"*30)
    logging.info("📊 Общие средние метрики по всем сериям:")
    logging.info(overall_avg.round(2).to_string())
    logging.info("-"*30)
    
    logging.info("\n🎉 Кросс-валидация успешно завершена!")


def main():
    """Основная функция для запуска."""
    setup_logging()
    
    try:
        run_timeseries_cv(
            template_path='prometheus_autots_template_tuned.csv',
            n_splits=5,
            forecast_length=48, # Этот параметр должен соответствовать test_size в TimeSeriesSplit
            days=7,
            start_date="2025-04-27 18:00:00",
            end_date="2025-05-13 11:41:00"
        )
    except Exception as e:
        logging.error(f"❌ Произошла критическая ошибка: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 