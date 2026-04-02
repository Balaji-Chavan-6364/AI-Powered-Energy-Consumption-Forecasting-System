# ==============================================
# AI-Powered Energy Consumption Forecasting System
# ==============================================

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_synthetic_energy_data(days=365, seed=42):
    """Generate a synthetic daily energy consumption dataset."""
    np.random.seed(seed)

    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    temperature = 20 + 10 * np.sin(2 * np.pi * (np.arange(days) / 365)) + np.random.normal(0, 2, days)
    humidity = 50 + 20 * np.cos(2 * np.pi * (np.arange(days) / 365)) + np.random.normal(0, 5, days)
    wind_speed = np.clip(np.random.normal(10, 3, days), 0, None)

    energy_consumption = (
        40
        + 1.5 * temperature
        + 0.4 * humidity
        - 0.2 * wind_speed
        + 8 * np.sin(2 * np.pi * (np.arange(days) / 7))
        + np.random.normal(0, 4, days)
    )

    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'energy_consumption': energy_consumption,
    })

    logging.info('Generated synthetic dataset: %d rows', len(df))
    return df


def load_data_from_csv(csv_file):
    """Load dataset from CSV and validate required columns."""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f'CSV file not found: {csv_file}')

    df = pd.read_csv(csv_file, parse_dates=['date'], dayfirst=False)
    required = {'date', 'temperature', 'humidity', 'wind_speed', 'energy_consumption'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Missing required columns in CSV: ' + ', '.join(sorted(missing)))

    logging.info('Loaded dataset from CSV: %s', csv_file)
    return df


def add_time_features(df):
    """Add day-of-month and month features."""
    df = df.copy()
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df = df.drop(columns=['date'])
    return df


def analyze_data(df):
    """Display basic information and summary stats."""
    logging.info('Dataset shape: %s', df.shape)
    logging.info('\n%s', df.info())
    logging.info('\n%s', df.describe())


def plot_data(df, output_dir=None):
    """Plot energy consumption and feature relationships."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['energy_consumption'], label='Energy')
    ax.set_title('Energy Consumption Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Energy')
    ax.legend()

    if output_dir is not None:
        out_path = Path(output_dir) / 'energy_time_series.png'
        fig.savefig(out_path)
        logging.info('Saved plot: %s', out_path)
    else:
        plt.show()
    plt.close(fig)


def prepare_train_test(df, test_size=0.2, random_state=42):
    """Split dataset into train/test sets."""
    X = df.drop(columns=['energy_consumption'])
    y = df['energy_consumption']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, use_grid_search=False):
    """Train random forest model with optional grid search."""
    if use_grid_search:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 8, 12],
            'min_samples_split': [2, 4, 6],
        }
        base = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(base, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        logging.info('GridSearchCV best params: %s', grid.best_params_)
        return best

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics and predictions."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info('Evaluation metrics: RMSE=%.4f, R2=%.4f', rmse, r2)
    return {'rmse': rmse, 'r2': r2}, y_pred


def plot_predictions(y_test, y_pred, output_dir=None):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual Energy')
    ax.set_ylabel('Predicted Energy')
    ax.set_title('Actual vs Predicted Energy Consumption')

    if output_dir is not None:
        out_path = Path(output_dir) / 'actual_vs_predicted.png'
        fig.savefig(out_path)
        logging.info('Saved plot: %s', out_path)
    else:
        plt.show()
    plt.close(fig)


def save_model(model, path='energy_model.joblib'):
    """Persist model to disk."""
    joblib.dump(model, path)
    logging.info('Saved model to %s', path)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Energy consumption forecasting')
    parser.add_argument('--csv', type=str, help='Optional CSV input dataset')
    parser.add_argument('--days', type=int, default=365, help='Number of synthetic days to generate')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search hyperparameter tuning')
    parser.add_argument('--output-dir', type=str, default=None, help='Save plots and model output to directory')
    args = parser.parse_args()

    if args.csv:
        df = load_data_from_csv(args.csv)
    else:
        df = generate_synthetic_energy_data(days=args.days)

    if 'date' in df.columns:
        df = add_time_features(df)

    analyze_data(df)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    plot_data(df, output_dir=args.output_dir)

    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=args.test_size)
    model = train_model(X_train, y_train, use_grid_search=args.grid_search)

    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print('\nModel Performance:')
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")

    plot_predictions(y_test, y_pred, output_dir=args.output_dir)

    model_path = args.output_dir + '/energy_model.joblib' if args.output_dir else 'energy_model.joblib'
    save_model(model, model_path)

    print('\nProject Completed Successfully!')


if __name__ == '__main__':
    main()
