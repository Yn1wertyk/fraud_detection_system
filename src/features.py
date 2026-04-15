import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional

def build_features(df: pd.DataFrame, single: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Побудова ознак для виявлення фінансових махінацій
    
    Args:
        df: DataFrame з транзакціями
        single: True якщо обробляємо одну транзакцію для інференсу
    
    Returns:
        Tuple[features_df, target_series]
    """
    df = df.copy()
    
    # Базові ознаки
    df['amount_log'] = np.log1p(df['amount'])
    df['hour'] = pd.to_datetime(df['tx_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['tx_time']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Часові ознаки (time-aware)
    df['tx_time'] = pd.to_datetime(df['tx_time'])
    df = df.sort_values('tx_time')
    
    # Ознаки по користувачу за 1 годину
    df['user_tx_count_1h'] = df.groupby('user_id').apply(
        lambda group: group.set_index('tx_time').rolling('1h')['amount'].count().shift(1).fillna(0)
    ).reset_index(level=0, drop=True).reindex(df.index).fillna(0)
    
    # Ознаки по мерчанту за 1 годину  
    df['merchant_tx_count_1h'] = df.groupby('merchant').apply(
        lambda group: group.set_index('tx_time').rolling('1h')['amount'].count().shift(1).fillna(0)
    ).reset_index(level=0, drop=True).reindex(df.index).fillna(0)
    
    # Ознаки по користувачу за 24 години
    df['user_amount_mean_24h'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
    )
    df['user_amount_std_24h'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
    )
    
    # Правила-флаги
    df['high_amount_flag'] = (df['amount'] > 1000).astype(int)
    df['new_device_flag'] = df.groupby('user_id')['device_id'].transform(
        lambda x: (~x.isin(x.shift(1).dropna())).astype(int)
    )
    
    # Відстань від середнього
    df['amount_vs_user_mean'] = df['amount'] / (df['user_amount_mean_24h'] + 1e-8)
    
    # Вибираємо фічі для моделі
    feature_cols = [
        'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_tx_count_1h', 'user_amount_mean_24h', 'user_amount_std_24h',
        'merchant_tx_count_1h', 'high_amount_flag', 'new_device_flag',
        'amount_vs_user_mean'
    ]
    
    # Заповнюємо NaN
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    X = df[feature_cols]
    y = df['is_fraud'] if 'is_fraud' in df.columns and not single else None
    
    return X, y

def prepare_single_transaction(tx_dict: dict, historical_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Підготовка однієї транзакції для інференсу
    """
    tx_df = pd.DataFrame([tx_dict])
    
    # Якщо є історичні дані, об'єднуємо для розрахунку агрегатів
    if historical_data is not None:
        combined_df = pd.concat([historical_data, tx_df], ignore_index=True)
        X, _ = build_features(combined_df, single=True)
        return X.iloc[-1:].copy()  # Повертаємо тільки останню транзакцію
    else:
        X, _ = build_features(tx_df, single=True)
        return X
