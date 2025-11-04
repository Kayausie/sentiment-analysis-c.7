import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, accuracy_score, roc_curve
from stock_prediction import process_data
from sklearn.utils import resample
def process_sentiment_csv(file_path, stock_data, test_ratio=0.2):
    # Read CSV
    result={}
    df = pd.read_csv(file_path)
    # Keep only the relevant columns
    # -------- Sentiment (load + filter + aggregate daily) --------
    df = pd.read_csv(file_path)
    df = df[['Date','symbols','title','sentiment_polarity','sentiment_neg','sentiment_neu','sentiment_pos']].dropna(subset=['Date','title'])
    # Drop rows with missing critical data
    # keep Apple-only rows
    df = df[df['symbols'].fillna('').str.contains("AAPL", case=False)]
    # drop fully neutral or zero-polarity rows
    df = df.query("sentiment_neu != 1 and sentiment_polarity != 0")
    # Drop duplicate rows
    df = df.drop_duplicates(subset=['Date', 'title'])
    # Convert date to datetime (for sorting or further analysis)
    df['Date'] = pd.to_datetime(df.pop('Date'), utc=True).dt.tz_localize(None).dt.date
    # Remove rows where date conversion failed
    df = df.dropna(subset=['Date'])
    # Optional: sort by date
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    daily_sent = (
        df.groupby('Date', as_index=False)[['sentiment_polarity','sentiment_neg','sentiment_pos']]
          .mean()
    )
    result['sentiment_df'] = daily_sent.copy()
    daily_sent['sentiment_polarity'] = daily_sent['sentiment_polarity'].rolling(3, min_periods=1).mean()
    daily_sent['trend3'] = daily_sent['sentiment_polarity'].diff(3)
    stock_data.reset_index(inplace=True) # ensure Date is a column, not index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None).dt.date
    
    date_col = stock_data['Date']        # keep Date aside (single-level)
    price_cols = stock_data.drop(columns=['Date']) # only the price MultiIndex part
    # select ticker level from MultiIndex (e.g., 'AAPL')
    price_cols = price_cols.xs('AAPL', axis=1, level=-1)
    stock_data = pd.concat([date_col, price_cols], axis=1)

    merged_df = pd.merge(stock_data, daily_sent, on='Date', how='inner')

    # target variable: 1 if next day close > today close, else 0
    merged_df['Target'] = (merged_df['Close'].shift(-3) > merged_df['Close']).astype(int)
    merged_df['Return'] = merged_df['Close'].pct_change()
    merged_df['MA_5'] = merged_df['Close'].rolling(window=5).mean()
    merged_df['MA_10'] = merged_df['Close'].rolling(window=10).mean()
    merged_df['Volatility'] = merged_df['Return'].rolling(window=5).std()
    merged_df['sentiment_score'] = (merged_df['sentiment_pos'] - merged_df['sentiment_neg'])/(merged_df['sentiment_pos'] + merged_df['sentiment_neg'])
    merged_df['Sentiment_lag1'] = merged_df['sentiment_polarity'].shift(1)
    merged_df['Return_1'] = merged_df['Close'].pct_change(1)
    merged_df['Return_3'] = merged_df['Close'].pct_change(3)
    merged_df['Lag_Close'] = merged_df['Close'].shift(1)
    merged_df['RSI_14'] = 100 - (100 / (1 + (
        merged_df['Close'].diff().clip(lower=0).rolling(14).mean() /
        merged_df['Close'].diff().clip(upper=0).abs().rolling(14).mean()
    )))
    merged_df.dropna(inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Return','MA_5', 'MA_10', 'Volatility',
            'sentiment_score', 'sentiment_polarity', 'Sentiment_lag1',
            'trend3',
            'Return_1', 'Return_3', 'Lag_Close', 'RSI_14'
            ]
    X = merged_df[features]
    y = merged_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    train = pd.concat([X_train, y_train], axis=1)

    majority = train[train['Target'] == 0]
    minority = train[train['Target'] == 1]

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    train_bal = pd.concat([majority, minority_upsampled])

    X_train_bal = train_bal.drop('Target', axis=1)
    y_train_bal = train_bal['Target']
    result['X_train'] = X_train
    result['X_test'] = X_test
    result['y_train'] = y_train
    result['y_test'] = y_test
    result['X_train_bal'] = X_train_bal
    result['y_train_bal'] = y_train_bal
    return result
    
def train_classification_model(trained_data, n_estimators=200, random_state=42):
    result = {}
    model = RandomForestClassifier(n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1)
    model.fit(trained_data['X_train'], trained_data['y_train'])
    result['model'] = model
    print("y_train dist:\n", trained_data['y_train'].value_counts())
    print("y_test dist:\n", trained_data['y_test'].value_counts())

    # After predictions:
    y_pred = model.predict(trained_data['X_test'])
    print("y_pred dist:", np.unique(y_pred, return_counts=True))
    result['y_pred'] = y_pred
    result['accuracy'] = accuracy_score(trained_data['y_test'], y_pred)
    result['classification_report'] = classification_report(trained_data['y_test'], y_pred)
    y_prob = model.predict_proba(trained_data['X_test'])[:,1] 
    y_pred = (y_prob > 0.4).astype(int) 
    print("Balanced Acc:", balanced_accuracy_score(trained_data['y_test'], y_pred)) 
    print(confusion_matrix(trained_data['y_test'], y_pred)) 
    print(classification_report(trained_data['y_test'], y_pred, zero_division=0))
    return model
     # Placeholder for the actual model training code
if __name__ == "__main__":
    train_classification_model(process_sentiment_csv("apple_news_data.csv", 
                                                     process_data("AAPL", "2016-02-19", "2024-11-26", test_ratio=0.2)['df']))