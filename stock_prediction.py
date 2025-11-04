
from collections import deque
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import finnhub
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from parameters import *
from visualizations import visualize_sentiment
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, BertTokenizer
import torch
from train_classification_model import process_sentiment_csv
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


def shuffle_in_unison(a, b):
    # in the first line, we are going to create a random state using numpy's random get_state function
    # this state will be used to ensure that we shuffle both arrays in the same way
    state = np.random.get_state()
    #shuffle the first array
    np.random.shuffle(a)
    #set the random state back for the second array
    np.random.set_state(state)
    #this shuffle will be the same as the first one, as they shared the same random state
    np.random.shuffle(b)
def process_data(ticker, startDate, endDate, test_ratio, scale=True, split_by_date=True, shuffle=True, store=True,
                 feature_columns=['Close', 'Volume', 'Open', 'High', 'Low'], nan_strategy="drop",
                 n_steps=50, lookup_step=1, horizon=1):
    # containers
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    result = {}
    X, y = [], []
    if startDate is None or endDate is None:
        raise Exception("Please provide a start date and end date in the format YYYY-MM-DD")
    if nan_strategy not in ["drop", "ffill", "bfill"]:
        raise ValueError("nan_strategy must be one of: 'drop', 'ffill', 'bfill'")
    # Load data
    if isinstance(ticker, str):
        data = yf.download(ticker, startDate, endDate, auto_adjust=False)
    elif isinstance(ticker, pd.DataFrame):
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instance")
    # NaN handling
    if nan_strategy == "drop":
        data.dropna(inplace=True)
    else:  # 'ffill' or 'bfill'
        data[feature_columns] = data[feature_columns].fillna(method=nan_strategy)
    # Save a copy of original DF
    result['df'] = data.copy()
    if store:
        data.to_csv("data.csv", index=True)
    # Ensure date column exists
    if "date" not in data.columns:
        data["date"] = data.index
    # Scale features if requested
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            # fit on 2D, write back 1D to keep Series shape and avoid (n,1)
            data[column] = scaler.fit_transform(data[[column]].to_numpy()).ravel()
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
    # -------------------------------
    # Multi-step targets (NO shift)
    # Predict next `horizon` closes for each window of length n_steps
    # -------------------------------
    # Keep 'date' as datetime64, only features are float
    feature_block = data[feature_columns + ["date"]].to_numpy()
    close = data["Close"].to_numpy(dtype=np.float32).reshape(-1)  # force 1-D
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    N = len(feature_block)

    for i, entry in enumerate(feature_block):
        sequences.append(entry)
        if len(sequences) == n_steps:
            start = i + 1
            end = start + horizon
            if end <= len(close):  # only keep samples with full horizon available
                target_vec = close[start:end]  # shape (horizon,)
                sequence_data.append([np.array(sequences), target_vec.astype(np.float32)])
            # else: insufficient future steps; skip

    # Build last_sequence (for out-of-sample forecasting)
    # Take the last n_steps timesteps from the deque and drop the date column
    if len(sequences) == n_steps:
        last_seq = np.array([row[:len(feature_columns)] for row in list(sequences)], dtype=np.float32)
        result['last_sequence'] = last_seq
    else:
        result['last_sequence'] = None

    # Pack arrays
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # allow mixed types (features + datetime) for now
    X = np.array(X)                  # dtype will be object (ok)
    y = np.array(y, dtype=np.float32)
    # Split
    if split_by_date:
        train_samples = int((1 - test_ratio) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=shuffle)
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = X_train, X_test, y_train, y_test

    # Map test samples to their last input date (X_test[:, -1, -1] is the date column)
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    # Remove date column from X_* and cast
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"]  = result["X_test"][:,  :, :len(feature_columns)].astype(np.float32)

    return result
# To do: build a function to collect textual data aligned with stock data
# Constraints: date range must be within 1 year from today
def scrape_news(ticker, startDate, endDate):
    
    #Collect news from finnhub
    finnhub_client = finnhub.Client(api_key="d43ad2hr01qvk0javkkgd43ad2hr01qvk0javkl0")
    news = finnhub_client.company_news(ticker, _from=startDate, to=endDate)
    #Preprocess news data
    df = pd.DataFrame(news)
    if df.empty:
        raise ValueError("No news data found for the given ticker and date range.")
    # Convert UNIX timestamps → UTC → Melbourne local
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.tz_convert('Australia/Melbourne').dt.strftime('%d/%m/%Y')
    # Keep only columns relevant for sentiment: drop null, drop duplicates, keep date and summary
    df = df[['date', 'summary']].dropna().drop_duplicates()
    return df
def fin_bert_sentiment_analysis():
    model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_URL,
    num_labels=3,
    id2label={0: "neutral", 1: "positive", 2: "negative"},
    label2id={"neutral": 0, "positive": 1, "negative": 2},
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_URL
    )

    # Construct a Huggingface pipeline!
    sentiment_classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=1, 
        top_k=None, padding=True, truncation=True, max_length=256
    )
    with torch.amp.autocast("cuda"):
        outputs = sentiment_classifier(scrape_news("AAPL", START_DATE, END_DATE)['summary'].tolist()[7])
    print(scrape_news(TICKER, START_DATE, END_DATE)['summary'].tolist()[2])
    print(outputs)
    print("This comment is rated %s with confidence %.3f" % (outputs[0][0]["label"], outputs[0][0]["score"]))
    #TO DO: predict sentiments from scraped news data
    log_positive_score = []
    log_neutral_score = []
    log_negative_score = []
    log_datetime = []
    
    df = scrape_news(TICKER, START_DATE, END_DATE)
    texts = df["summary"].tolist()   # list[str]
    dates = df["date"].tolist() 
    
    with torch.amp.autocast("cuda"):
        outputs = sentiment_classifier(texts, batch_size=len(texts))

    for (output, td) in zip(outputs, dates):
        for label in output:
            if label["label"] == 'positive':
                log_positive_score.append(label["score"])
            elif label["label"] == 'neutral':
                log_neutral_score.append(label["score"])
            elif label["label"] == 'negative':
                log_negative_score.append(label["score"])
        log_datetime.append(td)
    df = pd.DataFrame({
    'Datetime': log_datetime,
    'positive': log_positive_score,
    'neutral': log_neutral_score,
    'negative': log_negative_score,
    })       
    df['Datetime'] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y", dayfirst=True)
    # Calculate a 'score' using the positive and negative predictions
    df['score'] = (df['positive'] - df['negative'])/(df['positive'] + df['negative'])

    # Calculate median sentiment for everyday
    df_daily = (
        df.groupby('Datetime', as_index=True)
        .median(numeric_only=True)
        .sort_index()
    )
    df_daily.fillna(0, inplace=True)
    # Count the number of entries per day
    df_daily_count = df.groupby('Datetime').size()

    # If you want to add this count to your existing df_daily DataFrame:
    df_daily['entry_count'] = df_daily_count
    df_daily['entry_count_ma'] = df_daily['entry_count'].rolling(window=28, center=False).mean()

    # Calculate a 28 day moving average of the score
    df_daily['score_ma'] = df_daily['score'].rolling(window=28, center=False).mean()
    # Sort all rows by the datetimeindex
    print(df_daily)
    stock_data = yf.download(TICKER, START_DATE, END_DATE, auto_adjust=True)
    visualize_sentiment(df_daily, stock_data)
if __name__ == "__main__":
    fin_bert_sentiment_analysis()
    
    

