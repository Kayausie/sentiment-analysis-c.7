import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
def visualize_sentiment(df_daily, stock_data):
    """Visualize sentiment scores and stock closing prices over time.
    
    Args:
        df_daily (pd.DataFrame): DataFrame with daily sentiment scores.
        stock_data (pd.DataFrame): DataFrame with stock closing prices.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot the sentiment score on the first y-axis
    ax1.plot(df_daily.index, 
            df_daily['score'], color='blue', label='Sentiment Score')

    # ax1.plot(df_daily.index, 
    #         df_daily['score_ma'], color='red', label='Sentiment Score 28d MA')
    ax1.set_xlabel('Date (m/y)', fontsize=16)
    ax1.set_ylabel('Sentiment', color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue')
    # Increase the number of x-axis ticks
    ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
    ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y'))
    # Create the second y-axis
    ax2 = ax1.twinx()

    # Plot negative sentiment on the second y-axis
    ax2.plot(stock_data.index, stock_data['Close'], 
            color='red', label='Close Price')

    ax2.set_ylabel('Price ($AUD)', color='red', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left', fontsize=12)
    _ = plt.title('Sentiment Scores ' + "Apple", fontsize=16)
    plt.show()