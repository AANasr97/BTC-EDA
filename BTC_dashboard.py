
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from datetime import datetime
import glob
import os
from PIL import Image
import mplfinance as mpf
import requests
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from mplfinance.original_flavor import candlestick_ohlc
# Set the theme to 'seaborn' for a visually appealing style
sns.set()
# Format numbers to display normally
pd.options.display.float_format = '{:,.4f}'.format

st.set_page_config(page_title="Bitcoin!!!", page_icon=":chart_with_upwards_trend: :bar_chart: :chart_with_downwards_trend:",layout="wide")





# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Introduction", "Explore Data", "Trading Strategies", "Conclusion"])

# Streamlit sidebar
st.sidebar.title("Crypto Tools")

# Fetch live BTC-USD price using an API
response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
btc_price = response.json()["bitcoin"]["usd"]

# Display live BTC-USD price in the sidebar
st.sidebar.write(f"Live BTC-USD Price: ${btc_price:.2f}")

# Calculator in the sidebar
st.sidebar.subheader("Crypto Calculator")

amount = st.sidebar.number_input("Enter Amount (BTC):", min_value=0.0)
converted_amount = amount * btc_price

st.sidebar.write(f"Converted Amount (USD): ${converted_amount:.2f}")


# Introduction page
def introduction():
    # App title and introduction
    st.title(":chart_with_upwards_trend: :bar_chart: :chart_with_downwards_trend: Bitcoin EDA and Trading Analysis")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
    st.write("Welcome to the Bitcoin Exploratory Data Analysis (EDA) and Trading Analysis dashboard.")
    

    # Load the images
    image1 = Image.open("BTC.jpeg")
    image2 = Image.open("BTC2.jpeg")

    # Define the column layout
    col1, col2 = st.columns(2)

    # Display the images with adjusted size
    with col1:
        st.image(image1, use_column_width=True, caption="BTC")

    with col2:
        st.image(image2, use_column_width=True, caption="BTC")
    
    
    # Objective
    st.header("Objective")
    st.write("Our objective is to provide you with valuable insights into Bitcoin's historical price data, explore various trading strategies, and offer an informative analysis of cryptocurrency trends.")
    
    # Key Features
    st.header("Key Features")
    st.write("✅ Interactive visualizations to explore Bitcoin data.")
    st.write("✅ Analyze Bitcoin's performance over different time intervals.")
    st.write("✅ Discover and test trading strategies based on historical data.")

    # Audience
    st.header("Audience")
    st.write("This dashboard is designed for cryptocurrency enthusiasts, traders, and anyone interested in analyzing Bitcoin data.")

    # Navigation
    st.header("Navigation")
    st.write("Use the navigation menu on the left to explore different sections of the dashboard. You can start with data exploration, delve into trading strategies, and find additional information about Bitcoin.")
    
    st.write("---")  # Add a separator between the introduction and the rest of the content

    # Introduction to Bitcoin
    st.header("What is Bitcoin?")
    st.write(
        "Bitcoin (BTC) is a decentralized digital currency that enables peer-to-peer transactions without the need for intermediaries, "
        "such as banks. It was created in 2009 by an anonymous entity known as Satoshi Nakamoto. Bitcoin operates on a technology called "
        "blockchain, which is a distributed ledger that records all transactions across a network of computers."
    )

    # Purpose of the Dashboard
    st.header("Explore Bitcoin Data")
    st.write(
        "This dashboard allows you to explore Bitcoin price data, perform data analysis, and visualize trends over time. "
        "You can select different time intervals, view charts, analyze trading volumes, and more. Whether you're a novice or an "
        "experienced trader, this tool provides valuable insights into Bitcoin's historical performance."
    )

    # Getting Started
    st.header("Getting Started")
    st.write(
        "1. Use the date range selectors to choose the analysis period you're interested in."
        "\n2. Select the chart type from the options provided."
        "\n3. Apply data filtering criteria if needed."
        "\n4. Explore and analyze Bitcoin data to make informed decisions."
    )

    # About the Author
    st.header("About the Author")
    st.write(
        "This dashboard was created by `Eng.Ahmed NasrElDin` to provide a user-friendly interface for exploring Bitcoin data. "
        "For questions or inquiries, please contact ahmad.nasr13197@gmail.com ."
    )

    # Data Sources and Disclaimer
    st.header("Data Sources and Disclaimer")
    st.write(
        "The Bitcoin price data used in this dashboard is obtained from [KAGGLE](https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution). While this tool provides insights and analysis, "
        "it is important to note that cryptocurrency markets are highly volatile, and any trading or investment decisions should "
        "be made with caution. Always conduct your research and consult financial experts if needed."
    )

def explore_data():
    
    
    # Load your original BTC data and resampled DataFrames here
    BTC = pd.read_csv('btcusd.csv')
    st.write("---")
    # Create an expander to show expected columns and file uploader
    with st.expander("Upload and Check File Or ignore to explore BTC data"):
        st.write("The Bitcoin price data used in this dashboard is obtained from [KAGGLE](https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution).")
        expected_columns = dict(zip(BTC.columns, BTC.dtypes))

        # Create a file uploader
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "xls"])
        one_row_sample = BTC.sample(1).iloc[0]
        # Provide a description to the user
        st.write("Please upload a file with the following columns and their respective data types:")

        # Create a list of dictionaries for column names and data types
        column_info = [{'Column Name': column, 'Data Type': data_type} for column, data_type in expected_columns.items()]

        # Convert the list of dictionaries into a DataFrame
        column_info_df = pd.DataFrame(column_info)

        # Display the DataFrame
        st.dataframe(column_info_df.T)
        st.write("One Row Sample")
        st.dataframe(pd.DataFrame(one_row_sample).T)

        # Check if a file is uploaded
        if uploaded_file is not None:
            # Read the uploaded file into a DataFrame
            uploaded_df = None
            file_extension = uploaded_file.name.split(".")[-1]
            
            if file_extension == "csv":
                uploaded_df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                uploaded_df = pd.read_excel(uploaded_file)
            
            # Check if the columns of the uploaded DataFrame match the original DataFrame
            if set(uploaded_df.columns) == set(BTC.columns):
                st.write("Columns match. You can proceed with the uploaded data.")
                BTC = uploaded_df  # Use the uploaded DataFrame
            else:
                st.write("Error: The columns of the uploaded file do not match the original DataFrame.")
        else:
            st.write("Upload a file to get started exploring and analyzing its data.")

    # App title and introduction
    st.title(":chart_with_upwards_trend: :bar_chart: :chart_with_downwards_trend: Bitcoin EDA and Trading Analysis")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
    st.write("Welcome to the Bitcoin Exploratory Data Analysis (EDA) and Trading Analysis dashboard.")

    # Convert milliseconds to seconds by dividing by 1000 and apply to the 'time' column
    BTC['time'] = BTC['time'] / 1000

    # Convert to a human-readable date and time format
    BTC['time'] = BTC['time'].apply(lambda x: datetime.utcfromtimestamp(x))

    # Set the 'time' column as the index of the DataFrame
    BTC.set_index('time', inplace=True)

    #
    BTC_1min = BTC
    # Calculate the percentage change for the 'close' column
    BTC_1min['Return'] = BTC_1min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_1min = BTC_1min.dropna()
    BTC_1min['Total Traded'] = BTC_1min['close'] * BTC_1min['volume']


    # Resample to 15-minute intervals and apply the aggregations
    BTC_15min = BTC.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_15min['Return'] = BTC_15min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_15min = BTC_15min.dropna()
    BTC_15min['Total Traded'] = BTC_15min['close'] * BTC_15min['volume']


    # Resample to 30-minute intervals and apply the aggregations
    BTC_30min = BTC.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_30min['Return'] = BTC_30min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_30min = BTC_30min.dropna()
    BTC_30min['Total Traded'] = BTC_30min['close'] * BTC_30min['volume']


    # Resample the data to 1-hour intervals and apply the aggregations
    BTC_1hr = BTC.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_1hr['Return'] = BTC_1hr['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_1hr = BTC_1hr.dropna()
    BTC_1hr['Total Traded'] = BTC_1hr['close'] * BTC_1hr['volume']


    # Resample the data to 4-hour intervals and apply the aggregations
    BTC_4hr = BTC.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_4hr['Return'] = BTC_4hr['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_4hr = BTC_4hr.dropna()
    BTC_4hr['Total Traded'] = BTC_4hr['close'] * BTC_4hr['volume']


    # Resample to daily intervals and apply the aggregations
    BTC_daily = BTC.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_daily['Return'] = BTC_daily['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_daily = BTC_daily.dropna()
    BTC_daily['Total Traded'] = BTC_daily['close'] * BTC_daily['volume']


    BTC_weekly = BTC.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_weekly['Return'] = BTC_weekly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_weekly = BTC_weekly.dropna()
    BTC_weekly['Total Traded'] = BTC_weekly['close'] * BTC_weekly['volume']


    # Resample to monthly intervals and apply the aggregations
    BTC_monthly = BTC.resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_monthly['Return'] = BTC_monthly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_monthly = BTC_monthly.dropna()
    BTC_monthly['Total Traded'] = BTC_monthly['close'] * BTC_monthly['volume']


    # Resample to quarterly intervals and apply the aggregations
    BTC_quarterly = BTC.resample('Q').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_quarterly['Return'] = BTC_quarterly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_quarterly = BTC_quarterly.dropna()
    BTC_quarterly['Total Traded'] = BTC_quarterly['close'] * BTC_quarterly['volume']


    # Create a dictionary to store all resampled dataframes
    resampled_dataframes = {
        '1 Minute' : BTC_1min,
        '15 Minutes': BTC_15min,
        '30 Minutes': BTC_30min,
        '1 Hour': BTC_1hr,
        '4 Hours': BTC_4hr,
        'Daily': BTC_daily,
        'Weekly': BTC_weekly,
        'Monthly': BTC_monthly,
        'Quarterly': BTC_quarterly
    }
    # Add content for exploring Bitcoin data here
    st.header("Explore Bitcoin Data")
    st.write("---")
    # Add explanations for the data columns
    st.markdown("### Data Columns Explanation")
    st.write("Let's explore the meaning of the data columns in the Bitcoin dataset:")
    st.write("1. **Open Price:** The price of Bitcoin at the beginning of the time interval.")
    st.write("2. **Close Price:** The final price of Bitcoin at the end of the time interval.")
    st.write("3. **High Price:** The highest price reached by Bitcoin during the time interval.")
    st.write("4. **Low Price:** The lowest price reached by Bitcoin during the time interval.")
    st.write("5. **Volume:** The total number of Bitcoin units traded during the time interval.")
    st.write("6. **Total Traded:** The total value of Bitcoin traded during the time interval.")
    st.write("7. ****Percentage Change:** The percentage change measures how much the closing price has moved relative to its previous value.**")
    st.write("---")
    # Define the LaTeX equation with a larger font size
    latex_equation = r'<span style="font-size: 30px;">$ r_t = \frac{p_t}{p_{t-1}} -1$</span>'
    # Render the LaTeX equation
    st.markdown(latex_equation, unsafe_allow_html=True)
    st.write("---")
    st.sidebar.header('User Input Features')

    # Add a sidebar to select the time interval
    interval = st.sidebar.selectbox('Select Time Interval:', list(resampled_dataframes.keys()))

    # Use the selected interval to access the corresponding dataframe
    selected_dataframe = resampled_dataframes[interval]

    # Add a date range filter
    date_start = st.sidebar.date_input("Start Date", min(selected_dataframe.index))
    date_end = st.sidebar.date_input("End Date", max(selected_dataframe.index))

    # Convert date_start and date_end to datetime objects
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)

    # Add a checkbox to enable data statistics display
    show_df = st.sidebar.checkbox("Show DataFrame")

    # Add a checkbox to enable data statistics display
    show_stats = st.sidebar.checkbox("Show Data Statistics")
    
    show_chart = st.sidebar.checkbox("Show Price Chart")
    
    
    show_normaldist = st.sidebar.checkbox("Show Normal Distribution")
    show_returns = st.sidebar.checkbox("Show Returns Distribution")
    
    # Filter the DataFrame based on the selected date range
    EDA_data = selected_dataframe[(selected_dataframe.index >= date_start) & (selected_dataframe.index <= date_end)]

    # Display summary statistics if the checkbox is selected
    if show_df:
        col1, col2, col3 = st.columns(3)
        
        st.subheader("DataFrame")
        st.write("---")
        # Create Streamlit columns to display DataFrame heads
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"Head {interval} Close Price")
            st.dataframe(EDA_data.head(2))

        with col2:
            st.subheader(f"Tail {interval} Close Price")
            st.dataframe(EDA_data.tail(2))

        with col3:
            st.subheader(f"Sample {interval} Close Price")
            st.dataframe(EDA_data.sample(2))

    # Display summary statistics if the checkbox is selected
    if show_stats:
        st.write("---")
        st.subheader("Data Statistics")
        st.write("---")
        st.dataframe(EDA_data.describe(), width=1000)
    # Create a Plotly figure
    fig = go.Figure()

    def update_chart(fig, data):
            fig.data = []
            for column in plot_data:
                fig.add_trace(go.Scatter(x=EDA_data.index, y=EDA_data[column], mode='lines', name=f"BTC {column}"))


                # Add a line trace for BTC close price
                # fig.add_trace(go.Scatter(x=EDA_data.index, y=EDA_data['close'], mode='lines', name=f"BTC {interval} Close Price"))
                
                # Customize the figure layout
                fig.update_layout(
                    title='BTC Price Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    showlegend=True,
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label='1m', step='month', stepmode='backward'),
                                dict(count=6, label='6m', step='month', stepmode='backward'),
                                dict(count=1, label='YTD', step='year', stepmode='todate'),
                                dict(count=1, label='1y', step='year', stepmode='backward'),
                                dict(step='all')
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type='date'
                    ),
                    # Adjust height and width for the figure
                    height=600,  # Adjust the height as needed
                    width=1000   # Adjust the width as needed
                )

    if show_chart:
        st.sidebar.subheader('Line chart parameters')
        plot_data = st.sidebar.multiselect('Select data', ['close', 'high', 'low', 'open', 'volume', 'Return'], ['close'])
        
        # Code to show the graph
        # Call the update_chart function to update the chart
        update_chart(fig, plot_data)
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

    if show_normaldist:

        # Streamlit app title and description
        st.title('Normal Distribution with Standard Deviation Bands and Probability Area')
        st.write('This app visualizes a normal distribution with standard deviation bands and a specified probability area.')
        st.write("---")
        # User input for mean and standard deviation
        mu = st.slider('Mean (μ)', -10.0, 10.0, 0.0)
        sigma = st.slider('Standard Deviation (σ)', 0.1, 10.0, 1.0)

        # Generate data for a normal distribution
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
        pdf = stats.norm.pdf(x, mu, sigma)

        # Create a Plotly figure
        fig = go.Figure()

        # Plot the normal distribution curve
        fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Normal Distribution', line=dict(color='blue')))

        # Fill standard deviation bands
        probabilities = []
        for std in range(1, 4):
            y_band = stats.norm.pdf(x, mu, std * sigma)
            fig.add_trace(go.Scatter(x=x, y=y_band, fill='tozeroy', mode='none', 
                                    fillcolor='orange', name=f'{std} STD'))
            
            # Calculate the probability within each band
            prob = stats.norm.cdf(mu + std * sigma, mu, sigma) - stats.norm.cdf(mu - std * sigma, mu, sigma)
            probabilities.append(prob)
            prob_text = f'Probability within ±{std}σ: {prob:.2%}'
            fig.add_annotation(
                x=mu + std * sigma,
                y=max(y_band),
                text=prob_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor='blue',
                arrowwidth=2,
                textangle=0,
                xshift=10,
                yshift=10
            )

        # Calculate and shade the probability area
        x_fill = np.linspace(mu - 2*sigma, mu + 2*sigma, 1000)
        y_fill = stats.norm.pdf(x_fill, mu, sigma)
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='none', 
                                fillcolor='green', name='Probability Area'))

        # Customize the layout
        fig.update_layout(
            xaxis_title='X-axis',
            yaxis_title='Probability Density',
            title='Normal Distribution with Standard Deviation Bands and Probability Area',
            legend=dict(x=0.75, y=0.95),
            height=500,  # Adjust the height as needed
            width=800   # Adjust the width as needed
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    if show_returns:
        
        # Streamlit app title and description
        st.subheader('Interactive Histogram with KDE Overlay')
        
        st.write("---")
        # User input for mean and standard deviation
        mean = st.slider('Mean (μ)', min_value=EDA_data['Return'].min(), max_value=EDA_data['Return'].max(), value=EDA_data['Return'].mean())
        std_dev = st.slider('Standard Deviation (σ)', min_value=0.01, max_value=EDA_data['Return'].std(), value=EDA_data['Return'].std())

        # Create a range of x values
        x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)

        # Calculate the probability density function (PDF) values
        pdf_values = stats.norm.pdf(x, loc=mean, scale=std_dev)

        # Create a DataFrame with x and PDF values
        pdf_df = pd.DataFrame({'x': x, 'PDF': pdf_values})

        # Create a Plotly figure for the histogram with KDE overlay
        fig = px.histogram(EDA_data, x='Return', nbins=100, marginal='box', opacity=0.7,
                        labels={'Return': 'Returns'}, title='Histogram with KDE Overlay for Returns')

        # Add the KDE overlay curve
        fig.add_trace(go.Scatter(x=pdf_df['x'], y=pdf_df['PDF'], mode='lines', name='KDE Overlay'))

        # Customize the figure layout
        fig.update_layout(
            xaxis_title='Returns',
            yaxis_title='Frequency',
            showlegend=True,
            height=600,  # Adjust the height as needed
            width=1000,  # Adjust the width as needed
        )

        # Display the Plotly figure
        st.plotly_chart(fig, use_container_width=True)
        
################################################################################################################################################################
################################################################################################################################################################
        
        # Create a Streamlit app title and description
        st.subheader('Histogram with KDE Overlay and Box Plot')
        st.write("---")

        # Load your EDA_data data (replace this with your data loading code)
        # Example: EDA_data = pd.read_csv('your_data.csv')

        # User input for selecting the column
        selected_column = st.selectbox('Select a column:', EDA_data.columns)

        # Create a Plotly figure for both the histogram and box plot
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=(f'Histogram with KDE Overlay for {selected_column}', f'Box Plot for {selected_column}'))

        # Add the histogram
        show_histogram = st.checkbox('Show Histogram', value=True)
        if show_histogram:
            histogram_trace = go.Histogram(x=EDA_data[selected_column], nbinsx=100, name='Histogram')
            fig.add_trace(histogram_trace, row=1, col=1)

        # Calculate KDE
        kde_x = np.linspace(EDA_data[selected_column].min(), EDA_data[selected_column].max(), 1000)
        kde_y = stats.gaussian_kde(EDA_data[selected_column])(kde_x)

        # Add the KDE overlay curve
        show_kde = st.checkbox('Show KDE Overlay', value=True)
        if show_kde:
            kde_trace = go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE Overlay', line=dict(color='blue'))
            fig.add_trace(kde_trace, row=1, col=1)

        # Create a box plot
        show_box_plot = st.checkbox('Show Box Plot', value=True)
        if show_box_plot:
            box_plot_trace = go.Box(y=EDA_data[selected_column], name='Box Plot')
            fig.add_trace(box_plot_trace, row=1, col=2)

        # Customize the layout
        fig.update_layout(
            xaxis_title=f'{selected_column} Value',
            yaxis_title='Frequency',
            height=500,  # Adjust the height as needed
            width=1000,  # Adjust the width as needed
            showlegend=False  # Hide legend to avoid duplicate legend entries
        )

        # Display the combined plot using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
################################################################################################################################################################
################################################################################################################################################################
        
        mean = EDA_data[selected_column].mean()
        var = EDA_data[selected_column].var()
        std = np.sqrt(var)


        # Create a DataFrame to display the statistics
        statistics_df = pd.DataFrame({
            'Statistic': [f'{interval} Mean {selected_column}', f'{interval} Variance of {selected_column}', f'{interval} Standard Deviation of {selected_column}'],
            'Value': [mean, var, std]
            })

        # Set the 'Statistic' column as the index (optional)
        statistics_df.set_index('Statistic', inplace=True)

        # Display the DataFrame
        st.dataframe(statistics_df, width=1000)

    
        st.write("---")
        st.subheader("Explore Maximum Trading Data Values")

        # Find and display the date with the maximum trading Value
        max_date = EDA_data[selected_column].idxmax()
        st.write(f"The date with the maximum {selected_column} is {max_date}.")
        st.write("---")
        # Display the entire row corresponding to the maximum Value
        max_row = EDA_data.loc[max_date]
        st.write(f"Details of the maximum {selected_column}:")
        st.dataframe(pd.DataFrame(max_row).T)
        st.write("---")
        # Display the top 10 records with the highest trading Value
        top_10_data = EDA_data.nlargest(10, selected_column)
        st.write(f"Top 10 records with the highest {selected_column}:")
        st.dataframe(top_10_data)
        
        
        top_10_data = top_10_data.sort_index()
        # Bar Chart
        st.subheader(f"Top 10 Dates with Highest {selected_column} (Bar Chart)")
        bar_fig = px.bar(top_10_data, x=top_10_data.index, y=selected_column, title=f'Top 10 Dates with Highest {selected_column}')
        st.plotly_chart(bar_fig)

        # Line Chart
        st.subheader(f"Top 10 Dates with Highest {selected_column} (Line Chart)")
        line_fig = px.line(top_10_data, x=top_10_data.index, y=selected_column, markers=True, title=f'Top 10 Dates with Highest {selected_column}')
        st.plotly_chart(line_fig)

        # Interactive Plot
        st.subheader(f"Top 10 Dates with Highest {selected_column} (Interactive Plot)")
        fig = px.scatter(top_10_data, x=top_10_data.index, y=selected_column, text=top_10_data.index, title=f'Top 10 Dates with Highest {selected_column}')
        fig.update_traces(textposition='top center')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text=selected_column)
        st.plotly_chart(fig)



    ETS = st.sidebar.checkbox('ETS Decomposition')
    
    if ETS:
        # Add a sidebar to allow user input
        st.sidebar.header('Seasonal Decomposition Options')

        # User input for decomposition period
        period = st.sidebar.number_input('Enter Decomposition Period:', min_value=1, step=1, value=365 * 4)

        # User input for decomposition mode
        decomposition_mode = st.sidebar.selectbox('Choose Decomposition Mode:', ['additive', 'multiplicative'])

        # Perform seasonal decomposition based on user input
        decomposition = sm.tsa.seasonal_decompose(EDA_data['close'], period=period, model=decomposition_mode)

        # Create a Plotly figure for seasonal decomposition
        fig = go.Figure()

        # Add original 'close' data
        fig.add_trace(go.Scatter(x=EDA_data.index, y=EDA_data['close'], mode='lines', name='Original Close'))

        # Add seasonal component
        fig.add_trace(go.Scatter(x=EDA_data.index, y=decomposition.seasonal, mode='lines', name='Seasonal Component'))

        # Add trend component
        fig.add_trace(go.Scatter(x=EDA_data.index, y=decomposition.trend, mode='lines', name='Trend Component'))

        # Add residual component
        fig.add_trace(go.Scatter(x=EDA_data.index, y=decomposition.resid, mode='lines', name='Residual Component'))

        # Customize the figure layout
        fig.update_layout(
            title=f'Seasonal Decomposition of BTC Close Price (Mode: {decomposition_mode.capitalize()})',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label='1y', step='year', stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )

        # Customize the figure size and aspect ratio
        fig.update_layout(
            autosize=True,
            width=1000,
            height=800  # Adjust the height as needed
        )

        # Show the Plotly figure in Streamlit
        st.plotly_chart(fig)
################################################################################################################################################################

    ADF = st.sidebar.checkbox('Test for Stationarity')
    if ADF:
            
        # Add a section for the ADF test
        st.header('Augmented Dickey-Fuller Test')
        st.write('The Augmented Dickey-Fuller (ADF) test is used to determine whether a time series is stationary or not.')

        # Function to perform ADF test and explain results
        def adf_check(time_series):
            result = adfuller(time_series)
            st.write('ADF Test Statistic:', round(result[0], 2))
            st.write('p-value:', round(result[1], 2))
            st.write('#Lags Used:', result[2])
            st.write('Number of Observations Used:', result[3])
            
            if result[1] <= 0.05:
                st.write("Strong evidence against the null hypothesis. Reject the null hypothesis. Data has no unit root and is stationary.")
            else:
                st.write("Weak evidence against the null hypothesis. Time series has a unit root, indicating it is non-stationary.")

        # Perform ADF test on the 'close' data
        st.subheader('ADF Test Results for Close Price')
        adf_check(EDA_data['close'])
        st.subheader('**We have now realized that our data is seasonal (it is also pretty obvious from the plot itself). This means we need to use Seasonal ARIMA on our model.**')

################################################################################################################################################################
################################################################################################################################################################


        # Create synthetic non-stationary and stationary time series data
        np.random.seed(42)
        index = pd.date_range(start='2020-01-01', periods=365, freq='D')
        non_stationary_data = pd.Series(np.cumsum(np.random.randn(365)), index=index, name='Non-Stationary Data')
        stationary_data = pd.Series(np.random.randn(365), index=index, name='Stationary Data')

        # Streamlit UI
        st.subheader("Non-Stationary vs. Stationary Time Series")

        # Select the time series to analyze
        selected_data = st.selectbox("Select Time Series Data", ["Non-Stationary", "Stationary"])

        if selected_data == "Non-Stationary":
            data = non_stationary_data
            st.header("Non-Stationary Time Series")
            st.write("This is an example of a non-stationary time series.")
        else:
            data = stationary_data
            st.header("Stationary Time Series")
            st.write("This is an example of a stationary time series.")

        # Plot the selected time series
        st.subheader("Time Series Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Time Series'))
        fig.update_layout(
            title=f"{selected_data} Time Series Plot",
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label='1y', step='year', stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        st.plotly_chart(fig)

        # Perform ADF test and display results
        st.subheader("Augmented Dickey-Fuller Test")
        result = adfuller(data)
        st.write('ADF Test Statistic:', round(result[0], 2))
        st.write('p-value:', round(result[1], 2))
        st.write('#Lags Used:', result[2])
        st.write('Number of Observations Used:', result[3])

        if result[1] <= 0.05:
            st.write("Strong evidence against the null hypothesis. Reject the null hypothesis. Data has no unit root and is stationary.")
        else:
            st.write("Weak evidence against the null hypothesis. Time series has a unit root, indicating it is non-stationary.")

################################################################################################################################################################


    download = st.sidebar.checkbox('Export Data')
    if download:
        
        st.subheader('Data Export Example')
        st.write('---')

        # User input for selecting the column
        selected_columns = st.multiselect('Select columns:', EDA_data.columns)
        
        # Check if at least one column is selected
        if selected_columns:
            # Filter the data based on the selected columns
            filtered_data = EDA_data[selected_columns]

            # Display the filtered data in a DataFrame
            st.write(filtered_data)

            # Create a button to trigger data export
            if st.button('Export Data'):
                # Generate a CSV file for the filtered data with the index
                csv_file = filtered_data.to_csv(index=True)

                # Offer the file for download
                st.download_button(
                    label='Download CSV',
                    data=csv_file.encode(),
                    file_name='filtered_data.csv',
                    mime='text/csv',
                )
        else:
            st.write('Please select at least one column.')

################################################################################################################################################################

################################################################################################################################################################

def trading_strategies(): 
################################################################################################################################################################
    BTC = pd.read_csv('btcusd.csv')
    # Convert milliseconds to seconds by dividing by 1000 and apply to the 'time' column
    BTC['time'] = BTC['time'] / 1000
    # Convert to a human-readable date and time format
    BTC['time'] = BTC['time'].apply(lambda x: datetime.utcfromtimestamp(x))
    # Set the 'time' column as the index of the DataFrame
    BTC.set_index('time', inplace=True)

    #
    BTC_1min = BTC
    # Calculate the percentage change for the 'close' column
    BTC_1min['Return'] = BTC_1min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_1min = BTC_1min.dropna()
    BTC_1min['Total Traded'] = BTC_1min['close'] * BTC_1min['volume']


    # Resample to 15-minute intervals and apply the aggregations
    BTC_15min = BTC.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_15min['Return'] = BTC_15min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_15min = BTC_15min.dropna()
    BTC_15min['Total Traded'] = BTC_15min['close'] * BTC_15min['volume']


    # Resample to 30-minute intervals and apply the aggregations
    BTC_30min = BTC.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_30min['Return'] = BTC_30min['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_30min = BTC_30min.dropna()
    BTC_30min['Total Traded'] = BTC_30min['close'] * BTC_30min['volume']


    # Resample the data to 1-hour intervals and apply the aggregations
    BTC_1hr = BTC.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_1hr['Return'] = BTC_1hr['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_1hr = BTC_1hr.dropna()
    BTC_1hr['Total Traded'] = BTC_1hr['close'] * BTC_1hr['volume']


    # Resample the data to 4-hour intervals and apply the aggregations
    BTC_4hr = BTC.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_4hr['Return'] = BTC_4hr['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_4hr = BTC_4hr.dropna()
    BTC_4hr['Total Traded'] = BTC_4hr['close'] * BTC_4hr['volume']


    # Resample to daily intervals and apply the aggregations
    BTC_daily = BTC.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_daily['Return'] = BTC_daily['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_daily = BTC_daily.dropna()
    BTC_daily['Total Traded'] = BTC_daily['close'] * BTC_daily['volume']


    BTC_weekly = BTC.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_weekly['Return'] = BTC_weekly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_weekly = BTC_weekly.dropna()
    BTC_weekly['Total Traded'] = BTC_weekly['close'] * BTC_weekly['volume']


    # Resample to monthly intervals and apply the aggregations
    BTC_monthly = BTC.resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_monthly['Return'] = BTC_monthly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_monthly = BTC_monthly.dropna()
    BTC_monthly['Total Traded'] = BTC_monthly['close'] * BTC_monthly['volume']


    # Resample to quarterly intervals and apply the aggregations
    BTC_quarterly = BTC.resample('Q').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Calculate the percentage change for the 'close' column
    BTC_quarterly['Return'] = BTC_quarterly['close'].pct_change()
    # Drop the NaN value in the 'Return' column (first row)
    BTC_quarterly = BTC_quarterly.dropna()
    BTC_quarterly['Total Traded'] = BTC_quarterly['close'] * BTC_quarterly['volume']


    # Create a dictionary to store all resampled dataframes
    resampled_dataframes = {
        '1 Minute' : BTC_1min,
        '15 Minutes': BTC_15min,
        '30 Minutes': BTC_30min,
        '1 Hour': BTC_1hr,
        '4 Hours': BTC_4hr,
        'Daily': BTC_daily,
        'Weekly': BTC_weekly,
        'Monthly': BTC_monthly,
        'Quarterly': BTC_quarterly
    }
    # Add a sidebar to select the time interval
    interval = st.sidebar.selectbox('Select Time Interval:', list(resampled_dataframes.keys()))

    # Use the selected interval to access the corresponding dataframe
    selected_dataframe = resampled_dataframes[interval]

    # Add a date range filter
    date_start = st.sidebar.date_input("Start Date", min(selected_dataframe.index))
    date_end = st.sidebar.date_input("End Date", max(selected_dataframe.index))

    # Convert date_start and date_end to datetime objects
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)

    # Filter the DataFrame based on the selected date range
    EDA_data = selected_dataframe[(selected_dataframe.index >= date_start) & (selected_dataframe.index <= date_end)]
    
################################################################################################################################################################
    # Streamlit app
    st.title('Trading Strategies Explorer')
    st.subheader('Select a Trading Strategy')

    strategy = st.selectbox('Select Strategy:', ['Simple Momentum','Simple Contrarian', 'Threshold_based', 'SMA1', 'SMA2', 'EWMA', 'Perfect'])
    
    def summary_ann(returns):
        summary = returns.agg(["mean", "std"]).T
        summary["Return"] = summary["mean"] * 365
        summary["Risk"] = summary["std"] * np.sqrt(365)
        summary.drop(columns = ["mean", "std"], inplace = True)
        return summary

    if strategy == 'Simple Momentum':
        st.write('You selected the Simple Momentum strategy.')
        def simple_momentum_strategy(df):
            # Calculate trading positions
            df['Positions'] = np.sign(df['Return'])
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='Simple Momentum Strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = simple_momentum_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))



    if strategy == 'Simple Contrarian':
        st.write('You selected the Simple Contrarian strategy.')
        def simple_contrarian_strategy(df):
            # Calculate trading positions
            df['Positions'] = -np.sign(df['Return'])
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='Simple Contrarian Strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = simple_contrarian_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))


    if strategy == 'Threshold_based':
        st.write('You selected the Threshold-based strategy.')
        def Threshold_based_strategy(df):
            # Calculate trading positions
            df['Positions'] = np.where(df['Return'] > 0.3, -1, 1)
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='Threshold_based Strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = Threshold_based_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))


    if strategy == 'SMA1':
        st.write('You selected the SMA (Momentum) Crossover strategy.')
        st.markdown('__a) Investing (+1)__: SMA50 __>__ SMA200 \n __b) Short Selling (-1)__: SMA50 __<__ SMA200')
        def SMA1_strategy(df):
            df["SMA50"] = df.close.rolling(window = 50).mean()
            df["SMA200"] = df.close.rolling(window = 200).mean()
            
            # Create a Plotly figure
            fig = go.Figure()

            # Add SMA50
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))

            # Add SMA200
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))

            # Customize the figure layout
            fig.update_layout(
                title='SMA50 vs SMA200',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )

            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1000,
                height=800  # Adjust the height as needed
            )

            # Show the Plotly figure in Streamlit
            st.plotly_chart(fig)
            
            # Calculate trading positions
            df['Positions'] = np.sign(df.SMA50.sub(df.SMA200)) 
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='SMA (Momentum) Crossover Strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = SMA1_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))


    if strategy == 'SMA2':
        st.write('You selected the SMA (Contrarian) Crossover strategy.')
        st.markdown('__a) Investing (+1)__: SMA50 __<__ SMA200 \n __b) Short Selling (-1)__: SMA50 __>__ SMA200')
        def SMA2_strategy(df):
            df["SMA50"] = df.close.rolling(window = 50).mean()
            df["SMA200"] = df.close.rolling(window = 200).mean()
            
            # Create a Plotly figure
            fig = go.Figure()

            # Add SMA50
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))

            # Add SMA200
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))

            # Customize the figure layout
            fig.update_layout(
                title='SMA50 vs SMA200',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )

            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1000,
                height=800  # Adjust the height as needed
            )

            # Show the Plotly figure in Streamlit
            st.plotly_chart(fig)
            
            # Calculate trading positions
            df['Positions'] = -np.sign(df.SMA50.sub(df.SMA200)) 
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='SMA (Contrarian) Crossover Strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = SMA2_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))

    if strategy == 'EWMA':
        st.write('You selected the EWMA Exponentially-weighted moving average strategy.')
        st.markdown("__a) Investing (+1)__: When the asset's price is above its 12-period EWMA. __b) Short Selling (-1)__: When the asset's price is below or equal to its 12-period EWMA.")
        def EWMA_strategy(df):
            df['EWMA12'] = df['close'].ewm(span=12).mean()
            
            # Create a Plotly figure
            fig = go.Figure()

            # Add SMA50
            fig.add_trace(go.Scatter(x=df.index, y=df['EWMA12'], mode='lines', name='EWMA12'))

            # Add SMA200
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='close'))

            # Customize the figure layout
            fig.update_layout(
                title='EWMA12',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )

            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1000,
                height=800  # Adjust the height as needed
            )

            # Show the Plotly figure in Streamlit
            st.plotly_chart(fig)
            
            # Calculate trading positions
            df['Positions'] = np.where(df['close'] > df['EWMA12'], 1, -1)
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='EWMA Exponentially-weighted moving average strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = EWMA_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))


    if strategy == 'Perfect':
        st.write('You selected the Perfect strategy.')
        st.subheader("(...in case you can predict the future...)")
        def perfect_strategy(df):
            # Calculate trading positions
            df['Positions'] = np.sign(df['Return'])
            # Calculate strategy returns
            df["Strategy_Ret"] = df["Positions"] * df["Return"]
            # Calculate cumulative strategy performance
            df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()

            df.dropna(inplace= True)
            
            # Create a Plotly figure for the strategy and asset's closing price
            fig = go.Figure()
            # Add asset's closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Asset Close Price'))
            # Add strategy performance
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Strategy Performance'))
            # Customize the figure layout
            fig.update_layout(
                title='Perfect strategy',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            # Customize the figure size and aspect ratio
            fig.update_layout(
                autosize=True,
                width=1200,
                height=800  # Adjust the height as needed
            )
            
            # Show the Plotly figure
            st.plotly_chart(fig)
            return EDA_data
        # Apply the Simple Momentum strategy
        strategy_data = perfect_strategy(EDA_data)

        # Display the strategy data
        st.dataframe(strategy_data.head(3))
        st.dataframe(strategy_data.tail(3))
        st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))



# Define the Conclousion page content
def conclusion():
    st.title("Conclusion")

    # Provide a summary of the key insights from the analysis
    st.header("Key Takeaways")
    st.write("In this dashboard, we explored various aspects of financial data and trading strategies related to Bitcoin (BTC). Here are some key takeaways:")

    # Add specific insights or findings based on your analysis
    st.write("1. Bitcoin (BTC) is a highly volatile and speculative asset, subject to rapid price fluctuations.")
    st.write("2. Past performance of trading strategies does not guarantee future results.")
    st.write("3. It's important to conduct thorough research and risk assessment before participating in cryptocurrency markets.")
    st.write("4. Diversification and risk management are crucial when investing in cryptocurrencies.")

    # Optionally, you can provide visualizations or charts to highlight key points
    st.header("Visual Summary")
    st.write("You can include visualizations or charts summarizing the analysis here.")

    # Provide recommendations or next steps
    st.header("Next Steps")
    st.write("Based on the analysis, consider the following next steps:")
    st.write("1. Continue to stay informed about the cryptocurrency market.")
    st.write("2. Consult with financial advisors and consider risk management strategies.")
    st.write("3. Diversify your investment portfolio to mitigate risks.")
    st.write("4. Be cautious when trading cryptocurrencies, and never invest more than you can afford to lose.")

    # Add a prominent disclaimer about the risks associated with Bitcoin
    st.warning("Disclaimer: Bitcoin (BTC) and cryptocurrency trading involve significant risks. Prices can be highly volatile and may result in substantial financial losses. Any decisions to invest in Bitcoin should be made at your own risk, and it is advisable to seek professional financial advice before making any investment decisions.")

    # Add any final thoughts or closing remarks
    st.header("Closing Remarks")
    st.write("In conclusion, this dashboard provides insights into financial data and trading strategies related to Bitcoin. It's essential to exercise caution and conduct thorough due diligence when dealing with cryptocurrencies. Your financial decisions are your responsibility, and you should be aware of the inherent risks involved.")

# ...




# Introduction page
if page == "Introduction":
    introduction()

# Explore Data page
if page == "Explore Data":
    explore_data()

# Trading Strategies page
if page == "Trading Strategies":
    trading_strategies()

# About page
if page == "Conclusion":
    conclusion()
