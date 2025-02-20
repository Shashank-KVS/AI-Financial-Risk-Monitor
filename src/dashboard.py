import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import logging
import logging.handlers
import yfinance as yf

# Configure logging
logger = logging.getLogger("Dashboard")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler("dashboard.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Page config
st.set_page_config(
    page_title="AI Financial Risk Monitor",
    page_icon="📈",
    layout="wide"
)

# Auto refresh every 10 seconds
st_autorefresh(interval=10000)

# Constants
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

def load_models():
    """Loads our trained risk assessment models from disk
    
    Returns:
        tuple: The random forest model, outlier detector, and feature scaler
               Returns None values if loading fails
    """
    try:
        rf_model = joblib.load("rf_model.pkl")
        lof_model = joblib.load("lof_model.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("Models loaded successfully")
        return rf_model, lof_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        logger.error(f"Error loading models: {str(e)}")
        return None, None, None

def prepare_features(data: pd.DataFrame) -> np.ndarray:
    """Cleans and prepares stock data for risk prediction
    
    Args:
        data: DataFrame containing the raw stock metrics
        
    Returns:
        Clean feature matrix ready for prediction
    """
    feature_columns = [
        "Daily Return",
        "Volatility",
        "Price_Range",
        "Price_Range_Pct",
        "Volume_Change",
        "Log Return",
        "Volatility_7d",
        "Volatility_7d_Ann",
        "Volatility_14d",
        "Volatility_14d_Ann",
        "Volatility_30d",
        "Volatility_30d_Ann"
    ]
    
    # Replace infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values
    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Extract features
    X = data[feature_columns].values
    
    # Handle any remaining infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X

def validate_numeric(value):
    """Checks if a value is a valid number
    
    Args:
        value: The value to check
        
    Returns:
        True if it's a valid number, False otherwise
    """
    try:
        float_val = float(value)
        # Only check if it's NaN or infinity
        if pd.isna(float_val) or float_val in [float('inf'), float('-inf')]:
            return False
        return True
    except (ValueError, TypeError):
        return False

def get_risk_metrics(ticker, df):
    """Analyzes current risk levels for a given stock
    
    Args:
        ticker: Stock symbol to analyze
        df: Historical stock data
        
    Returns:
        Dictionary with risk metrics including:
        - Current risk level (normal/high)
        - Risk probability
        - Daily returns
        - Current volatility
        - Volume changes
        - Price information
        Returns None if analysis fails
    """
    try:
        ticker_data = df[df["Ticker"] == ticker].sort_values("Date")
        if ticker_data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return None
        
        # Get latest data point
        latest = ticker_data.iloc[-1]
        
        # Load models
        rf_model, lof_model, scaler = load_models()
        if None in (rf_model, lof_model, scaler):
            return None
        
        # Prepare features
        features_df = pd.DataFrame([latest])
        X = prepare_features(features_df)
        
        # Basic validation
        if not isinstance(X, np.ndarray) or X.size == 0:
            logger.error("Invalid feature array")
            return None
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get LOF scores
        lof_scores = -lof_model.decision_function(X_scaled).reshape(-1, 1)
        
        # Combine features with LOF scores
        X_aug = np.hstack([X_scaled, lof_scores])
        
        # Get predictions
        prediction = rf_model.predict(X_aug)[0]
        probability = rf_model.predict_proba(X_aug)[0, 1]
        
        try:
            # Extract metrics
            metrics = {
                "prediction": int(prediction),
                "probability": float(probability),
                "daily_return": float(latest.get("Daily Return", 0)),
                "volatility": float(latest.get("Volatility", 0)),
                "volume_change": float(latest.get("Volume_Change", 0)),
                "price_range": float(latest.get("Price_Range", 0)),
                "price": float(latest.get("Close", 0)),
                "timestamp": latest.get("Date", pd.Timestamp.now())
            }
            
            # Validate numeric values
            for key, value in metrics.items():
                if key != "timestamp" and not validate_numeric(value):
                    logger.error(f"Invalid value for {key}: {value}")
                    return None
            
            return metrics
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting metrics: {str(e)}")
            return None
        
    except Exception as e:
        st.error(f"Error in risk metrics calculation: {str(e)}")
        logger.error(f"Error in risk metrics calculation: {str(e)}", exc_info=True)
        return None

def create_risk_gauge(value, title):
    """Creates a visual gauge showing risk levels
    
    Args:
        value: Risk value (0-1)
        title: Gauge title
        
    Returns:
        A Plotly gauge chart with green/yellow/red risk zones
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(height=200)
    return fig

def get_company_news(ticker_symbol):
    """Gets the latest news for a company
    
    Args:
        ticker_symbol: Company's stock symbol
        
    Returns:
        Recent news articles with titles, sources, and summaries
        Returns None if no news is found
    """
    try:
        # Create Ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Get news data
        news = ticker.news
        
        # Process and validate news items
        processed_news = []
        for item in news[:5]:  # Get latest 5 news items
            processed_item = {
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', ''),
                'providerPublishTime': item.get('providerPublishTime', ''),
                'type': item.get('type', ''),
                'relatedTickers': item.get('relatedTickers', []),
                'text': item.get('text', '')
            }
            # Only add items that have at least a title and text
            if processed_item['title'] and processed_item['text']:
                processed_news.append(processed_item)
        
        return processed_news
    except Exception as e:
        logger.error(f"Error fetching news for {ticker_symbol}: {str(e)}")
        return None

def get_company_info(ticker_symbol):
    """Gets key information about a company
    
    Args:
        ticker_symbol: Company's stock symbol
        
    Returns:
        Company details including name, sector, description,
        market cap, and other key metrics
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker_symbol}: {str(e)}")
        return None

def main():
    """Runs our financial risk monitoring dashboard
    
    Creates an interactive dashboard showing:
    - Live risk monitoring
    - Stock performance metrics
    - Company details and news
    - Historical trends
    
    Updates automatically every 10 seconds
    """
    st.title("AI Financial Risk Monitor 📈")
    
    try:
        # Load data
        df = pd.read_csv("multi_stock_data.csv", parse_dates=["Date"])
        
        if df.empty:
            st.error("No data available")
            return
            
        # Sidebar
        st.sidebar.header("Settings")
        selected_ticker = st.sidebar.selectbox("Select Stock", TICKERS)
        
        # Add Company Info Section
        company_info = get_company_info(selected_ticker)
        if company_info:
            st.sidebar.markdown("---")
            st.sidebar.header("Company Information")
            
            # Display company logo if available
            if company_info.get('logo_url'):
                st.sidebar.image(company_info['logo_url'], width=100)
            
            # Display company name and details
            st.sidebar.markdown(f"**{company_info.get('longName', selected_ticker)}**")
            st.sidebar.markdown(f"**Sector:** {company_info.get('sector', 'N/A')}")
            st.sidebar.markdown(f"**Industry:** {company_info.get('industry', 'N/A')}")
            
            # Display company description in expandable section
            with st.sidebar.expander("About Company"):
                st.write(company_info.get('longBusinessSummary', 'No description available'))
            
            # Display key statistics
            st.sidebar.markdown("**Key Statistics:**")
            st.sidebar.markdown(f"Market Cap: ${company_info.get('marketCap', 0):,.0f}")
            st.sidebar.markdown(f"P/E Ratio: {company_info.get('trailingPE', 'N/A')}")
            st.sidebar.markdown(f"52 Week High: ${company_info.get('fiftyTwoWeekHigh', 0):,.2f}")
            st.sidebar.markdown(f"52 Week Low: ${company_info.get('fiftyTwoWeekLow', 0):,.2f}")
        
        # Add News Section
        st.sidebar.markdown("---")
        st.sidebar.header("Latest Company News")
        news = get_company_news(selected_ticker)
        
        if news and len(news) > 0:
            for article in news:
                with st.sidebar.expander(article['title']):
                    if article.get('publisher'):
                        st.write(f"**Source:** {article['publisher']}")
                    
                    if article.get('providerPublishTime'):
                        publish_time = pd.Timestamp(article['providerPublishTime'], unit='s')
                        st.write(f"**Published:** {publish_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    if article.get('text'):
                        st.write(article['text'])
                    
                    if article.get('link'):
                        st.write(f"[Read More]({article['link']})")
                    
                    if article.get('relatedTickers'):
                        st.write(f"**Related Tickers:** {', '.join(article['relatedTickers'])}")
        else:
            st.sidebar.write("No recent news available for this company.")
        
        # Main content
        col1, col2, col3 = st.columns(3)
        
        metrics = get_risk_metrics(selected_ticker, df)
        
        with col1:
            st.subheader("Risk Analysis")
            if metrics is not None:
                st.plotly_chart(create_risk_gauge(metrics["probability"], "Risk Score"))
                
                risk_status = "High Risk" if metrics["prediction"] == 1 else "Normal"
                risk_color = "🔴" if metrics["prediction"] == 1 else "🟢"
                st.markdown(f"### Risk Status: {risk_status} {risk_color}")
                
                if validate_numeric(metrics["daily_return"]):
                    st.metric(
                        "Daily Return",
                        f"{metrics['daily_return']:.2%}",
                        delta=f"{metrics['daily_return']:.2%}"
                    )
            else:
                st.warning("Unable to calculate risk metrics")
        
        with col2:
            st.subheader("Key Metrics")
            if metrics is not None:
                if validate_numeric(metrics["price"]):
                    st.metric("Current Price", f"${metrics['price']:.2f}")
                if validate_numeric(metrics["volatility"]):
                    st.metric("Volatility", f"{metrics['volatility']:.2%}")
                if validate_numeric(metrics["volume_change"]):
                    st.metric("Volume Change", f"{metrics['volume_change']:.2%}")
        
        with col3:
            st.subheader("Technical Indicators")
            if metrics is not None:
                if validate_numeric(metrics["price_range"]):
                    st.metric("Price Range", f"${metrics['price_range']:.2f}")
                if validate_numeric(metrics["probability"]):
                    st.metric("Risk Probability", f"{metrics['probability']:.2%}")
        
        # Historical Data
        st.subheader("Historical Price Data")
        ticker_data = df[df["Ticker"] == selected_ticker].sort_values("Date")
        if not ticker_data.empty:
            fig = px.line(ticker_data, x="Date", y="Close", title=f"{selected_ticker} Price History")
            st.plotly_chart(fig)
        
        # Risk History
        st.subheader("Risk History")
        if not ticker_data.empty:
            risk_history = []
            for _, row in ticker_data.tail(30).iterrows():
                row_df = pd.DataFrame([row])
                metrics = get_risk_metrics(selected_ticker, row_df)
                if metrics is not None and validate_numeric(metrics["probability"]):
                    risk_history.append({
                        'Date': row['Date'],
                        'Risk Score': metrics['probability']
                    })
            
            if risk_history:
                risk_df = pd.DataFrame(risk_history)
                fig = px.line(risk_df, x="Date", y="Risk Score", 
                             title=f"{selected_ticker} Risk Score History")
                st.plotly_chart(fig)
        
        # Additional Information
        with st.expander("About the Risk Monitor"):
            st.write("""
                This AI-powered Financial Risk Monitor uses:
                - Machine Learning Classification (Random Forest)
                - Anomaly Detection (Local Outlier Factor)
                - Real-time market data processing
                - Feature scaling and normalization
                
                The risk assessment is based on multiple factors including:
                - Daily returns and volatility
                - Price movements and ranges
                - Volume changes
                - Technical indicators
                - Historical patterns
            """)
        
        st.markdown("---")
        st.markdown("_Dashboard refreshes automatically every 10 seconds_")
        
    except Exception as e:
        st.error(f"Error in dashboard: {str(e)}")
        logger.error(f"Error in dashboard: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()