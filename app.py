import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

# Page Configuration
st.set_page_config(
    page_title="Delta Exchange Live Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title & Header
st.title("🚀 Delta Exchange Live Orderbook & Signal Bot")
st.markdown("Yeh dashboard aapke Delta Exchange API se live data fetch karke reversal zones aur signals dikhayega.")

# Sidebar - Configuration & API Keys
st.sidebar.header("⚙️ Settings & API Config")

# Toggle for Testnet vs Live
mode = st.sidebar.selectbox("Trading Mode", ["Testnet (Recommended for testing)", "Live Exchange"])
if "Testnet" in mode:
    BASE_URL = "https://testnet-api.delta.exchange"
else:
    BASE_URL = "https://api.delta.exchange"

# API Keys input (Securely fetched from Streamlit Secrets or user input)
# st.secrets se lene ke liye: st.secrets.get("DELTA_API_KEY", "")
api_key = st.sidebar.text_input("Delta API Key", type="password")
api_secret = st.sidebar.text_input("Delta API Secret", type="password")

symbol = st.sidebar.selectbox("Select Asset", ["BTC-PERP", "ETH-PERP"])

# Function to fetch product ID and orderbook from Delta Exchange
@st.cache_data(ttl=10)
def fetch_delta_orderbook(symbol_name):
    try:
        # 1. Get Product ID first
        products_url = f"{BASE_URL}/v2/products"
        resp = requests.get(products_url, timeout=5).json()
        product_id = None
        if "result" in resp:
            for p in resp["result"]:
                if p["symbol"] == symbol_name:
                    product_id = p["id"]
                    break
        
        if not product_id:
            return None, None, "Product ID not found"

        # 2. Get Orderbook using Product ID
        book_url = f"{BASE_URL}/v2/products/{product_id}/orderbook"
        book_resp = requests.get(book_url, timeout=5).json()
        
        if "result" in book_resp:
            return book_resp["result"].get("bids", []), book_resp["result"].get("asks", []), None
        else:
            return None, None, "Failed to fetch orderbook"
    except Exception as e:
        return None, None, str(e)

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Live Price & Market View ({symbol})")
    
    # Placeholder for live price metrics
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Dummy candlestick chart for UI layout (Aap ise live kline API se bhi jod sakte hain)
    fig = go.Figure(data=[go.Candlestick(
        x=['2026-06-01 10:00', '2026-06-01 10:05', '2026-06-01 10:10', '2026-06-01 10:15'],
        open=[67100, 67200, 67450, 67350],
        high=[67350, 67500, 67600, 67400],
        low=[66950, 67150, 67300, 67000],
        close=[67200, 67450, 67350, 67050]
    )])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=400
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚡ Orderbook Walls & Signals")
    signal_box = st.empty()
    orderbook_container = st.container()

# Fetch and Display Live Orderbook Data
bids, asks, error = fetch_delta_orderbook(symbol)

if error:
    st.error(f"Error connecting to Delta Exchange: {error}")
else:
    if bids and asks:
        total_bid_qty = sum([float(b[1]) for b in bids[:5]])
        total_ask_qty = sum([float(a[1]) for a in asks[:5]])
        
        # Signal logic based on Bid/Ask imbalance
        if total_bid_qty > total_ask_qty * 1.2:
            signal_box.success("🟢 SIGNAL: STRONG BUY (Support Wall Active)")
        elif total_ask_qty > total_bid_qty * 1.2:
            signal_box.error("🔴 SIGNAL: STRONG SELL (Resistance Wall Active)")
        else:
            signal_box.info("⚪ NEUTRAL: Market Consolidating")

        with orderbook_container:
            st.markdown("**Top Asks (Resistance / Sell Walls)**")
            asks_df = pd.DataFrame(asks[:5], columns=["Price", "Size"])
            st.dataframe(asks_df, use_container_width=True, hide_index=True)

            st.markdown("**Top Bids (Support / Buy Walls)**")
            bids_df = pd.DataFrame(bids[:5], columns=["Price", "Size"])
            st.dataframe(bids_df, use_container_width=True, hide_index=True)
            
            price_placeholder.metric("Current Best Bid", f"₹ {bids[0][0] if bids else 'N/A'}")

# Auto-refresh button or timer
if st.button("🔄 Refresh Data"):
    st.rerun()
  
