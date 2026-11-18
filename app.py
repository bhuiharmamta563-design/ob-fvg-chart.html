import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import hmac
import hashlib
import json

# Page Configuration
st.set_page_config(
    page_title="Delta Exchange Auto-Trading Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Delta Exchange Fully Automated Trading Bot")
st.markdown("Yeh bot Streamlit Secrets se API keys read karke live orderbook ke aadhar par automatically trades execute karega.")

# 1. Load API Keys securely from Streamlit Secrets
try:
    API_KEY = st.secrets["DELTA_API_KEY"]
    API_SECRET = st.secrets["DELTA_API_SECRET"]
    st.sidebar.success("✅ API Keys loaded successfully from Secrets!")
except Exception as e:
    st.sidebar.error("❌ API Keys not found in Streamlit Secrets! Please configure secrets.toml")
    API_KEY = ""
    API_SECRET = ""

# Sidebar Configuration
mode = st.sidebar.selectbox("Trading Mode", ["Testnet (Recommended)", "Live Exchange"])
if "Testnet" in mode:
    BASE_URL = "https://testnet-api.delta.exchange"
else:
    BASE_URL = "https://api.delta.exchange"

symbol = st.sidebar.selectbox("Select Asset", ["BTC-PERP", "ETH-PERP"])
order_size = st.sidebar.number_input("Order Size (Contracts)", min_value=1, value=1)

# Master Switch for Auto-Trading
auto_trade_enabled = st.sidebar.checkbox("🚀 Enable Live Auto-Trading (Automatic Execution)", value=False)

# Helper function to sign Delta Exchange API requests (Authentication)
def generate_signature(method, endpoint, payload_string=''):
    import time
    timestamp = str(int(time.time()))
    signature_data = timestamp + method + endpoint + payload_string
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        signature_data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return timestamp, signature

# Function to place order on Delta Exchange
def place_delta_order(product_id, side, size):
    if not API_KEY or not API_SECRET:
        st.error("API Keys missing! Cannot place order.")
        return False
    
    endpoint = "/v2/orders"
    url = BASE_URL + endpoint
    
    payload = {
        "product_id": product_id,
        "size": size,
        "side": side,  # "buy" or "sell"
        "order_type": "market"
    }
    payload_str = json.dumps(payload)
    
    timestamp, signature = generate_signature("POST", endpoint, payload_str)
    
    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload_str, timeout=5)
        res_json = response.json()
        if response.status_code == 200 and res_json.get("success"):
            st.success(f"🎉 Order Executed Successfully! Side: {side.upper()} | Size: {size}")
            return True
        else:
            st.error(f"Order Failed: {res_json.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"API Request Exception: {str(e)}")
        return False

# Fetch product ID and orderbook
@st.cache_data(ttl=5)
def fetch_delta_data(symbol_name):
    try:
        products_url = f"{BASE_URL}/v2/products"
        resp = requests.get(products_url, timeout=5).json()
        product_id = None
        if "result" in resp:
            for p in resp["result"]:
                if p["symbol"] == symbol_name:
                    product_id = p["id"]
                    break
        
        if not product_id:
            return None, None, None, "Product ID not found"

        book_url = f"{BASE_URL}/v2/products/{product_id}/orderbook"
        book_resp = requests.get(book_url, timeout=5).json()
        
        if "result" in book_resp:
            bids = book_resp["result"].get("bids", [])
            asks = book_resp["result"].get("asks", [])
            return product_id, bids, asks, None
        else:
            return product_id, None, None, "Failed to fetch orderbook"
    except Exception as e:
        return None, None, None, str(e)

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Live Market View ({symbol})")
    price_placeholder = st.empty()
    
    # Static/Live chart layout placeholder
    fig = go.Figure(data=[go.Candlestick(
        x=['10:00', '10:05', '10:10', '10:15'],
        open=[67100, 67200, 67450, 67350],
        high=[67350, 67500, 67600, 67400],
        low=[66950, 67150, 67300, 67000],
        close=[67200, 67450, 67350, 67050]
    )])
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚡ Bot Signals & Execution Status")
    signal_box = st.empty()
    execution_log = st.empty()
    orderbook_container = st.container()

# Fetch data and evaluate signals
product_id, bids, asks, error = fetch_delta_data(symbol)

if error:
    st.error(f"Error: {error}")
else:
    if bids and asks:
        total_bid_qty = sum([float(b[1]) for b in bids[:5]])
        total_ask_qty = sum([float(a[1]) for a in asks[:5]])
        
        current_signal = "NEUTRAL"
        if total_bid_qty > total_ask_qty * 1.2:
            signal_box.success("🟢 SIGNAL: STRONG BUY (Support Wall Active)")
            current_signal = "BUY"
        elif total_ask_qty > total_bid_qty * 1.2:
            signal_box.error("🔴 SIGNAL: STRONG SELL (Resistance Wall Active)")
            current_signal = "SELL"
        else:
            signal_box.info("⚪ NEUTRAL: Market Consolidating")

        with orderbook_container:
            st.markdown("**Top Asks (Resistance)**")
            st.dataframe(pd.DataFrame(asks[:5], columns=["Price", "Size"]), use_container_width=True, hide_index=True)
            st.markdown("**Top Bids (Support)**")
            st.dataframe(pd.DataFrame(bids[:5], columns=["Price", "Size"]), use_container_width=True, hide_index=True)
            price_placeholder.metric("Best Bid Price", f"₹ {bids[0][0]}")

        # --- AUTO-TRADING EXECUTION LOGIC ---
        if auto_trade_enabled and product_id:
            if current_signal == "BUY":
                execution_log.warning("⚡ Auto-Bot triggered: Placing BUY Order...")
                place_delta_order(product_id, "buy", order_size)
            elif current_signal == "SELL":
                execution_log.warning("⚡ Auto-Bot triggered: Placing SELL Order...")
                place_delta_order(product_id, "sell", order_size)
            else:
                execution_log.info("💤 Bot waiting for strong signal...")
        else:
            execution_log.info("ℹ️ Auto-Trading is OFF. Turn it on from sidebar to trade automatically.")

# Auto-refresh loop to keep checking signals every few seconds
if st.button("🔄 Refresh Data / Run Cycle"):
    st.rerilled = True
        
