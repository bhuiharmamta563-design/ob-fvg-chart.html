import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------------
# Delta Exchange India: ऑर्डर बुक ज़ोन + सिग्नल + चार्ट
# ------------------------------------------------------------------
BASE = "https://api.india.delta.exchange"
COINS = ["UNIUSD", "AVAXUSD", "BTCUSD"]
TF = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

st.set_page_config(
    page_title="Delta ऑर्डर बुक ज़ोन",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def fmt(x):
    return f"{x:,.4f}" if x < 100 else f"{x:,.2f}"


# ---------------------------- डेटा --------------------------------
@st.cache_data(ttl=10)
def get_candles(symbol, res, n):
    end = int(time.time())
    start = end - TF[res] * n
    r = requests.get(
        f"{BASE}/v2/history/candles",
        params={"symbol": symbol, "resolution": res, "start": start, "end": end},
        timeout=10,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("result", []))
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def get_book(symbol, depth=100):
    r = requests.get(f"{BASE}/v2/l2orderbook/{symbol}", params={"depth": depth}, timeout=10)
    r.raise_for_status()
    res = r.json()["result"]

    def to_df(rows):
        d = pd.DataFrame(rows)
        if d.empty:
            return pd.DataFrame(columns=["price", "size", "notional"])
        d["price"] = d["price"].astype(float)
        d["size"] = d["size"].astype(float)
        d["notional"] = d["price"] * d["size"]
        return d

    return to_df(res.get("buy", [])), to_df(res.get("sell", []))


# ---------------------------- ज़ोन --------------------------------
def find_zones(df, mid, bucket_pct, top_n, min_strength, side):
    if df.empty:
        return []
    step = mid * bucket_pct / 100
    d = df.copy()
    d["b"] = np.floor(d["price"] / step).astype(int)
    g = d.groupby("b")["notional"].sum().reset_index()
    avg = g["notional"].mean()
    if not avg or np.isnan(avg):
        return []
    g["strength"] = g["notional"] / avg
    g = g[g["strength"] >= min_strength].sort_values("notional", ascending=False).head(top_n)
    return [
        {
            "side": side,
            "low": float(r.b * step),
            "high": float((r.b + 1) * step),
            "notional": float(r.notional),
            "strength": float(r.strength),
        }
        for r in g.itertuples()
    ]


def imbalance(bids, asks, mid, band_pct):
    lo, hi = mid * (1 - band_pct / 100), mid * (1 + band_pct / 100)
    b = bids.loc[bids["price"] >= lo, "notional"].sum()
    a = asks.loc[asks["price"] <= hi, "notional"].sum()
    tot = b + a
    return (b / tot) if tot else 0.5


# ---------------------------- सिग्नल -------------------------------
def make_signal(price, buy_z, sell_z, imb, near_pct):
    sup = [z for z in buy_z if z["low"] <= price]
    res = [z for z in sell_z if z["high"] >= price]
    sup = max(sup, key=lambda z: z["high"]) if sup else None
    res = min(res, key=lambda z: z["low"]) if res else None

    if sup and imb >= 0.55:
        dist = max(0.0, (price - sup["high"]) / price * 100)
        if dist <= near_pct:
            sl = sup["low"] * 0.999
            tp = res["low"] if res else price + 2 * (price - sl)
            return {"type": "BUY", "entry": price, "sl": sl, "tp": tp, "zone": sup}
    if res and imb <= 0.45:
        dist = max(0.0, (res["low"] - price) / price * 100)
        if dist <= near_pct:
            sl = res["high"] * 1.001
            tp = sup["high"] if sup else price - 2 * (sl - price)
            return {"type": "SELL", "entry": price, "sl": sl, "tp": tp, "zone": res}
    return {"type": "WAIT"}


def check_breaks(price, prev_price, prev_zones):
    out = []
    if prev_price is None:
        return out
    for z in prev_zones:
        rng = f"{fmt(z['low'])} – {fmt(z['high'])}"
        if z["side"] == "buy" and prev_price >= z["low"] and price < z["low"]:
            out.append(f"🔻 सपोर्ट ज़ोन टूटा ({rng}), ताक़त {z['strength']:.1f}x")
        if z["side"] == "sell" and prev_price <= z["high"] and price > z["high"]:
            out.append(f"🔺 रेज़िस्टेंस ज़ोन टूटा ({rng}), ताक़त {z['strength']:.1f}x")
    return out


# ---------------------------- चार्ट --------------------------------
def build_chart(df, zones, sig, height):
    fig = go.Figure(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="",
        )
    )
    step = df["time"].iloc[-1] - df["time"].iloc[-2] if len(df) > 1 else pd.Timedelta(minutes=1)
    x0, x1 = df["time"].iloc[0], df["time"].iloc[-1] + step * 10

    for z in zones:
        alpha = min(0.15 + 0.05 * z["strength"], 0.5)
        color = f"rgba(0,200,120,{alpha})" if z["side"] == "buy" else f"rgba(240,70,70,{alpha})"
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=z["low"], y1=z["high"],
            fillcolor=color, line_width=0, layer="below",
        )
        fig.add_annotation(
            x=x1, y=(z["low"] + z["high"]) / 2, xanchor="right", showarrow=False,
            text=("सपोर्ट " if z["side"] == "buy" else "रेज़िस्टेंस ") + f"{z['strength']:.1f}x",
            font=dict(size=11, color="white"),
        )

    if sig["type"] != "WAIT":
        for name, val, col in [("एंट्री", sig["entry"], "#ffd54f"),
                               ("SL", sig["sl"], "#ff5252"),
                               ("TP", sig["tp"], "#69f0ae")]:
            fig.add_hline(y=val, line_dash="dot", line_color=col, line_width=1,
                          annotation_text=name, annotation_position="left")

    lo, hi = df["low"].min(), df["high"].max()
    pad = (hi - lo) * 0.08
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",              # एक उँगली = खिसकाना, दो उँगली = ज़ूम
        xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", range=[lo - pad, hi + pad]),
        showlegend=False,
        uirevision="keep",           # रिफ्रेश पर आपका ज़ूम बना रहे
    )
    return fig


CHART_CONFIG = {
    "scrollZoom": True,              # माउस स्क्रॉल / दो उँगली पिंच ज़ूम
    "displaylogo": False,
    "displayModeBar": True,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
}


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# ---------------------------- UI ----------------------------------
st.markdown(
    "<style>.block-container{padding-top:1rem;padding-bottom:0.5rem;}</style>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([2, 2, 2])
symbol = c1.selectbox("कॉइन", COINS, label_visibility="collapsed")
tf = c2.selectbox("टाइमफ्रेम", list(TF), index=1, label_visibility="collapsed")
focus = c3.toggle("🎯 फ़ोकस मोड", value=False)

if focus:
    st.markdown(
        "<style>header[data-testid='stHeader']{display:none;}"
        ".block-container{padding-top:0.3rem;}</style>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("सेटिंग")
    bucket_pct = st.slider("ज़ोन की चौड़ाई (%)", 0.05, 1.0, 0.20, 0.05)
    top_n = st.slider("हर तरफ़ कितने ज़ोन", 1, 6, 3)
    min_strength = st.slider("न्यूनतम ताक़त (x)", 1.0, 5.0, 1.5, 0.1)
    band_pct = st.slider("इम्बैलेंस दायरा (± %)", 0.2, 3.0, 1.0, 0.1)
    near_pct = st.slider("सिग्नल के लिए ज़ोन से दूरी (%)", 0.1, 1.5, 0.4, 0.1)
    n_candles = st.slider("कैंडल की संख्या", 50, 300, 150, 10)
    refresh = st.slider("ऑटो-रिफ्रेश (सेकंड, 0 = बंद)", 0, 30, 5)

if not focus:
    st.title(f"📊 {symbol} ऑर्डर बुक ज़ोन")

for key, default in [("prev_zones", []), ("prev_price", None), ("alerts", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


def render():
    try:
        df = get_candles(symbol, tf, n_candles)
        bids, asks = get_book(symbol)
    except Exception as e:
        st.error(f"डेटा नहीं आया: {e}")
        return
    if df.empty or bids.empty or asks.empty:
        st.warning("इस कॉइन का डेटा अभी खाली है।")
        return

    mid = (bids["price"].max() + asks["price"].min()) / 2
    zones = (
        find_zones(bids, mid, bucket_pct, top_n, min_strength, "buy")
        + find_zones(asks, mid, bucket_pct, top_n, min_strength, "sell")
    )
    buy_z = [z for z in zones if z["side"] == "buy"]
    sell_z = [z for z in zones if z["side"] == "sell"]
    imb = imbalance(bids, asks, mid, band_pct)
    sig = make_signal(mid, buy_z, sell_z, imb, near_pct)

    # ज़ोन टूटने का अलर्ट
    stamp = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")
    new_alerts = check_breaks(mid, st.session_state["prev_price"], st.session_state["prev_zones"])
    for a in new_alerts:
        st.toast(a)
        st.session_state["alerts"].insert(0, f"{stamp}  {a}")
    st.session_state["alerts"] = st.session_state["alerts"][:20]
    st.session_state["prev_price"] = mid
    st.session_state["prev_zones"] = zones

    # सिग्नल की एक लाइन
    if sig["type"] == "BUY":
        st.success(
            f"🟢 BUY  |  एंट्री {fmt(sig['entry'])}  |  SL {fmt(sig['sl'])}  |  TP {fmt(sig['tp'])}"
        )
    elif sig["type"] == "SELL":
        st.error(
            f"🔴 SELL  |  एंट्री {fmt(sig['entry'])}  |  SL {fmt(sig['sl'])}  |  TP {fmt(sig['tp'])}"
        )
    else:
        st.info(f"⚪ अभी रुकें (WAIT)  |  भाव {fmt(mid)}")
    for a in new_alerts:
        st.warning(a)

    fig = build_chart(df, zones, sig, height=780 if focus else 600)
    show_chart(fig)

    if not focus:
        st.caption(f"खरीदार {imb*100:.0f}%  vs  बेचने वाले {(1-imb)*100:.0f}%  (भाव के ±{band_pct}% के अंदर)")
        st.progress(float(imb))

        rows = [
            {
                "तरफ़": "सपोर्ट" if z["side"] == "buy" else "रेज़िस्टेंस",
                "से": fmt(z["low"]),
                "तक": fmt(z["high"]),
                "ताक़त (x)": round(z["strength"], 1),
                "ऑर्डर वैल्यू (USD)": f"{z['notional']:,.0f}",
            }
            for z in sorted(zones, key=lambda z: -z["high"])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        with st.expander("🔔 ज़ोन टूटने के अलर्ट (आज का लॉग)"):
            if st.session_state["alerts"]:
                for a in st.session_state["alerts"]:
                    st.write(a)
            else:
                st.write("अभी कोई ज़ोन नहीं टूटा।")
        st.caption(f"आख़िरी अपडेट: {stamp}  •  यह सिर्फ़ सहायक संकेत है, ट्रेड का फ़ैसला आपका।")


st.fragment(render, run_every=refresh if refresh > 0 else None)()
