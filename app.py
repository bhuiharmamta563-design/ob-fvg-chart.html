import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

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
@st.cache_data(ttl=1)   # 1 सेकंड रिफ्रेश के लिए कैश भी 1 सेकंड
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


@st.cache_data(ttl=1)
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


@st.cache_data(ttl=5)
def get_context(symbol, res):
    """ट्रेंड (EMA 20 vs 50) और वॉल्यूम (आख़िरी बंद कैंडल / पिछले 20 का औसत)"""
    df = get_candles(symbol, res, 120)
    if len(df) < 55:
        return None
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    avg = df["volume"].iloc[-22:-2].mean()
    ratio = float(df["volume"].iloc[-2] / avg) if avg else 1.0
    return {"trend": "UP" if ema20 > ema50 else "DOWN", "vol_ratio": ratio}


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
def make_signal(price, buy_z, sell_z, imb, near_pct,
                ctx=None, trend_filter=True, min_vol=0.0, min_rr=0.0):
    sup = [z for z in buy_z if z["low"] <= price]
    res = [z for z in sell_z if z["high"] >= price]
    sup = max(sup, key=lambda z: z["high"]) if sup else None
    res = min(res, key=lambda z: z["low"]) if res else None

    cand = None
    if sup and imb >= 0.55:
        dist = max(0.0, (price - sup["high"]) / price * 100)
        if dist <= near_pct:
            sl = sup["low"] * 0.999
            tp = res["low"] if res else price + 2 * (price - sl)
            cand = ("BUY", sup, sl, tp)
    if cand is None and res and imb <= 0.45:
        dist = max(0.0, (res["low"] - price) / price * 100)
        if dist <= near_pct:
            sl = res["high"] * 1.001
            tp = sup["high"] if sup else price - 2 * (sl - price)
            cand = ("SELL", res, sl, tp)
    if cand is None:
        return {"type": "WAIT", "why": ""}

    side, zone, sl, tp = cand
    risk = abs(price - sl)
    rr = abs(tp - price) / risk if risk > 0 else 0.0
    notes, block = [], None

    if ctx:
        if trend_filter:
            if (ctx["trend"] == "UP") != (side == "BUY"):
                block = f"{side} सेटअप है, पर ट्रेंड {'नीचे' if ctx['trend'] == 'DOWN' else 'ऊपर'} है"
            else:
                notes.append("ट्रेंड ✔")
        if not block:
            if ctx["vol_ratio"] < min_vol:
                block = f"{side} सेटअप है, पर वॉल्यूम कम ({ctx['vol_ratio']:.1f}x)"
            else:
                notes.append(f"वॉल्यूम {ctx['vol_ratio']:.1f}x")
    if not block and min_rr and rr < min_rr:
        block = f"{side} सेटअप है, पर रिवॉर्ड:रिस्क कम ({rr:.1f})"
    if block:
        return {"type": "WAIT", "why": block}

    notes.append(f"RR {rr:.1f}")
    return {"type": side, "entry": price, "sl": sl, "tp": tp, "zone": zone, "notes": notes}


def beep(kind):
    """सिग्नल पर आवाज़ + वाइब्रेशन (ब्राउज़र में)"""
    freqs = [880, 1175] if kind == "BUY" else [660, 440]
    js = """<script>/*ID*/(function(){try{var P=window.parent;
    var A=P.AudioContext||P.webkitAudioContext;var c=new A();var t=c.currentTime;
    FREQS.forEach(function(f,i){var o=c.createOscillator(),g=c.createGain();
    o.frequency.value=f;o.connect(g);g.connect(c.destination);
    g.gain.setValueAtTime(0.25,t+i*0.22);
    g.gain.exponentialRampToValueAtTime(0.001,t+i*0.22+0.2);
    o.start(t+i*0.22);o.stop(t+i*0.22+0.22);});
    if(P.navigator.vibrate)P.navigator.vibrate([200,100,200]);}catch(e){}})();</script>"""
    js = js.replace("FREQS", json.dumps(freqs)).replace("ID", str(time.time()))
    components.html(js, height=0)


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
def build_chart(df, zones, sig, height, smooth=False, rev="keep", marks=None):
    if smooth:
        # स्मूथ मोड: क्लोज़ भाव की मुलायम (spline) लाइन
        fig = go.Figure(
            go.Scatter(
                x=df["time"],
                y=df["close"],
                mode="lines",
                line=dict(color="#4fc3f7", width=2.2, shape="spline", smoothing=1.2),
                name="",
            )
        )
        lo, hi = df["close"].min(), df["close"].max()
    else:
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
        lo, hi = df["low"].min(), df["high"].max()
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

    # BUY/SELL मार्कर: जहाँ-जहाँ सिग्नल आया
    for kind, sym, col in [("BUY", "triangle-up", "#00e676"), ("SELL", "triangle-down", "#ff1744")]:
        m = [x for x in (marks or []) if x["type"] == kind]
        if m:
            fig.add_trace(
                go.Scatter(
                    x=[x["time"] for x in m], y=[x["price"] for x in m],
                    mode="markers+text", text=[kind] * len(m),
                    textposition="top center" if kind == "BUY" else "bottom center",
                    textfont=dict(size=10, color=col),
                    marker=dict(symbol=sym, size=14, color=col, line=dict(width=1, color="white")),
                    name=kind, hoverinfo="skip",
                )
            )

    if sig["type"] != "WAIT":
        for name, val, col in [("एंट्री", sig["entry"], "#ffd54f"),
                               ("SL", sig["sl"], "#ff5252"),
                               ("TP", sig["tp"], "#69f0ae")]:
            fig.add_hline(y=val, line_dash="dot", line_color=col, line_width=1,
                          annotation_text=name, annotation_position="left")

    for z in zones:  # ज़ोन भी दिखें, इसलिए रेंज में जोड़ें
        lo, hi = min(lo, z["low"]), max(hi, z["high"])
    pad = (hi - lo) * 0.08
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",              # एक उँगली = खिसकाना, दो उँगली = पिंच ज़ूम
        xaxis_rangeslider_visible=False,
        xaxis=dict(fixedrange=False),   # दो उँगली दाएँ-बाएँ पिंच = समय ज़ूम
        yaxis=dict(side="right", range=[lo - pad, hi + pad],
                   fixedrange=False),   # दो उँगली ऊपर-नीचे पिंच = भाव ज़ूम
        showlegend=False,
        uirevision=rev,              # रिफ्रेश पर ज़ूम बना रहे, कॉइन/टाइमफ्रेम बदलने पर रीसेट
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

c1, c2 = st.columns(2)
symbol = c1.selectbox("कॉइन", COINS, label_visibility="collapsed")
tf = c2.selectbox("टाइमफ्रेम", list(TF), index=1, label_visibility="collapsed")

MODES = ["🎯 फ़ोकस", "⛶ फुल", "〰️ स्मूथ"]
if hasattr(st, "pills"):
    picked = st.pills("मोड", MODES, selection_mode="multi", label_visibility="collapsed")
else:
    picked = st.multiselect("मोड", MODES, label_visibility="collapsed")
picked = picked or []
full = MODES[1] in picked
focus = MODES[0] in picked or full      # फुल में फ़ोकस अपने-आप शामिल
smooth = MODES[2] in picked

# दो उँगली पिंच ठीक से चले, इसलिए चार्ट पर पेज-स्क्रॉल बंद
st.markdown(
    "<style>.stPlotlyChart, .js-plotly-plot{touch-action:none;}</style>",
    unsafe_allow_html=True,
)

# दो उँगली पिंच-ज़ूम (Plotly मोबाइल पर ख़ुद नहीं करता, इसलिए हम जोड़ते हैं)
#   दाएँ-बाएँ पिंच = समय ज़ूम | ऊपर-नीचे पिंच = भाव ज़ूम | तिरछा = दोनों
components.html(
    """
    <script>
    (function () {
      var P = window.parent, doc = P.document;
      if (P.__pinchZoomV2) return;
      P.__pinchZoomV2 = true;
      var st = null;
      function gdOf(t) {
        while (t && t.nodeType === 1) {
          if (t.classList.contains('js-plotly-plot')) return t;
          t = t.parentElement;
        }
        return null;
      }
      function pts(e) {
        var a = e.touches[0], b = e.touches[1];
        return {dx: Math.abs(a.clientX - b.clientX), dy: Math.abs(a.clientY - b.clientY),
                mx: (a.clientX + b.clientX) / 2, my: (a.clientY + b.clientY) / 2};
      }
      function wheel(el, d, x, y) {
        if (!el || !isFinite(d)) return;
        var left = d;
        while (Math.abs(left) > 0.01) {
          var step = Math.max(-20, Math.min(20, left));
          el.dispatchEvent(new P.WheelEvent('wheel', {deltaY: step, deltaMode: 0,
            clientX: x, clientY: y, bubbles: true, cancelable: true}));
          left -= step;
        }
      }
      doc.addEventListener('touchstart', function (e) {
        if (e.touches.length === 2) {
          var gd = gdOf(e.target);
          if (gd) { st = Object.assign({gd: gd}, pts(e)); e.preventDefault(); e.stopPropagation(); }
        }
      }, {capture: true, passive: false});
      doc.addEventListener('touchmove', function (e) {
        if (!st || e.touches.length !== 2) return;
        e.preventDefault(); e.stopPropagation();
        var c = pts(e), gd = st.gd;
        var sx = (st.dx > 30 && c.dx > 30) ? st.dx / c.dx : 1;
        var sy = (st.dy > 30 && c.dy > 30) ? st.dy / c.dy : 1;
        var ew = gd.querySelector('.ewdrag'), ns = gd.querySelector('.nsdrag');
        if (ew && ns) {
          if (sx !== 1) wheel(ew, 200 * Math.log(sx), c.mx, c.my);
          if (sy !== 1) wheel(ns, 200 * Math.log(sy), c.mx, c.my);
        } else {
          var main = gd.querySelector('.nsewdrag');
          wheel(main, 100 * Math.log(sx * sy), c.mx, c.my);
        }
        st.dx = c.dx; st.dy = c.dy; st.mx = c.mx; st.my = c.my;
      }, {capture: true, passive: false});
      function end(e) { if (e.touches.length < 2) st = null; }
      doc.addEventListener('touchend', end, true);
      doc.addEventListener('touchcancel', end, true);
    })();
    </script>
    """,
    height=0,
)

if focus:
    st.markdown(
        "<style>header[data-testid='stHeader']{display:none;}"
        ".block-container{padding-top:0.3rem;}</style>",
        unsafe_allow_html=True,
    )
if full:
    st.markdown(
        "<style>[data-testid='stSidebar'],[data-testid='collapsedControl'],"
        "[data-testid='stSidebarCollapsedControl']{display:none;}"
        ".block-container{padding:0.2rem 0.4rem 0 0.4rem;max-width:100%;}</style>",
        unsafe_allow_html=True,
    )
    # असली फुलस्क्रीन (ब्राउज़र/डेस्कटॉप/Android पर चलता है; iPhone Safari इसे सपोर्ट नहीं करता)
    components.html(
        """
        <button id="fs" style="width:100%;padding:8px;border-radius:8px;border:1px solid #555;
        background:#1e1e1e;color:#fff;font-size:15px;">⛶ पूरी स्क्रीन (Fullscreen)</button>
        <script>
        document.getElementById('fs').onclick = () => {
          try {
            const d = window.parent.document;
            if (d.fullscreenElement) d.exitFullscreen();
            else d.documentElement.requestFullscreen();
          } catch (e) { alert('इस ब्राउज़र में फुलस्क्रीन उपलब्ध नहीं है'); }
        };
        </script>
        """,
        height=48,
    )

with st.sidebar:
    st.header("सेटिंग")
    bucket_pct = st.slider("ज़ोन की चौड़ाई (%)", 0.05, 1.0, 0.20, 0.05)
    top_n = st.slider("हर तरफ़ कितने ज़ोन", 1, 6, 3)
    min_strength = st.slider("न्यूनतम ताक़त (x)", 1.0, 5.0, 1.5, 0.1)
    band_pct = st.slider("इम्बैलेंस दायरा (± %)", 0.2, 3.0, 1.0, 0.1)
    near_pct = st.slider("सिग्नल के लिए ज़ोन से दूरी (%)", 0.1, 1.5, 0.4, 0.1)
    n_candles = st.slider("कैंडल की संख्या", 50, 300, 150, 10)
    refresh = st.slider("भाव/सिग्नल रिफ्रेश (सेकंड, 0 = बंद)", 0, 30, 1)
    chart_refresh = st.slider("चार्ट रिफ्रेश (सेकंड)", 2, 60, 10)
    st.subheader("सिग्नल")
    sound = st.toggle("🔔 सिग्नल पर आवाज़/वाइब्रेशन", value=True)
    trend_filter = st.toggle("ट्रेंड फ़िल्टर (EMA 20/50)", value=True)
    min_vol = st.slider("न्यूनतम वॉल्यूम (औसत का x)", 0.0, 3.0, 0.8, 0.1)
    min_rr = st.slider("न्यूनतम रिवॉर्ड:रिस्क (0 = बंद)", 0.0, 3.0, 1.2, 0.1)

if not focus:
    st.title(f"📊 {symbol} ऑर्डर बुक ज़ोन")

for key, default in [("prev_zones", []), ("prev_price", None), ("alerts", []),
                     ("marks", []), ("last_sig", {}), ("sym", None), ("last_beep", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default


def book_state():
    """ऑर्डर बुक से भाव, ज़ोन, इम्बैलेंस और सिग्नल (1 सेकंड कैश)"""
    bids, asks = get_book(symbol)
    if bids.empty or asks.empty:
        return None
    mid = (bids["price"].max() + asks["price"].min()) / 2
    zones = (
        find_zones(bids, mid, bucket_pct, top_n, min_strength, "buy")
        + find_zones(asks, mid, bucket_pct, top_n, min_strength, "sell")
    )
    buy_z = [z for z in zones if z["side"] == "buy"]
    sell_z = [z for z in zones if z["side"] == "sell"]
    imb = imbalance(bids, asks, mid, band_pct)
    try:
        ctx = get_context(symbol, tf)
    except Exception:
        ctx = None
    sig = make_signal(mid, buy_z, sell_z, imb, near_pct, ctx, trend_filter, min_vol, min_rr)
    return {"mid": mid, "zones": zones, "imb": imb, "sig": sig}


# हर सेकंड: सिर्फ़ भाव + सिग्नल + अलर्ट बदलते हैं, चार्ट नहीं छेड़ा जाता
def live_top():
    try:
        st_ = book_state()
    except Exception as e:
        st.error(f"डेटा नहीं आया: {e}")
        return
    if not st_:
        st.warning("इस कॉइन का डेटा अभी खाली है।")
        return
    mid, zones, sig = st_["mid"], st_["zones"], st_["sig"]

    if st.session_state["sym"] != symbol:      # कॉइन बदला: पुराना रेफ़रेंस साफ़
        st.session_state["sym"] = symbol
        st.session_state["prev_price"] = None
        st.session_state["prev_zones"] = []

    # नया सिग्नल आया? → मार्कर + आवाज़
    last = st.session_state["last_sig"].get(symbol, "WAIT")
    if sig["type"] != last:
        st.session_state["last_sig"][symbol] = sig["type"]
        if sig["type"] != "WAIT":
            st.session_state["marks"].append(
                {"symbol": symbol, "time": pd.Timestamp.now(tz="Asia/Kolkata"),
                 "type": sig["type"], "price": mid}
            )
            st.session_state["marks"] = st.session_state["marks"][-40:]
            st.toast(f"{'🟢' if sig['type'] == 'BUY' else '🔴'} {symbol}: {sig['type']} सिग्नल @ {fmt(mid)}")
            if sound and time.time() - st.session_state["last_beep"] > 10:
                st.session_state["last_beep"] = time.time()
                beep(sig["type"])

    stamp = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")
    new_alerts = check_breaks(mid, st.session_state["prev_price"], st.session_state["prev_zones"])
    for a in new_alerts:
        st.toast(a)
        st.session_state["alerts"].insert(0, f"{stamp}  {a}")
    st.session_state["alerts"] = st.session_state["alerts"][:20]
    st.session_state["prev_price"] = mid
    st.session_state["prev_zones"] = zones

    price = f"भाव {fmt(mid)}"
    if sig["type"] == "BUY":
        st.success(f"🟢 BUY  |  {price}  |  SL {fmt(sig['sl'])}  |  TP {fmt(sig['tp'])}  |  {' · '.join(sig['notes'])}")
    elif sig["type"] == "SELL":
        st.error(f"🔴 SELL  |  {price}  |  SL {fmt(sig['sl'])}  |  TP {fmt(sig['tp'])}  |  {' · '.join(sig['notes'])}")
    else:
        why = f"  |  {sig['why']}" if sig.get("why") else ""
        st.info(f"⚪ अभी रुकें (WAIT)  |  {price}{why}")
    for a in new_alerts:
        st.warning(a)


# चार्ट अपनी धीमी रफ़्तार से (ताकि हर सेकंड झपकी/रीसेट न हो)
def live_chart():
    try:
        df = get_candles(symbol, tf, n_candles)
        st_ = book_state()
    except Exception as e:
        st.error(f"चार्ट का डेटा नहीं आया: {e}")
        return
    if df.empty or not st_:
        st.warning("इस कॉइन का डेटा अभी खाली है।")
        return
    fig = build_chart(
        df, st_["zones"], st_["sig"],
        height=900 if full else (780 if focus else 600),
        smooth=smooth,
        rev=f"{symbol}-{tf}-{'s' if smooth else 'c'}",
        marks=[m for m in st.session_state["marks"] if m["symbol"] == symbol],
    )
    show_chart(fig)


# हर सेकंड: इम्बैलेंस बार, ज़ोन टेबल, अलर्ट लॉग
def live_bottom():
    try:
        st_ = book_state()
    except Exception:
        return
    if not st_:
        return
    zones, imb = st_["zones"], st_["imb"]
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
    st.caption("यह सिर्फ़ सहायक संकेत है, ट्रेड का फ़ैसला आपका।")


every = refresh if refresh > 0 else None
st.fragment(live_top, run_every=every)()
st.fragment(live_chart, run_every=chart_refresh)()
if not focus:
    st.fragment(live_bottom, run_every=every)()
