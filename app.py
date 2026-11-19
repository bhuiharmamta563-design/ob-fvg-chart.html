import json
import time

import numpy as np
import pandas as pd
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


# ------------------ लिक्विडिटी + फंडिंग/OI ------------------------
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=5)
def get_ticker(symbol):
    r = requests.get(f"{BASE}/v2/tickers/{symbol}", timeout=10)
    r.raise_for_status()
    return r.json().get("result", {}) or {}


@st.cache_data(ttl=60)
def get_price_move_6h(symbol):
    df = get_candles(symbol, "1h", 10)
    if len(df) < 7:
        return None
    return float((df["close"].iloc[-1] / df["close"].iloc[-7] - 1) * 100)


def slippage(levels, trade_usd, mid):
    """ऑर्डर बुक में चलकर देखो: trade_usd भरने पर औसत भाव मिड से कितना दूर (%)"""
    left, qty, spent = trade_usd, 0.0, 0.0
    for price, notional in zip(levels["price"], levels["notional"]):
        take = min(notional, left)
        qty += take / price
        spent += take
        left -= take
        if left <= 1e-9:
            break
    if left > 1e-6 or qty == 0:
        return None          # बुक में इतना साइज़ है ही नहीं
    return abs(spent / qty - mid) / mid * 100


def liquidity_info(bids, asks, mid, trade_usd):
    spread = (asks["price"].min() - bids["price"].max()) / mid * 100
    band = 0.5
    bid_depth = bids[bids["price"] >= mid * (1 - band / 100)]["notional"].sum()
    ask_depth = asks[asks["price"] <= mid * (1 + band / 100)]["notional"].sum()
    buy_slip = slippage(asks.sort_values("price"), trade_usd, mid)
    sell_slip = slippage(bids.sort_values("price", ascending=False), trade_usd, mid)
    if buy_slip is None or sell_slip is None:
        level = 2
    else:
        worst = max(buy_slip, sell_slip)
        level = 2 if (spread > 0.15 or worst > 0.3) else 1 if (spread > 0.05 or worst > 0.1) else 0
    return {"spread": spread, "bid_depth": bid_depth, "ask_depth": ask_depth,
            "buy_slip": buy_slip, "sell_slip": sell_slip, "level": level}


def derivs_info(symbol, funding_thr):
    t = get_ticker(symbol)
    funding = num(t.get("funding_rate"))        # % में माना गया है
    oi_val = num(t.get("oi_value_usd"))
    oi_chg = num(t.get("oi_change_usd_6h"))
    oi_pct = None
    if oi_val and oi_chg is not None and oi_val - oi_chg > 0:
        oi_pct = oi_chg / (oi_val - oi_chg) * 100
    mv = get_price_move_6h(symbol)

    f_text, f_flag = "सामान्य", None
    if funding is not None:
        if funding >= funding_thr:
            f_text, f_flag = "लॉन्ग की भीड़ — उलटी चाल (गिरावट) का रिस्क", "long"
        elif funding <= -funding_thr:
            f_text, f_flag = "शॉर्ट की भीड़ — अचानक उछाल (short squeeze) का रिस्क", "short"

    read = "OI में बड़ा बदलाव नहीं"
    if oi_pct is not None and mv is not None and abs(oi_pct) >= 0.3 and abs(mv) >= 0.2:
        if mv > 0 and oi_pct > 0:
            read = "नए लॉन्ग बन रहे हैं — तेज़ी में दम"
        elif mv > 0:
            read = "शॉर्ट कवरिंग — तेज़ी कमज़ोर"
        elif oi_pct > 0:
            read = "नए शॉर्ट बन रहे हैं — गिरावट में दम"
        else:
            read = "लॉन्ग कट रहे हैं — गिरावट कमज़ोर"
    return {"funding": funding, "f_text": f_text, "f_flag": f_flag,
            "oi_val": oi_val, "oi_pct": oi_pct, "mv": mv, "read": read}


def liq_levels(df, price, top=2, bin_pct=0.15):
    """अनुमानित लिक्विडेशन ज़ोन.
    हर कैंडल को 'एंट्री' मानकर 10x/25x/50x/100x लीवरेज के लिक्विडेशन भाव निकालते हैं,
    वॉल्यूम से वज़न देते हैं, और सिर्फ़ वही रखते हैं जिन्हें भाव बाद में छुआ नहीं।
    यह असली लिक्विडेशन डेटा नहीं, सिर्फ़ अनुमान है।"""
    n = len(df)
    if n < 20 or not price:
        return []
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    vol = df["volume"].to_numpy()
    typ = (hi + lo + df["close"].to_numpy()) / 3
    min_after = np.append(np.minimum.accumulate(lo[::-1])[::-1][1:], np.inf)
    max_after = np.append(np.maximum.accumulate(hi[::-1])[::-1][1:], -np.inf)
    width = price * bin_pct / 100
    mm = 0.005                       # मेंटेनेंस मार्जिन (मोटा अनुमान)
    acc = {"long": {}, "short": {}}
    for L, w in [(10, 0.35), (25, 0.30), (50, 0.20), (100, 0.15)]:
        long_p = typ * (1 - 1 / L + mm)
        short_p = typ * (1 + 1 / L - mm)
        ok_l = (long_p < price) & (long_p < min_after)
        ok_s = (short_p > price) & (short_p > max_after)
        for side, arr, ok in (("long", long_p, ok_l), ("short", short_p, ok_s)):
            for p, v in zip(arr[ok], vol[ok]):
                k = int(round(p / width))
                acc[side][k] = acc[side].get(k, 0.0) + float(v) * w
    out = []
    for side, d in acc.items():
        if not d:
            continue
        mean = sum(d.values()) / len(d)
        for k, wgt in sorted(d.items(), key=lambda kv: -kv[1])[:top]:
            strength = wgt / mean if mean else 1.0
            if strength >= 1.3:
                out.append({"side": "liq_" + side, "price": k * width, "strength": strength})
    return out


def depth_profile(bids, asks, mid, step_pct=0.05, keep=120):
    """ऑर्डर बुक को छोटे भाव-डिब्बों में बाँटो: कहाँ कितने BUY (0) और SELL (1) ऑर्डर पड़े हैं"""
    step = mid * step_pct / 100
    rows = []
    for d, side in ((bids, 0), (asks, 1)):
        if d.empty:
            continue
        g = d.groupby((d["price"] / step).round().astype(int))["notional"].sum()
        rows += [[float(k * step), float(v), side] for k, v in g.items()]
    rows = sorted(rows, key=lambda r: -r[1])[:keep]
    return {"step": step, "rows": rows}


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
CHART_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
html,body{margin:0;height:100%;background:#131722;overflow:hidden;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 -webkit-user-select:none;user-select:none;-webkit-touch-callout:none;}
#wrap{position:relative;width:100%;height:100%;}
#zones{position:absolute;left:0;top:0;right:0;bottom:0;pointer-events:none;overflow:hidden;z-index:1;}
#c{position:absolute;left:0;top:0;right:0;bottom:0;z-index:2;}
.z{position:absolute;left:0;display:none;box-sizing:border-box;z-index:1;}
#dp{position:absolute;left:0;top:0;z-index:2;pointer-events:none;}
.z span{position:absolute;left:6px;top:1px;font-size:11px;color:#fff;opacity:.92;white-space:nowrap;}
#lg{position:absolute;left:8px;top:6px;z-index:5;color:#d1d4dc;font-size:12px;
 pointer-events:none;line-height:1.4;text-shadow:0 0 3px #131722;}
#rs{position:absolute;left:8px;bottom:34px;z-index:5;width:34px;height:34px;border-radius:17px;
 border:1px solid #363a45;background:rgba(30,34,45,.92);color:#d1d4dc;font-size:17px;padding:0;}
#msg{position:absolute;left:0;top:0;right:0;bottom:0;display:none;align-items:center;
 justify-content:center;color:#ef9a9a;z-index:9;font-size:14px;text-align:center;padding:20px;}
</style></head><body>
<div id="wrap"><div id="zones"></div><div id="c"></div><div id="lg"></div>
<button id="rs" title="रीसेट">&#8634;</button><div id="msg"></div></div>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function () {
  var SMOOTH = __SMOOTH__, OFF = 19800, P = window.parent;   // 19800 = IST (+5:30)
  var wrap = document.getElementById('wrap'), el = document.getElementById('c'),
      zl = document.getElementById('zones'), lg = document.getElementById('lg'),
      msg = document.getElementById('msg');
  function fail(t) { msg.style.display = 'flex'; msg.textContent = t; }
  if (!window.LightweightCharts) { fail('चार्ट लाइब्रेरी लोड नहीं हुई — इंटरनेट/ब्लॉकर जाँचें'); return; }
  try { void P.document; } catch (e) { fail('चार्ट को डेटा नहीं मिल पा रहा (ब्राउज़र ने रोका)'); return; }

  var chart = LightweightCharts.createChart(el, {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { type: 'solid', color: 'rgba(0,0,0,0)' }, textColor: '#b2b5be', fontSize: 12 },
    grid: { vertLines: { color: 'rgba(42,46,57,.55)' }, horzLines: { color: 'rgba(42,46,57,.55)' } },
    crosshair: { mode: 0, vertLine: { color: '#758696', labelBackgroundColor: '#2a2e39' },
                 horzLine: { color: '#758696', labelBackgroundColor: '#2a2e39' } },
    rightPriceScale: { borderColor: '#2a2e39', scaleMargins: { top: .12, bottom: .12 } },
    timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false,
                 rightOffset: 8, barSpacing: 9, minBarSpacing: 1.5 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
    handleScale: { axisPressedMouseMove: { time: true, price: true }, mouseWheel: true, pinch: true },
    kineticScroll: { touch: true, mouse: false }
  });
  new ResizeObserver(function () { chart.resize(el.clientWidth, el.clientHeight); }).observe(wrap);

  var series;
  if (SMOOTH) {
    series = chart.addAreaSeries({ lineColor: '#4fc3f7', topColor: 'rgba(79,195,247,.28)',
      bottomColor: 'rgba(79,195,247,0)', lineWidth: 2, lineType: 2, crosshairMarkerRadius: 4 });
  } else {
    series = chart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350' });
  }

  // ऊपर-नीचे दो उँगली पिंच = भाव (Y) ज़ूम
  var yZoom = 1;
  function provider(orig) {
    var r = orig();
    if (r !== null && yZoom !== 1 && r.priceRange) {
      var mn = r.priceRange.minValue, mx = r.priceRange.maxValue, c = (mn + mx) / 2, h = (mx - mn) / 2 * yZoom;
      return { priceRange: { minValue: c - h, maxValue: c + h }, margins: r.margins };
    }
    return r;
  }
  function refreshScale() { series.applyOptions({ autoscaleInfoProvider: function (o) { return provider(o); } }); }
  refreshScale();

  var tp = null;
  function tpts(e) {
    var a = e.touches[0], b = e.touches[1];
    return { dx: Math.abs(a.clientX - b.clientX), dy: Math.abs(a.clientY - b.clientY) };
  }
  wrap.addEventListener('touchstart', function (e) {
    if (e.touches.length === 2) {
      var p = tpts(e);
      if (p.dy > 25 && p.dy > p.dx * 1.2) { tp = p; e.stopPropagation(); e.preventDefault(); } else { tp = null; }
    }
  }, { capture: true, passive: false });
  wrap.addEventListener('touchmove', function (e) {
    if (!tp || e.touches.length !== 2) return;
    e.stopPropagation(); e.preventDefault();
    var p = tpts(e);
    if (p.dy > 10) { yZoom = Math.min(20, Math.max(0.05, yZoom * tp.dy / p.dy)); tp.dy = p.dy; refreshScale(); }
  }, { capture: true, passive: false });
  function endT(e) { if (e.touches.length < 2) tp = null; }
  wrap.addEventListener('touchend', endT, true);
  wrap.addEventListener('touchcancel', endT, true);

  // ---- लेजेंड (OHLC) ----
  var prec = 2, label = '', lastBar = null, hovering = false;
  function fx(v) { return Number(v).toFixed(prec); }
  var DEP = null;
  function hint() {
    return (DEP && DEP.rows && DEP.rows.length)
      ? '<br><span style="color:#787b86;font-size:10px">हरा/लाल बार = पेंडिंग BUY/SELL ऑर्डर</span>' : '';
  }
  function showLegend(b) {
    if (!b) { lg.innerHTML = '<b>' + label + '</b>' + hint(); return; }
    if (b.open !== undefined) {
      var col = b.close >= b.open ? '#26a69a' : '#ef5350';
      lg.innerHTML = '<b>' + label + '</b><br><span style="color:' + col + '">O ' + fx(b.open) +
        '&nbsp; H ' + fx(b.high) + '&nbsp; L ' + fx(b.low) + '&nbsp; C ' + fx(b.close) + '</span>' + hint();
    } else {
      lg.innerHTML = '<b>' + label + '</b><br>' + fx(b.value) + hint();
    }
  }
  chart.subscribeCrosshairMove(function (p) {
    var b = (p && p.time && p.seriesData) ? (p.seriesData.get(series) || null) : null;
    hovering = !!b; showLegend(b || lastBar);
  });

  // ---- ज़ोन (पारदर्शी बैंड, चार्ट के पीछे) ----
  var Z = [], zEls = [];
  function setZones(zs) {
    Z = zs;
    while (zEls.length < zs.length) { var d = document.createElement('div'); d.className = 'z';
      d.appendChild(document.createElement('span')); zl.appendChild(d); zEls.push(d); }
    while (zEls.length > zs.length) { zl.removeChild(zEls.pop()); }
    zs.forEach(function (z, i) {
      var e = zEls[i], rgb, a, txt, dash = false;
      if (z.side === 'buy' || z.side === 'sell') {            // RS = सपोर्ट / रेज़िस्टेंस (ऑर्डर बुक की दीवार)
        rgb = z.side === 'buy' ? '0,200,120' : '240,70,70';
        a = Math.min(0.12 + 0.04 * z.strength, 0.4);
        txt = 'RS ' + z.strength.toFixed(1) + 'x';
      } else {                                                 // LQ = अनुमानित लिक्विडेशन
        var isLong = z.side === 'liq_long';
        rgb = isLong ? '255,167,38' : '206,147,216';
        a = Math.min(0.10 + 0.05 * z.strength, 0.4); dash = true;
        txt = 'LQ ' + (isLong ? 'लॉन्ग' : 'शॉर्ट') + ' ' + z.strength.toFixed(1) + 'x';
      }
      e.style.background = 'rgba(' + rgb + ',' + a + ')';
      e.style.borderTop = e.style.borderBottom = '1px ' + (dash ? 'dashed' : 'solid') + ' rgba(' + rgb + ',.8)';
      e.firstChild.textContent = txt;
      e.firstChild.style.left = dash ? '38%' : '6px';
    });
  }
  // पेंडिंग ऑर्डर के बार (दाईं किनारी से बाईं ओर), चार्ट के पीछे
  var dpc = document.createElement('canvas'); dpc.id = 'dp'; zl.appendChild(dpc);
  var dctx = dpc.getContext('2d');
  function drawDepth(w) {
    var dpr = window.devicePixelRatio || 1, cw = zl.clientWidth, ch = zl.clientHeight;
    if (dpc.width !== Math.round(cw * dpr) || dpc.height !== Math.round(ch * dpr)) {
      dpc.width = Math.round(cw * dpr); dpc.height = Math.round(ch * dpr);
      dpc.style.width = cw + 'px'; dpc.style.height = ch + 'px';
    }
    dctx.setTransform(dpr, 0, 0, dpr, 0, 0); dctx.clearRect(0, 0, cw, ch);
    if (!DEP || !DEP.rows || !DEP.rows.length) return;
    var mx = 0; DEP.rows.forEach(function (r) { if (r[1] > mx) mx = r[1]; });
    var maxW = w * 0.28, half = DEP.step / 2;
    DEP.rows.forEach(function (r) {
      var y1 = series.priceToCoordinate(r[0] + half), y2 = series.priceToCoordinate(r[0] - half);
      if (y1 === null || y2 === null) return;
      var bw = Math.max(2, Math.sqrt(r[1] / mx) * maxW);
      dctx.fillStyle = r[2] === 0 ? 'rgba(38,166,154,.5)' : 'rgba(239,83,80,.5)';
      dctx.fillRect(w - bw, Math.min(y1, y2), bw, Math.max(1, Math.abs(y2 - y1) - 0.5));
    });
  }
  function frame() {
    var w = chart.timeScale().width();
    drawDepth(w);
    for (var i = 0; i < Z.length; i++) {
      var e = zEls[i]; if (!e) continue;
      var y1 = series.priceToCoordinate(Z[i].high), y2 = series.priceToCoordinate(Z[i].low);
      if (y1 === null || y2 === null) { e.style.display = 'none'; continue; }
      e.style.display = 'block'; e.style.top = Math.min(y1, y2) + 'px';
      e.style.height = Math.max(Math.abs(y2 - y1), 2) + 'px'; e.style.width = w + 'px';
    }
    requestAnimationFrame(frame);
  }

  // ---- एंट्री / SL / TP लाइनें ----
  var lines = [];
  function setLines(s) {
    if (!s || s.type === 'WAIT') { lines.forEach(function (l) { series.removePriceLine(l); }); lines = []; return; }
    var spec = [['एंट्री', s.entry, '#ffd54f'], ['SL', s.sl, '#ff5252'], ['TP', s.tp, '#69f0ae']];
    if (lines.length === 3) { spec.forEach(function (a, i) { lines[i].applyOptions({ price: a[1] }); }); return; }
    spec.forEach(function (a) {
      lines.push(series.createPriceLine({ price: a[1], color: a[2], lineWidth: 1, lineStyle: 2,
        axisLabelVisible: true, title: a[0] }));
    });
  }

  // ---- BUY/SELL मार्कर ----
  var bars = [], markSig = '';
  function setMarks(ms) {
    var out = [];
    ms.forEach(function (m) {
      var t = m.time + OFF, bt = null;
      for (var i = bars.length - 1; i >= 0; i--) { if (bars[i].time <= t) { bt = bars[i].time; break; } }
      if (bt === null) return;
      var buy = m.type === 'BUY';
      out.push({ time: bt, position: buy ? 'belowBar' : 'aboveBar', color: buy ? '#00e676' : '#ff1744',
                 shape: buy ? 'arrowUp' : 'arrowDown', text: m.type });
    });
    out.sort(function (a, b) { return a.time - b.time; });
    var sg = JSON.stringify(out); if (sg === markSig) return; markSig = sg;
    series.setMarkers(out);
  }

  // ---- डेटा लेना: पूरा रीलोड नहीं, सिर्फ़ आख़िरी कैंडल अपडेट (झपकी/ज़ूम-रीसेट नहीं) ----
  var cur = { key: null, lastT: 0 };
  function showDefault() {
    var n = bars.length;
    chart.priceScale('right').applyOptions({ autoScale: true });
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 100), to: n + 8 });
  }
  function apply(d) {
    if (!d || !d.candles || !d.candles.length) return;
    prec = d.prec; label = d.label;
    bars = d.candles.map(function (r) { return { time: r[0] + OFF, open: r[1], high: r[2], low: r[3], close: r[4] }; });
    var data = SMOOTH ? bars.map(function (b) { return { time: b.time, value: b.close }; }) : bars;
    if (cur.key !== d.key) {
      series.applyOptions({ priceFormat: { type: 'price', precision: prec, minMove: Math.pow(10, -prec) } });
      yZoom = 1; refreshScale();
      series.setData(data); cur.key = d.key; showDefault();
    } else {
      for (var i = 0; i < data.length; i++) { if (data[i].time >= cur.lastT) series.update(data[i]); }
    }
    cur.lastT = data[data.length - 1].time;
    lastBar = data[data.length - 1];
    if (!hovering) showLegend(lastBar);
    DEP = d.depth || null;
    setZones(d.zones || []); setLines(d.sig); setMarks(d.marks || []);
  }
  document.getElementById('rs').addEventListener('click', function () { yZoom = 1; refreshScale(); showDefault(); });

  var ver = -1;
  function poll() {
    if (P.__lwData && P.__lwVer !== ver) { ver = P.__lwVer; try { apply(P.__lwData); } catch (e) { console.error(e); } }
  }
  setInterval(poll, 250); poll();
  requestAnimationFrame(frame);
})();
</script></body></html>
"""


def chart_html(smooth):
    return CHART_HTML.replace("__SMOOTH__", "true" if smooth else "false")


# ---------------------------- UI ----------------------------------
st.markdown(
    "<style>.block-container{padding-top:1rem;padding-bottom:0.5rem;}</style>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
symbol = c1.selectbox("कॉइन", COINS, label_visibility="collapsed")
tf = c2.selectbox("टाइमफ्रेम", list(TF), index=1, label_visibility="collapsed")

MODES = ["🎯 फ़ोकस", "⛶ फुल", "〰️ स्मूथ", "💥 LQ", "📶 ऑर्डर बार"]
if hasattr(st, "pills"):
    picked = st.pills("मोड", MODES, selection_mode="multi", default=MODES[3:], label_visibility="collapsed")
else:
    picked = st.multiselect("मोड", MODES, default=MODES[3:], label_visibility="collapsed")
picked = picked or []
full = MODES[1] in picked
focus = MODES[0] in picked or full      # फुल में फ़ोकस अपने-आप शामिल
smooth = MODES[2] in picked
show_liq = MODES[3] in picked          # अनुमानित लिक्विडेशन ज़ोन
show_depth = MODES[4] in picked        # पेंडिंग BUY/SELL ऑर्डर के बार

# छिपे (0 ऊँचाई वाले) डेटा-कंपोनेंट जगह न घेरें
st.markdown(
    "<style>div[data-testid='stElementContainer']:has(iframe[height='0'])"
    "{position:absolute;width:0;height:0;overflow:hidden;margin:0;}</style>",
    unsafe_allow_html=True,
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

with st.expander("⚙️ सेटिंग (ज़ोन, सिग्नल, साइज़)"):
    bucket_pct = st.slider("ज़ोन की चौड़ाई (%)", 0.05, 1.0, 0.20, 0.05)
    top_n = st.slider("हर तरफ़ कितने ज़ोन", 1, 6, 3)
    min_strength = st.slider("न्यूनतम ताक़त (x)", 1.0, 5.0, 1.5, 0.1)
    band_pct = st.slider("इम्बैलेंस दायरा (± %)", 0.2, 3.0, 1.0, 0.1)
    near_pct = st.slider("सिग्नल के लिए ज़ोन से दूरी (%)", 0.1, 1.5, 0.4, 0.1)
    n_candles = st.slider("कैंडल की संख्या", 50, 300, 150, 10)
    refresh = st.slider("लाइव रिफ्रेश (सेकंड, 0 = बंद)", 0, 30, 1)
    st.subheader("सिग्नल")
    sound = st.toggle("🔔 सिग्नल पर आवाज़/वाइब्रेशन", value=True)
    trend_filter = st.toggle("ट्रेंड फ़िल्टर (EMA 20/50)", value=True)
    min_vol = st.slider("न्यूनतम वॉल्यूम (औसत का x)", 0.0, 3.0, 0.8, 0.1)
    min_rr = st.slider("न्यूनतम रिवॉर्ड:रिस्क (0 = बंद)", 0.0, 3.0, 1.2, 0.1)
    st.subheader("लिक्विडिटी / फंडिंग")
    trade_usd = st.number_input("मेरी पोज़ीशन का साइज़ (USD, लीवरेज के बाद)", 10.0, 1_000_000.0, 500.0, step=100.0)
    funding_thr = st.slider("फंडिंग चेतावनी (± %)", 0.01, 0.30, 0.05, 0.01)

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
    liq = liquidity_info(bids, asks, mid, trade_usd)
    depth = depth_profile(bids, asks, mid)
    return {"mid": mid, "zones": zones, "imb": imb, "sig": sig, "liq": liq, "depth": depth}


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
    # चेतावनियाँ: पतली बुक और फंडिंग की भीड़ (फ़ोकस मोड में भी दिखती हैं)
    liq = st_["liq"]
    if liq["level"] == 2:
        st.warning(f"⚠️ पतली बुक: स्प्रेड {liq['spread']:.3f}% और ${trade_usd:,.0f} पर स्लिपेज ज़्यादा — ट्रेड से बचें।")
    try:
        dv = derivs_info(symbol, funding_thr)
    except Exception:
        dv = None
    if dv and dv["f_flag"]:
        extra = ""
        if (dv["f_flag"] == "long" and sig["type"] == "BUY") or (dv["f_flag"] == "short" and sig["type"] == "SELL"):
            extra = " — इस सिग्नल में ज़्यादा सावधानी रखें"
        st.warning(f"⚠️ फंडिंग {dv['funding']:+.4f}%: {dv['f_text']}{extra}")

    for a in new_alerts:
        st.warning(a)


# चार्ट को हर सेकंड नया डेटा भेजना (चार्ट ख़ुद दोबारा नहीं बनता, सिर्फ़ आख़िरी कैंडल बदलती है)
def live_feed():
    try:
        df = get_candles(symbol, tf, n_candles)
        st_ = book_state()
    except Exception as e:
        st.error(f"चार्ट का डेटा नहीं आया: {e}")
        return
    if df.empty or not st_:
        st.warning("इस कॉइन का डेटा अभी खाली है।")
        return
    df = df.drop_duplicates("time")
    mid, sig = st_["mid"], st_["sig"]
    lq_zones = []
    if show_liq:
        lq_zones = [
            {"side": l["side"], "low": l["price"] * 0.9996, "high": l["price"] * 1.0004,
             "strength": l["strength"]}
            for l in liq_levels(df, mid)
        ]
    payload = {
        "key": f"{symbol}|{tf}",
        "label": f"{symbol} · {tf}",
        "prec": 6 if mid < 1 else 4 if mid < 100 else 2,
        "candles": [
            [int(t.timestamp()), o, h, l, c]
            for t, o, h, l, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])
        ],
        "zones": [{"side": z["side"], "low": z["low"], "high": z["high"], "strength": z["strength"]}
                  for z in st_["zones"]] + lq_zones,
        "depth": st_["depth"] if show_depth else None,
        "sig": {k: sig.get(k) for k in ("type", "entry", "sl", "tp")},
        "marks": [{"time": int(m["time"].timestamp()), "type": m["type"]}
                  for m in st.session_state["marks"] if m["symbol"] == symbol],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    components.html(
        f"<script>var P=window.parent;P.__lwData={data};P.__lwVer=(P.__lwVer||0)+1;</script>",
        height=0,
    )


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

    # 💧 लिक्विडिटी
    liq = st_["liq"]
    icon = ["🟢", "🟡", "🔴"][liq["level"]]
    def slip_txt(v):
        return "बुक में इतना साइज़ नहीं" if v is None else f"{v:.3f}%"
    st.markdown(
        f"{icon} **लिक्विडिटी:** स्प्रेड {liq['spread']:.3f}%  ·  भाव के ±0.5% में "
        f"खरीद ${liq['bid_depth']:,.0f} / बिक्री ${liq['ask_depth']:,.0f}  ·  "
        f"${trade_usd:,.0f} पर स्लिपेज: खरीदने में {slip_txt(liq['buy_slip'])}, "
        f"बेचने में {slip_txt(liq['sell_slip'])}"
    )

    # 📊 फंडिंग और OI
    try:
        dv = derivs_info(symbol, funding_thr)
    except Exception:
        dv = None
    if dv and dv["funding"] is not None:
        st.markdown(f"💸 **फंडिंग:** {dv['funding']:+.4f}% — {dv['f_text']}")
    else:
        st.caption("फंडिंग का डेटा नहीं मिला।")
    if dv and dv["oi_val"] is not None:
        chg = f" ({dv['oi_pct']:+.1f}% / 6 घंटे)" if dv["oi_pct"] is not None else ""
        mv = f"  ·  6 घंटे में भाव {dv['mv']:+.1f}%" if dv["mv"] is not None else ""
        st.markdown(f"📊 **OI:** ${dv['oi_val']:,.0f}{chg}{mv} → {dv['read']}")
    else:
        st.caption("OI का डेटा नहीं मिला।")
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
chart_h = 900 if full else (780 if focus else 600)
components.html(chart_html(smooth), height=chart_h)      # चार्ट एक बार बनता है, बार-बार नहीं
st.fragment(live_feed, run_every=every)()                # और डेटा हर सेकंड पहुँचता है
if not focus:
    st.fragment(live_bottom, run_every=every)()
