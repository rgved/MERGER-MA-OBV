"""
streamlit run hover_fix.py
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("✅ Hover Sync — Fixed")

np.random.seed(42)
n   = 120
idx = pd.date_range("2024-01-01", periods=n, freq="D")

close  = 100 + np.cumsum(np.random.randn(n) * 1.5)
open_  = close - np.random.rand(n) * 2
high   = close + np.random.rand(n) * 2
low    = close - np.random.rand(n) * 2
volume = np.random.randint(100_000, 500_000, n).astype(float)
obv    = np.cumsum(np.where(np.diff(close, prepend=close[0]) > 0, volume, -volume))

delta    = np.diff(close, prepend=close[0])
avg_gain = pd.Series(np.where(delta > 0, delta, 0.0)).rolling(14).mean().values
avg_loss = pd.Series(np.where(delta < 0, -delta, 0.0)).rolling(14).mean().values
rsi      = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
vol_vals = np.abs(np.random.randn(n)) * 20 + 15

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.45, 0.18, 0.15, 0.22],
    subplot_titles=("Price", "OBV", "RSI", "Volatility"),
)

# Row 1: OHLC (visual only) + invisible Scatter anchor
fig.add_trace(go.Ohlc(
    x=idx, open=open_, high=high, low=low, close=close,
    name='Price',
    increasing=dict(line=dict(color='#26a69a')),
    decreasing=dict(line=dict(color='#ef5350')),
    hoverinfo='skip',
), row=1, col=1)

price_text = [
    f'O: {o:.2f}  H: {h:.2f}<br>L: {l:.2f}  C: {c:.2f}'
    for o, h, l, c in zip(open_, high, low, close)
]
fig.add_trace(go.Scatter(
    x=idx, y=close,
    name='Price Anchor',
    mode='lines',
    line=dict(color='rgba(0,0,0,0)', width=0),
    text=price_text,
    hovertemplate='%{text}<extra>Price</extra>',
    hoverlabel=dict(bgcolor='#1e222d', font_color='white', font_size=12),
    showlegend=False,
), row=1, col=1)

# Row 2: OBV
fig.add_trace(go.Scatter(
    x=idx, y=obv, name='OBV',
    line=dict(color='#9b59b6', width=1.5),
    hovertemplate='%{y:,.0f}<extra>OBV</extra>',
    hoverlabel=dict(bgcolor='#9b59b6', font_color='white'),
), row=2, col=1)

# Row 3: RSI
fig.add_trace(go.Scatter(
    x=idx, y=rsi, name='RSI',
    line=dict(color='#1abc9c', width=1.5),
    hovertemplate='%{y:.1f}<extra>RSI</extra>',
    hoverlabel=dict(bgcolor='#1abc9c', font_color='white'),
), row=3, col=1)
fig.add_hline(y=70, line=dict(color='rgba(231,76,60,0.6)', dash='dot'), row=3, col=1)
fig.add_hline(y=30, line=dict(color='rgba(46,204,113,0.6)', dash='dot'), row=3, col=1)

# Row 4: Volatility
fig.add_trace(go.Scatter(
    x=idx, y=vol_vals, name='Volatility',
    line=dict(color='#e74c3c', width=1.5),
    fill='tozeroy', fillcolor='rgba(231,76,60,0.1)',
    hovertemplate='%{y:.2f}%<extra>Volatility</extra>',
    hoverlabel=dict(bgcolor='#e74c3c', font_color='white'),
), row=4, col=1)

fig.update_xaxes(
    showspikes=True, spikemode='across', spikesnap='cursor',
    spikecolor='rgba(200,200,200,0.5)', spikethickness=1, spikedash='dot',
    gridcolor='rgba(255,255,255,0.06)',
)
fig.update_yaxes(
    showspikes=False,
    gridcolor='rgba(255,255,255,0.06)',
    zerolinecolor='rgba(255,255,255,0.1)',
)
fig.update_layout(
    hovermode='x',
    xaxis_rangeslider_visible=False,
    height=860, showlegend=False,
    paper_bgcolor='#0e1117', plot_bgcolor='#161b22',
    font=dict(color='#c9d1d9', size=11),
    margin=dict(l=60, r=20, t=50, b=40),
)

fig_json = fig.to_json()

html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0e1117; }}
  </style>
</head>
<body>
<div id="plt"></div>
<script>
(function () {{
  const fig = {fig_json};

  Plotly.newPlot('plt', fig.data, fig.layout, {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
  }}).then(function (gd) {{

    // Build anchor map — skip OHLC/candlestick, only Scatter works with Fx.hover
    const subplotAnchor = {{}};
    gd._fullData.forEach(function (trace, ci) {{
      if (trace.type === 'ohlc' || trace.type === 'candlestick') return;
      const xref = trace.xaxis || 'x';
      if (!(xref in subplotAnchor)) {{
        subplotAnchor[xref] = ci;
      }}
    }});

    function nearestIdx(xs, xVal) {{
      if (!xs || xs.length === 0) return 0;
      const isDate = typeof xs[0] === 'string';
      const target = isDate ? new Date(xVal).getTime() : Number(xVal);
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < xs.length; i++) {{
        const v = isDate ? new Date(xs[i]).getTime() : Number(xs[i]);
        const d = Math.abs(v - target);
        if (d < bestDist) {{ bestDist = d; best = i; }}
      }}
      return best;
    }}

    // CRITICAL: this flag must be set BEFORE Fx.hover is called,
    // not after. Fx.hover synchronously fires plotly_hover again,
    // which without this guard would immediately call Fx.hover again,
    // which clears the first hover before it renders.
    let syncing = false;

    gd.on('plotly_hover', function (evt) {{
      // If WE triggered this event via Fx.hover, ignore it
      if (syncing) return;
      if (!evt.points || evt.points.length === 0) return;

      const xVal = evt.points[0].x;
      const pts  = [];

      Object.keys(subplotAnchor).forEach(function (xref) {{
        const ci    = subplotAnchor[xref];
        const trace = gd._fullData[ci];
        pts.push({{ curveNumber: ci, pointNumber: nearestIdx(trace.x, xVal) }});
      }});

      // Set flag BEFORE calling Fx.hover so the re-entrant event is blocked
      syncing = true;
      Plotly.Fx.hover(gd, pts);
      syncing = false;
    }});

    gd.on('plotly_unhover', function () {{
      if (syncing) return;
      Plotly.Fx.unhover(gd);
    }});

  }});
}})();
</script>
</body>
</html>
"""

components.html(html, height=880, scrolling=False)