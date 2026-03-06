import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.volatility import compute_volatility
from utils.obv import compute_rsi


def plot_interactive_plotly(df, divs, ticker, vol_method, vol_window):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)

    vol_series = compute_volatility(df, vol_method, window=vol_window)
    vol_label = vol_series.name
    vol_color_hex = '#e74c3c' if 'GARCH' in vol_method else '#2980b9'

    def hex_rgba(h, a=0.15):
        h = h.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{a})'

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.18, 0.15, 0.22],
        subplot_titles=(f"{ticker} Price", "OBV", "RSI", vol_label),
    )

    fig.add_trace(go.Ohlc(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='#26a69a')),
        decreasing=dict(line=dict(color='#ef5350')),
        hovertemplate='O:%{open:.2f} H:%{high:.2f} L:%{low:.2f} C:%{close:.2f}<extra>Price</extra>',
        hoverlabel=dict(bgcolor='#1e222d', font_color='white', font_size=12),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA20'], name='SMA20',
        line=dict(color='#4e9af1', width=1),
        hovertemplate='%{y:.2f}<extra>SMA20</extra>',
        hoverlabel=dict(bgcolor='#4e9af1', font_color='white'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA50'], name='SMA50',
        line=dict(color='#f0a500', width=1),
        hovertemplate='%{y:.2f}<extra>SMA50</extra>',
        hoverlabel=dict(bgcolor='#f0a500', font_color='white'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['OBV'], name='OBV',
        line=dict(color='#9b59b6', width=1.2),
        hovertemplate='%{y:,.0f}<extra>OBV</extra>',
        hoverlabel=dict(bgcolor='#9b59b6', font_color='white'),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#1abc9c', width=1.2),
        hovertemplate='%{y:.1f}<extra>RSI</extra>',
        hoverlabel=dict(bgcolor='#1abc9c', font_color='white'),
    ), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(231,76,60,0.6)', dash='dot'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='rgba(46,204,113,0.6)', dash='dot'), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=vol_series.index, y=vol_series.values,
        name=vol_label,
        line=dict(color=vol_color_hex, width=1.5),
        fill='tozeroy', fillcolor=hex_rgba(vol_color_hex, 0.15),
        hovertemplate='%{y:.3f}%<extra>' + vol_label + '</extra>',
        hoverlabel=dict(bgcolor=vol_color_hex, font_color='white'),
    ), row=4, col=1)

    for d in divs:
        c = '#2ecc71' if d['Type'] == 'Bullish' else '#e74c3c'
        fig.add_trace(go.Scatter(
            x=[d['P1_Date'], d['P2_Date']], y=[d['P1_Price'], d['P2_Price']],
            mode='lines+markers', line=dict(color=c, width=2, dash='dot'),
            marker=dict(size=7, color=c), showlegend=False,
            hovertemplate='%{y:.2f}<extra>' + d['Type'] + ' Price</extra>',
            hoverlabel=dict(bgcolor=c, font_color='white'),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[d['P1_Date'], d['P2_Date']], y=[d['P1_OBV'], d['P2_OBV']],
            mode='lines+markers', line=dict(color=c, width=2, dash='dot'),
            marker=dict(size=7, color=c), showlegend=False,
            hovertemplate='%{y:,.0f}<extra>' + d['Type'] + ' OBV</extra>',
            hoverlabel=dict(bgcolor=c, font_color='white'),
        ), row=2, col=1)

    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='rgba(200,200,200,0.5)', spikethickness=1, spikedash='dot',
        gridcolor='rgba(255,255,255,0.06)',
    )
    fig.update_yaxes(showspikes=False, gridcolor='rgba(255,255,255,0.06)',
                     zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_layout(
        hovermode='x',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=False,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        margin=dict(l=60, r=20, t=50, b=40),
    )

    fig_json = fig.to_json()
    html = f"""
<!DOCTYPE html><html><head>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>* {{ margin:0;padding:0;box-sizing:border-box }} body {{ background:#0e1117 }} #plt {{ width:100% }}</style>
</head><body><div id="plt"></div>
<script>
(function () {{
  const fig = {fig_json};
  Plotly.newPlot('plt', fig.data, fig.layout, {{ responsive:true, displayModeBar:false, scrollZoom:true }})
  .then(function (gd) {{
    const xaxisForRow = ['x','x2','x3','x4'];
    const anchorCurve = {{}};
    gd.data.forEach(function(trace, ci) {{
      const xref = trace.xaxis || 'x';
      const idx  = xaxisForRow.indexOf(xref);
      if (idx !== -1 && !(idx in anchorCurve)) anchorCurve[idx] = ci;
    }});
    function nearestIdx(traceX, xVal) {{
      if (!traceX || traceX.length === 0) return 0;
      const isDate = typeof traceX[0] === 'string';
      const target = isDate ? new Date(xVal).getTime() : Number(xVal);
      let best = 0, bestDist = Infinity;
      traceX.forEach(function(v, i) {{
        const vn = isDate ? new Date(v).getTime() : Number(v);
        const dist = Math.abs(vn - target);
        if (dist < bestDist) {{ bestDist = dist; best = i; }}
      }});
      return best;
    }}
    let busy = false;
    gd.on('plotly_hover', function(evt) {{
      if (busy || !evt.points || evt.points.length === 0) return;
      busy = true;
      const xVal = evt.points[0].x;
      const hpts = [];
      Object.keys(anchorCurve).forEach(function(subIdx) {{
        const ci = anchorCurve[subIdx];
        const trace = gd.data[ci];
        const ptIdx = nearestIdx(trace.x, xVal);
        hpts.push({{ curveNumber: ci, pointNumber: ptIdx }});
      }});
      Plotly.Fx.hover(gd, hpts);
      setTimeout(function() {{ busy = false; }}, 16);
    }});
    gd.on('plotly_unhover', function() {{
      if (busy) return;
      busy = true;
      Plotly.Fx.unhover(gd);
      setTimeout(function() {{ busy = false; }}, 16);
    }});
  }});
}})();
</script></body></html>
"""
    components.html(html, height=920, scrolling=False)