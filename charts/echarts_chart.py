import json
import numpy as np
import streamlit.components.v1 as components

from utils.volatility import compute_volatility
from utils.obv import compute_rsi


def plot_echarts_synchronized(df, divs, ticker, vol_method, vol_window):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'], period=14)
    vol_series = compute_volatility(df, vol_method, window=vol_window)

    dates = [str(ts)[:10] for ts in df.index]

    def to_list(arr):
        return [
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)
            for v in arr
        ]

    ohlc_data = [
        [
            round(float(df['Open'].iloc[i]), 2),
            round(float(df['Close'].iloc[i]), 2),
            round(float(df['Low'].iloc[i]), 2),
            round(float(df['High'].iloc[i]), 2),
        ]
        for i in range(len(df))
    ]

    sma20_data = to_list(df['SMA20'].values)
    sma50_data = to_list(df['SMA50'].values)
    obv_data = to_list(df['OBV'].values)
    rsi_data = to_list(df['RSI'].values)
    vol_data = to_list(vol_series.values)
    vol_label = vol_series.name
    vol_color = '#e74c3c' if 'GARCH' in vol_method else '#3498db'
    vol_color_rgba = 'rgba(231,76,60,0.12)' if 'GARCH' in vol_method else 'rgba(52,152,219,0.12)'

    price_mark_lines = []
    obv_mark_lines = []
    for d in divs:
        d1 = str(d['P1_Date'])[:10]
        d2 = str(d['P2_Date'])[:10]
        color = '#2ecc71' if d['Type'] == 'Bullish' else '#e74c3c'
        label = d['Type']
        price_mark_lines.append([
            {"name": label, "xAxis": d1, "yAxis": round(float(d['P1_Price']), 2),
             "itemStyle": {"color": color}, "label": {"show": False}},
            {"xAxis": d2, "yAxis": round(float(d['P2_Price']), 2)},
        ])
        obv_mark_lines.append([
            {"name": label, "xAxis": d1, "yAxis": round(float(d['P1_OBV']), 0),
             "itemStyle": {"color": color},
             "label": {"show": True, "formatter": label, "color": color}},
            {"xAxis": d2, "yAxis": round(float(d['P2_OBV']), 0)},
        ])

    html = f"""
<!DOCTYPE html><html><head>
  <meta charset="utf-8"/>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <style>
    * {{ margin:0;padding:0;box-sizing:border-box }}
    html, body {{ background:#0e1117;width:100% }}
    .chart-wrap {{ width:100% }}
    #chart-price {{ height:360px }}
    #chart-obv   {{ height:200px }}
    #chart-rsi   {{ height:160px }}
    #chart-vol   {{ height:190px }}
    .panel-title {{
      color:#8b9dc3;font-family:'Segoe UI',sans-serif;font-size:11px;
      letter-spacing:1px;text-transform:uppercase;padding:6px 12px 0;background:#0e1117;
    }}
  </style>
</head>
<body>
<div class="chart-wrap">
  <div><div class="panel-title">{ticker} — Price / OHLC</div><div id="chart-price"></div></div>
  <div><div class="panel-title">OBV</div><div id="chart-obv"></div></div>
  <div><div class="panel-title">RSI (14)</div><div id="chart-rsi"></div></div>
  <div><div class="panel-title">{vol_label}</div><div id="chart-vol"></div></div>
</div>
<script>
(function() {{
  const dates     = {json.dumps(dates)};
  const ohlcData  = {json.dumps(ohlc_data)};
  const sma20Data = {json.dumps(sma20_data)};
  const sma50Data = {json.dumps(sma50_data)};
  const obvData   = {json.dumps(obv_data)};
  const rsiData   = {json.dumps(rsi_data)};
  const volData   = {json.dumps(vol_data)};
  const priceMark = {json.dumps(price_mark_lines)};
  const obvMark   = {json.dumps(obv_mark_lines)};

  const BG='#0e1117', GRID_BG='#161b22', GRID_LINE='rgba(255,255,255,0.06)';
  const AXIS_TICK='#5a6a8a', AXIS_LBL='#8b9dc3', TOOLTIP_BG='#1e2535';

  const sharedAxisPointer = {{
    link: [{{ xAxisIndex: 'all' }}],
    label: {{ backgroundColor: '#2c3e50' }}
  }};
  const sharedTooltip = {{
    trigger: 'axis',
    axisPointer: {{
      type: 'cross',
      crossStyle: {{ color: 'rgba(180,180,200,0.4)', width: 1 }},
      lineStyle:  {{ color: 'rgba(180,180,200,0.3)', width: 1, type: 'dashed' }},
      label: {{ show: true, backgroundColor: '#2c3e50', color: '#e0e6f0', fontSize: 11 }}
    }},
    backgroundColor: TOOLTIP_BG, borderColor: '#2c3e50', borderWidth: 1,
    textStyle: {{ color: '#e0e6f0', fontSize: 11, fontFamily: 'Segoe UI, sans-serif' }},
    extraCssText: 'box-shadow:0 4px 16px rgba(0,0,0,0.5);border-radius:6px;',
  }};
  const sharedGrid = {{
    left:'60px', right:'20px', top:'8px', bottom:'28px',
    backgroundColor: GRID_BG, show: true, borderColor: 'transparent',
  }};
  const xAxisBase = {{
    type:'category', data:dates, boundaryGap:true,
    axisLine:  {{ lineStyle:{{ color:AXIS_TICK }} }},
    axisTick:  {{ lineStyle:{{ color:AXIS_TICK }} }},
    axisLabel: {{ color:AXIS_LBL, fontSize:10 }},
    splitLine: {{ show:true, lineStyle:{{ color:GRID_LINE }} }},
  }};
  const yAxisBase = {{
    type:'value', scale:true,
    axisLine:  {{ lineStyle:{{ color:AXIS_TICK }} }},
    axisTick:  {{ lineStyle:{{ color:AXIS_TICK }} }},
    axisLabel: {{ color:AXIS_LBL, fontSize:10 }},
    splitLine: {{ show:true, lineStyle:{{ color:GRID_LINE }} }},
  }};

  const chartPrice = echarts.init(document.getElementById('chart-price'), null, {{ renderer:'canvas', useDirtyRect:true }});
  chartPrice.setOption({{
    backgroundColor: BG, axisPointer: sharedAxisPointer,
    tooltip: {{ ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.seriesType === 'candlestick') {{
            const v = p.value;
            html += `<div>O:<b>${{v[1]}}</b> H:<b>${{v[4]}}</b> L:<b>${{v[3]}}</b> C:<b>${{v[2]}}</b></div>`;
          }} else if (p.value !== null && p.value !== undefined) {{
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{p.color}};margin-right:5px"></span>${{p.seriesName}}: <b>${{typeof p.value==='number'?p.value.toFixed(2):p.value}}</b></div>`;
          }}
        }});
        return html;
      }}
    }},
    grid: sharedGrid, xAxis: {{ ...xAxisBase }}, yAxis: {{ ...yAxisBase }},
    dataZoom: [{{ type:'inside', xAxisIndex:0, start:0, end:100 }}],
    series: [
      {{ name:'OHLC', type:'candlestick', data:ohlcData,
         itemStyle:{{ color:'#26a69a', color0:'#ef5350', borderColor:'#26a69a', borderColor0:'#ef5350' }},
         markLine: priceMark.length > 0 ? {{ symbol:['circle','arrow'], symbolSize:6,
           lineStyle:{{ width:2, type:'dashed' }}, data:priceMark, label:{{ show:false }} }} : undefined }},
      {{ name:'SMA20', type:'line', data:sma20Data, smooth:false, symbol:'none',
         lineStyle:{{ color:'#4e9af1', width:1.2 }}, itemStyle:{{ color:'#4e9af1' }} }},
      {{ name:'SMA50', type:'line', data:sma50Data, smooth:false, symbol:'none',
         lineStyle:{{ color:'#f0a500', width:1.2 }}, itemStyle:{{ color:'#f0a500' }} }},
    ],
  }});

  const chartObv = echarts.init(document.getElementById('chart-obv'), null, {{ renderer:'canvas', useDirtyRect:true }});
  chartObv.setOption({{
    backgroundColor: BG, axisPointer: sharedAxisPointer,
    tooltip: {{ ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined)
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{p.color}};margin-right:5px"></span>${{p.seriesName}}: <b>${{typeof p.value==='number'?(+p.value).toLocaleString():p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }}, xAxis: {{ ...xAxisBase, axisLabel:{{ show:false }} }}, yAxis: {{ ...yAxisBase }},
    dataZoom: [{{ type:'inside', xAxisIndex:0, start:0, end:100 }}],
    series: [
      {{ name:'OBV', type:'line', data:obvData, smooth:false, symbol:'none',
         lineStyle:{{ color:'#9b59b6', width:1.5 }}, itemStyle:{{ color:'#9b59b6' }},
         areaStyle:{{ color:'rgba(155,89,182,0.08)' }},
         markLine: obvMark.length > 0 ? {{ symbol:['circle','arrow'], symbolSize:6,
           lineStyle:{{ width:2, type:'dashed' }}, data:obvMark }} : undefined }},
    ],
  }});

  const chartRsi = echarts.init(document.getElementById('chart-rsi'), null, {{ renderer:'canvas', useDirtyRect:true }});
  chartRsi.setOption({{
    backgroundColor: BG, axisPointer: sharedAxisPointer,
    tooltip: {{ ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined && p.seriesName !== 'OB' && p.seriesName !== 'OS')
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{p.color}};margin-right:5px"></span>${{p.seriesName}}: <b>${{typeof p.value==='number'?p.value.toFixed(1):p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }}, xAxis: {{ ...xAxisBase, axisLabel:{{ show:false }} }},
    yAxis: {{ ...yAxisBase, min:0, max:100, splitNumber:2 }},
    dataZoom: [{{ type:'inside', xAxisIndex:0, start:0, end:100 }}],
    series: [
      {{ name:'RSI', type:'line', data:rsiData, smooth:false, symbol:'none',
         lineStyle:{{ color:'#1abc9c', width:1.5 }}, itemStyle:{{ color:'#1abc9c' }} }},
      {{ name:'OB', type:'line', data:dates.map(()=>70), symbol:'none',
         lineStyle:{{ color:'rgba(231,76,60,0.5)', width:1, type:'dashed' }}, tooltip:{{ show:false }} }},
      {{ name:'OS', type:'line', data:dates.map(()=>30), symbol:'none',
         lineStyle:{{ color:'rgba(46,204,113,0.5)', width:1, type:'dashed' }}, tooltip:{{ show:false }} }},
    ],
  }});

  const chartVol = echarts.init(document.getElementById('chart-vol'), null, {{ renderer:'canvas', useDirtyRect:true }});
  chartVol.setOption({{
    backgroundColor: BG, axisPointer: sharedAxisPointer,
    tooltip: {{ ...sharedTooltip,
      formatter: function(params) {{
        let html = `<div style="font-weight:600;margin-bottom:4px">${{params[0]?.axisValue}}</div>`;
        params.forEach(p => {{
          if (p.value !== null && p.value !== undefined)
            html += `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{p.color}};margin-right:5px"></span>${{p.seriesName}}: <b>${{typeof p.value==='number'?p.value.toFixed(3)+'%':p.value}}</b></div>`;
        }});
        return html;
      }}
    }},
    grid: {{ ...sharedGrid }}, xAxis: {{ ...xAxisBase }},
    yAxis: {{ ...yAxisBase, axisLabel: {{ color:AXIS_LBL, fontSize:10, formatter: v => v.toFixed(3)+'%' }}, boundaryGap: ['10%', '10%'] }},
    dataZoom: [{{ type:'inside', xAxisIndex:0, start:0, end:100 }}],
    series: [
      {{ name:'{vol_label}', type:'line', data:volData, smooth:false, symbol:'none',
         lineStyle:{{ color:'{vol_color}', width:1.5 }}, itemStyle:{{ color:'{vol_color}' }},
         areaStyle:{{ color:'{vol_color_rgba}' }} }},
    ],
  }});

  echarts.connect([chartPrice, chartObv, chartRsi, chartVol]);

  function syncZoom(source, targets) {{
    source.on('dataZoom', function() {{
      const opt = source.getOption();
      const dz  = opt.dataZoom[0];
      targets.forEach(t => t.dispatchAction({{ type:'dataZoom', start:dz.start, end:dz.end }}));
    }});
  }}
  syncZoom(chartPrice, [chartObv, chartRsi, chartVol]);
  syncZoom(chartObv,   [chartPrice, chartRsi, chartVol]);
  syncZoom(chartRsi,   [chartPrice, chartObv, chartVol]);
  syncZoom(chartVol,   [chartPrice, chartObv, chartRsi]);

  window.addEventListener('resize', function() {{
    [chartPrice, chartObv, chartRsi, chartVol].forEach(c => c.resize());
  }});
}})();
</script></body></html>
"""
    components.html(html, height=950, scrolling=False)