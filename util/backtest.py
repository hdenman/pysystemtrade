"""
util/backtest.py

Run a pysystemtrade System and write a self-contained HTML report.

Usage
-----
    from util.backtest import run_backtest
    path = run_backtest(system, name="001_flat", starting_capital=1_000_000)
    # → reports/2026-07-29.001_flat/report.html
"""

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from systems.basesystem import System

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = _PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    system: System,
    name: str,
    starting_capital: float | None = None,
) -> Path:
    """
    Run *system*, write an HTML report, return the path to report.html.

    Parameters
    ----------
    system           : built System object (all stages wired)
    name             : label used in the report title and directory name
    starting_capital : notional capital; defaults to system config value

    Output
    ------
    reports/YYYY-MM-DD.<name>/report.html
    """
    config_capital = float(system.config.notional_trading_capital)
    if starting_capital is None:
        starting_capital = config_capital

    # --- portfolio equity curve ---
    portfolio_curve = system.accounts.portfolio()
    daily_pnl = pd.Series(portfolio_curve.value_terms).dropna()
    if config_capital != starting_capital:
        daily_pnl = daily_pnl * (starting_capital / config_capital)
    equity = (starting_capital + daily_pnl.cumsum()).rename("portfolio_value")

    # --- per-instrument data ---
    instruments = system.get_instrument_list()
    prices = {
        inst: system.rawdata.get_daily_prices(inst).dropna()
        for inst in instruments
    }
    # buffered = what's actually traded (drives P&L); notional = target (for chart)
    positions = {
        inst: system.accounts.get_buffered_position(inst, roundpositions=True).dropna()
        for inst in instruments
    }
    notional_positions = {
        inst: system.portfolio.get_notional_position(inst).dropna()
        for inst in instruments
    }
    buffers = {
        inst: system.accounts.get_buffers_for_position(inst)
        for inst in instruments
    }
    rule_names = list(system.config.trading_rules.keys())
    combined_forecasts = {
        inst: system.combForecast.get_combined_forecast(inst).dropna()
        for inst in instruments
    }
    volatility = {
        inst: system.rawdata.daily_returns_volatility(inst).dropna()
        for inst in instruments
    }
    rule_forecasts = {
        inst: {
            rule: system.forecastScaleCap.get_capped_forecast(inst, rule).dropna()
            for rule in rule_names
        }
        for inst in instruments
    }

    # trades = opening position + daily changes (non-zero)
    all_trades: list[dict] = []
    for inst in instruments:
        pos = positions[inst]
        if len(pos) == 0:
            continue
        # opening position (the first non-NaN value has no predecessor in diff)
        opening = pos.iloc[0]
        if opening != 0:
            all_trades.append({
                "date": pos.index[0].strftime("%Y-%m-%d"),
                "instrument": inst,
                "direction": "Buy" if opening > 0 else "Sell",
                "qty": round(abs(opening), 4),
                "note": "open",
            })
        # subsequent daily changes
        diffs = pos.diff().dropna()
        diffs = diffs[diffs != 0]
        for dt, qty in diffs.items():
            all_trades.append({
                "date": dt.strftime("%Y-%m-%d"),
                "instrument": inst,
                "direction": "Buy" if qty > 0 else "Sell",
                "qty": round(abs(qty), 4),
                "note": "",
            })
    all_trades.sort(key=lambda r: r["date"], reverse=False)

    # weekly equity summary
    weekly_equity = equity.resample("W").last().dropna()
    weekly_pnl = weekly_equity.diff()
    weekly_pct = (weekly_pnl / weekly_equity.shift(1) * 100)

    weekly_rows = []
    for dt in weekly_equity.index:
        val = weekly_equity[dt]
        pnl = weekly_pnl[dt]
        pct = weekly_pct[dt]
        weekly_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": _fmt2(val),
            "pnl": _fmt2(pnl) if not _nan(pnl) else None,
            "pct": _fmt2(pct) if not _nan(pct) else None,
        })

    # weekly holdings: position, market value, cash, total
    holdings_dates = (
        positions[instruments[0]].resample("W").last().dropna().index
        if instruments else pd.DatetimeIndex([])
    )

    # pointsize and base-currency FX rate per instrument
    base_ccy = system.config.get_element_or_default("base_currency", "USD")
    pointsizes = {inst: system.data.get_value_of_block_price_move(inst) for inst in instruments}
    weekly_fx = {
        inst: system.data.get_fx_for_instrument(inst, base_ccy).resample("W").last()
        for inst in instruments
    }

    holdings_pos:   dict[str, list] = {}
    holdings_val:   dict[str, list] = {}
    for inst in instruments:
        weekly_pos   = positions[inst].resample("W").last()
        weekly_price = prices[inst].resample("W").last()
        ps = pointsizes[inst]
        pos_list, val_list = [], []
        for dt in holdings_dates:
            p = weekly_pos.get(dt, float("nan"))
            pr = weekly_price.get(dt, float("nan"))
            fx = weekly_fx[inst].get(dt, 1.0)
            pos_list.append(_fmt4(p))
            val_list.append(_fmt2(p * pr * ps * fx) if not (_nan(p) or _nan(pr)) else None)
        holdings_pos[inst] = pos_list
        holdings_val[inst] = val_list

    # cash = total equity − Σ instrument values; total = equity at week-end
    weekly_equity_w = weekly_equity.reindex(holdings_dates)
    holdings_total, holdings_cash = [], []
    for ri, dt in enumerate(holdings_dates):
        tot = weekly_equity_w.get(dt, float("nan"))
        if _nan(tot):
            holdings_total.append(None)
            holdings_cash.append(None)
        else:
            inst_sum = sum(holdings_val[inst][ri] or 0.0 for inst in instruments)
            holdings_total.append(_fmt2(tot))
            holdings_cash.append(_fmt2(tot - inst_sum))

    holdings_date_strs = [dt.strftime("%Y-%m-%d") for dt in holdings_dates]

    # --- assemble data blob ---
    data = {
        "meta": {
            "name": name,
            "run_date": date.today().isoformat(),
            "start_date": equity.index[0].strftime("%Y-%m-%d"),
            "end_date": equity.index[-1].strftime("%Y-%m-%d"),
            "n_days": len(equity),
            "starting_capital": round(starting_capital, 2),
            "final_value": round(float(equity.iloc[-1]), 2),
            "total_return_pct": round(
                (float(equity.iloc[-1]) / starting_capital - 1) * 100, 2
            ),
            "forecast_cap": float(system.config.get_element_or_default("forecast_cap", 20.0)),
        },
        "instruments": instruments,
        "equity": _series_to_chartjs(equity),
        "prices":    {inst: _series_to_chartjs(prices[inst])             for inst in instruments},
        "positions": {inst: _series_to_chartjs(positions[inst])          for inst in instruments},
        "notional":  {inst: _series_to_chartjs(notional_positions[inst]) for inst in instruments},
        "buf_top":   {inst: _series_to_chartjs(buffers[inst]["top_pos"].dropna()) for inst in instruments},
        "buf_bot":        {inst: _series_to_chartjs(buffers[inst]["bot_pos"].dropna()) for inst in instruments},
        "rule_names":     rule_names,
        "combined_fc":    {inst: _series_to_chartjs(combined_forecasts[inst]) for inst in instruments},
        "volatility":     {inst: _series_to_chartjs(volatility[inst]) for inst in instruments},
        "rule_forecasts": {
            inst: {rule: _series_to_chartjs(rule_forecasts[inst][rule]) for rule in rule_names}
            for inst in instruments
        },
        "weekly": weekly_rows,
        "holdings": {
            "dates":  holdings_date_strs,
            "pos":    holdings_pos,
            "value":  holdings_val,
            "cash":   holdings_cash,
            "total":  holdings_total,
        },
    }

    # --- write report ---
    today = date.today().isoformat()
    report_dir = REPORTS_DIR / f"{today}.{name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "report.html"
    html_path.write_text(_render_html(data), encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _fmt2(v) -> float:
    return round(float(v), 2)


def _fmt4(v) -> float | None:
    if _nan(v):
        return None
    return round(float(v), 4)


def _series_to_chartjs(series: pd.Series) -> list[dict]:
    """Convert a DatetimeIndex Series to Chart.js {x, y} points."""
    out = []
    for idx, val in series.items():
        y = None if _nan(val) else round(float(val), 4)
        out.append({"x": idx.strftime("%Y-%m-%d"), "y": y})
    return out


# ---------------------------------------------------------------------------
# HTML rendering — data injected as a single JSON blob to avoid brace-escaping
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest — __NAME__ — __RUN_DATE__</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<style>
  body     { background:#f8fafc; }
  .header  { background:#fff; border-bottom:1px solid #e2e8f0;
             padding:1rem 1.5rem; margin-bottom:1.5rem; }
  .cards   { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.25rem; }
  .card    { min-width:160px; }
  .cl      { font-size:.7rem; text-transform:uppercase;
             letter-spacing:.06em; color:#64748b; }
  .cv      { font-size:1.4rem; font-weight:600; }
  .pos     { color:#16a34a; }
  .neg     { color:#dc2626; }
  .chart   { width:100%; height:380px; }
  .chart-sm{ width:100%; height:220px; }
  .tscroll { max-height:560px; overflow-y:auto; }
  #sticky-bar {
    position: sticky; top: 0; z-index: 200;
    background: #f8fafc;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    padding-bottom: 0;
  }
  th       { position:sticky; top:0; background:#f1f5f9;
             z-index:1; white-space:nowrap; }
  td       { white-space:nowrap; }
  .nav-tabs .nav-link       { color:#475569; }
  .nav-tabs .nav-link.active{ font-weight:600; color:#0f172a; }
</style>
</head>
<body>

<div id="sticky-bar">
  <div class="header">
    <h5 class="mb-0">Backtest — <strong id="h-name"></strong></h5>
    <small class="text-muted" id="h-sub"></small>
  </div>
  <div class="container-fluid px-4 pb-0">
    <div class="cards" id="stat-cards"></div>
    <ul class="nav nav-tabs" id="mainTabs" role="tablist"></ul>
  </div>
</div>
<div class="container-fluid px-4 pt-3">
  <div class="tab-content" id="mainContent"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.32.0/plotly.min.js"></script>

<script>
const D = __DATA__;

// ── header ────────────────────────────────────────────────────────────────
document.getElementById('h-name').textContent = D.meta.name;
document.getElementById('h-sub').textContent =
  `Run ${D.meta.run_date} · ${D.meta.n_days} trading days · ${D.meta.start_date} → ${D.meta.end_date}`;

// ── stat cards ────────────────────────────────────────────────────────────
const retCls = D.meta.total_return_pct >= 0 ? 'pos' : 'neg';
[
  ['Starting capital',  fmt(D.meta.starting_capital, 0), ''],
  ['Final value',       fmt(D.meta.final_value, 0),      ''],
  ['Total return',      fmtPct(D.meta.total_return_pct), retCls],
].forEach(([lbl, val, cls]) => {
  document.getElementById('stat-cards').insertAdjacentHTML('beforeend',
    `<div class="card p-3"><div class="cl">${lbl}</div>` +
    `<div class="cv ${cls}">${val}</div></div>`);
});

// ── tab helpers ───────────────────────────────────────────────────────────
const tabs  = document.getElementById('mainTabs');
const panes = document.getElementById('mainContent');

function addTab(id, label, contentHtml, active) {
  tabs.insertAdjacentHTML('beforeend',
    `<li class="nav-item" role="presentation">
       <button class="nav-link ${active ? 'active' : ''}"
         id="tab-${id}" data-bs-toggle="tab"
         data-bs-target="#pane-${id}" type="button" role="tab">
         ${label}
       </button>
     </li>`);
  panes.insertAdjacentHTML('beforeend',
    `<div class="tab-pane fade ${active ? 'show active' : ''}"
       id="pane-${id}" role="tabpanel">
       ${contentHtml}
     </div>`);
}

// Resize Plotly charts when a tab becomes visible (they render at 0×0 if hidden).
tabs.addEventListener('shown.bs.tab', e => {
  const pane = document.querySelector(e.target.getAttribute('data-bs-target'));
  if (pane) pane.querySelectorAll('.chart, .chart-sm').forEach(el => Plotly.Plots.resize(el));
});

// ── Plotly helpers ────────────────────────────────────────────────────────
const PLOTLY_CFG = { responsive: true, scrollZoom: true, displayModeBar: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','resetScale2d'] };

const PLOTLY_BASE_LAYOUT = {
  margin:    { t:20, r:20, b:50, l:90 },
  hovermode: 'x unified',
  plot_bgcolor:  '#fff',
  paper_bgcolor: '#f8fafc',
  xaxis: { type:'date', showgrid:true, gridcolor:'#e2e8f0' },
  yaxis: { showgrid:true, gridcolor:'#e2e8f0' },
};

function pts2xy(pts) {
  const x = [], y = [];
  pts.forEach(p => { if (p.y !== null) { x.push(p.x); y.push(p.y); } });
  return { x, y };
}

function plotLine(divId, traces, layout) {
  Plotly.newPlot(divId, traces,
    Object.assign({}, PLOTLY_BASE_LAYOUT, layout),
    PLOTLY_CFG);
}

// ── 1. Portfolio tab ──────────────────────────────────────────────────────
addTab('portfolio', 'Portfolio',
  `<div class="p-3"><div id="chart-portfolio" class="chart"></div></div>`,
  true);

{
  const {x, y} = pts2xy(D.equity);
  plotLine('chart-portfolio', [{
    x, y, type:'scatter', mode:'lines', name:'Portfolio value',
    line: { color:'#2563eb', width:1.5 },
    hovertemplate:'%{x}<br><b>%{y:,.0f}</b><extra></extra>',
  }], {
    yaxis: { tickformat:',.0f', title:{ text:'Value' }, showgrid:true, gridcolor:'#e2e8f0' },
  });
}

// ── 2. Per-instrument tabs ────────────────────────────────────────────────
D.instruments.forEach(inst => {
  const cap = D.meta.forecast_cap;

  // Build rule-forecast grid HTML
  const ruleDivs = D.rule_names.map(rule =>
    `<div class="col-md-6 col-xl-4">
       <p class="small text-muted mb-1 mt-2">${rule}</p>
       <div id="chart-rule-${inst}-${rule}" class="chart-sm"></div>
     </div>`
  ).join('');

  addTab(`inst-${inst}`, inst,
    `<div class="p-3">
       <div id="chart-price-${inst}" class="chart"></div>
       <div id="chart-pos-${inst}"   class="chart mt-3"></div>
       <h6 class="mt-4 mb-0 fw-semibold text-secondary">Combined forecast</h6>
       <div id="chart-fc-${inst}"  class="chart-sm"></div>
       <h6 class="mt-3 mb-0 fw-semibold text-secondary">Daily price volatility</h6>
       <div id="chart-vol-${inst}" class="chart-sm"></div>
       <h6 class="mt-3 mb-1 fw-semibold text-secondary">Rule forecasts</h6>
       <div class="row g-0">${ruleDivs}</div>
     </div>`,
    false);

  // ── price chart ──
  const {x:px, y:py} = pts2xy(D.prices[inst]);
  plotLine(`chart-price-${inst}`, [{
    x:px, y:py, type:'scatter', mode:'lines', name:'Price',
    line: { color:'#7c3aed', width:1.2 },
    hovertemplate:'%{x}<br><b>%{y:.4f}</b><extra></extra>',
  }], { yaxis: { title:{ text:'Price (back-adj)' }, showgrid:true, gridcolor:'#e2e8f0' } });

  // ── position chart ──
  const {x:qx, y:qy} = pts2xy(D.positions[inst]);
  const {x:nx, y:ny} = pts2xy(D.notional[inst]);
  const {x:tx, y:ty} = pts2xy(D.buf_top[inst]);
  const {x:bx, y:by} = pts2xy(D.buf_bot[inst]);
  plotLine(`chart-pos-${inst}`, [
    { x:tx, y:ty, type:'scatter', mode:'lines', name:'Buffer top',
      line:{ color:'#ef4444', width:1, dash:'dash' },
      hovertemplate:'%{x}<br>buf_top <b>%{y:.2f}</b><extra></extra>' },
    { x:bx, y:by, type:'scatter', mode:'lines', name:'Buffer bot',
      fill:'tonexty', fillcolor:'rgba(239,68,68,0.06)',
      line:{ color:'#ef4444', width:1, dash:'dash' },
      hovertemplate:'%{x}<br>buf_bot <b>%{y:.2f}</b><extra></extra>' },
    { x:nx, y:ny, type:'scatter', mode:'lines', name:'Target (notional)',
      line:{ color:'#f59e0b', width:1.5, dash:'dot' },
      hovertemplate:'%{x}<br>target <b>%{y:.2f}</b><extra></extra>' },
    { x:qx, y:qy, type:'scatter', mode:'lines', name:'Actual (buffered)',
      line:{ color:'#0891b2', width:2 },
      hovertemplate:'%{x}<br>actual <b>%{y:.2f}</b><extra></extra>' },
  ], { yaxis: { title:{ text:'Contracts' }, showgrid:true, gridcolor:'#e2e8f0' },
       showlegend: true,
       shapes:[{ type:'line', x0:qx[0], x1:qx[qx.length-1], y0:0, y1:0,
                 line:{ color:'#94a3b8', width:1, dash:'dot' } }] });

  // ── combined forecast ──
  const fcShapes = [
    { type:'line', x0:px[0], x1:px[px.length-1], y0: cap, y1: cap,
      line:{ color:'#ef4444', width:1, dash:'dash' } },
    { type:'line', x0:px[0], x1:px[px.length-1], y0:-cap, y1:-cap,
      line:{ color:'#ef4444', width:1, dash:'dash' } },
    { type:'line', x0:px[0], x1:px[px.length-1], y0:0, y1:0,
      line:{ color:'#94a3b8', width:1, dash:'dot' } },
  ];
  const {x:fcx, y:fcy} = pts2xy(D.combined_fc[inst]);
  plotLine(`chart-fc-${inst}`, [{
    x:fcx, y:fcy, type:'scatter', mode:'lines', name:'Combined forecast',
    line:{ color:'#16a34a', width:1.5 },
    hovertemplate:'%{x}<br><b>%{y:.2f}</b><extra></extra>',
  }], { yaxis:{ title:{ text:'Forecast' }, showgrid:true, gridcolor:'#e2e8f0' },
        shapes: fcShapes });

  // ── volatility ──
  const {x:vx, y:vy} = pts2xy(D.volatility[inst]);
  plotLine(`chart-vol-${inst}`, [{
    x:vx, y:vy, type:'scatter', mode:'lines', name:'Daily vol',
    fill:'tozeroy', fillcolor:'rgba(100,116,139,0.12)',
    line:{ color:'#64748b', width:1.2 },
    hovertemplate:'%{x}<br><b>%{y:.4f}</b><extra></extra>',
  }], { yaxis:{ title:{ text:'Price vol' }, showgrid:true, gridcolor:'#e2e8f0' } });

  // ── per-rule forecasts ──
  D.rule_names.forEach(rule => {
    const {x:rx, y:ry} = pts2xy(D.rule_forecasts[inst][rule]);
    plotLine(`chart-rule-${inst}-${rule}`, [{
      x:rx, y:ry, type:'scatter', mode:'lines', name:rule,
      line:{ color:'#2563eb', width:1 },
      hovertemplate:'%{x}<br><b>%{y:.2f}</b><extra></extra>',
    }], { yaxis:{ showgrid:true, gridcolor:'#e2e8f0' },
          shapes: fcShapes,
          margin:{ t:10, r:10, b:40, l:60 } });
  });
});

// ── 3. Weekly values tab ─────────────────────────────────────────────────
(function() {
  let rows = '';
  D.weekly.forEach(r => {
    const cls = r.pnl === null ? '' : (r.pnl >= 0 ? 'pos' : 'neg');
    rows +=
      `<tr>
         <td>${r.date}</td>
         <td class="text-end">${r.value === null ? '—' : fmt(r.value,2)}</td>
         <td class="text-end ${cls}">${r.pnl === null ? '—' : fmtSigned(r.pnl,2)}</td>
         <td class="text-end ${cls}">${r.pct  === null ? '—' : fmtPct(r.pct)}</td>
       </tr>`;
  });
  addTab('weekly', 'Weekly values',
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead><tr>
           <th>Week ending</th><th class="text-end">Portfolio value</th>
           <th class="text-end">Weekly P&amp;L</th><th class="text-end">Weekly return</th>
         </tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`, false);
})();

// ── 4. Trades tab ─────────────────────────────────────────────────────────
(function() {
  let rows = '';
  D.trades.forEach(t => {
    const cls = t.direction === 'Buy' ? 'pos' : 'neg';
    const badge = t.note === 'open'
      ? '<span class="badge bg-secondary ms-1" style="font-size:.65rem">open</span>' : '';
    rows +=
      `<tr>
         <td>${t.date}</td><td>${t.instrument}</td>
         <td class="${cls}">${t.direction}${badge}</td>
         <td class="text-end">${t.qty.toLocaleString(undefined,
             {minimumFractionDigits:2, maximumFractionDigits:4})}</td>
       </tr>`;
  });
  addTab('trades',
    `Trades <span class="badge bg-secondary ms-1">${D.trades.length}</span>`,
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead><tr>
           <th>Date</th><th>Instrument</th>
           <th>Direction</th><th class="text-end">Quantity</th>
         </tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`, false);
})();

// ── 5. Holdings tab ───────────────────────────────────────────────────────
(function() {
  const h = D.holdings;
  const insts = D.instruments;

  // Two-level header: instrument name spans pos+value cols, then Cash, Total
  const hdr1 = '<th rowspan="2">Week ending</th>' +
    insts.map(i => `<th colspan="2" class="text-center border-start">${i}</th>`).join('') +
    '<th rowspan="2" class="text-end border-start">Cash</th>' +
    '<th rowspan="2" class="text-end border-start fw-bold">Total</th>';
  const hdr2 = insts.map(() =>
    '<th class="text-end border-start">Position</th><th class="text-end">Value</th>'
  ).join('');

  let rows = '';
  h.dates.forEach((dt, ri) => {
    const instCells = insts.map(inst => {
      const pos = h.pos[inst][ri];
      const val = h.value[inst][ri];
      return `<td class="text-end border-start">${pos === null ? '—' :
          pos.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>` +
        `<td class="text-end">${val === null ? '—' : fmt(val, 2)}</td>`;
    }).join('');
    const cash  = h.cash[ri];
    const total = h.total[ri];
    rows += `<tr>
      <td>${dt}</td>${instCells}
      <td class="text-end border-start">${cash  === null ? '—' : fmt(cash,  2)}</td>
      <td class="text-end border-start fw-bold">${total === null ? '—' : fmt(total, 2)}</td>
    </tr>`;
  });

  addTab('holdings', 'Holdings',
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead>
           <tr>${hdr1}</tr>
           <tr>${hdr2}</tr>
         </thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`, false);
})();

// ── utilities ─────────────────────────────────────────────────────────────
function fmt(v, dp) {
  if (v == null) return '—';
  return Number(v).toLocaleString(undefined,
    {minimumFractionDigits:dp, maximumFractionDigits:dp});
}
function fmtSigned(v, dp) {
  return (v >= 0 ? '+' : '−') + fmt(Math.abs(v), dp);
}
function fmtPct(v) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) + '%';
}
</script>
</body>
</html>
"""


def _render_html(data: dict) -> str:
    json_blob = json.dumps(data, allow_nan=False)
    return (
        _HTML_TEMPLATE
        .replace("__NAME__",     data["meta"]["name"])
        .replace("__RUN_DATE__", data["meta"]["run_date"])
        .replace("__DATA__",     json_blob)
    )
