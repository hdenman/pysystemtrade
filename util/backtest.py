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
    positions = {
        inst: system.portfolio.get_notional_position(inst).dropna()
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

    # weekly holdings: one row per week-end, one column per instrument
    # column value = position (contracts)
    holdings_dates = (
        positions[instruments[0]].resample("W").last().dropna().index
        if instruments else pd.DatetimeIndex([])
    )
    holdings: dict[str, list] = {}
    for inst in instruments:
        weekly_pos = positions[inst].resample("W").last().dropna()
        holdings[inst] = [
            _fmt4(weekly_pos.get(dt, float("nan")))
            for dt in holdings_dates
        ]
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
        },
        "instruments": instruments,
        "equity": _series_to_chartjs(equity),
        "prices": {inst: _series_to_chartjs(prices[inst]) for inst in instruments},
        "positions": {inst: _series_to_chartjs(positions[inst]) for inst in instruments},
        "trades": all_trades,
        "weekly": weekly_rows,
        "holdings": {
            "dates": holdings_date_strs,
            "instruments": holdings,
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
  canvas   { max-height:380px; }
  .tscroll { max-height:540px; overflow-y:auto; }
  th       { position:sticky; top:0; background:#f1f5f9;
             z-index:1; white-space:nowrap; }
  td       { white-space:nowrap; }
  .nav-tabs .nav-link { color:#475569; }
  .nav-tabs .nav-link.active { font-weight:600; color:#0f172a; }
</style>
</head>
<body>

<div class="header">
  <h5 class="mb-0">Backtest — <strong id="h-name"></strong></h5>
  <small class="text-muted" id="h-sub"></small>
</div>

<div class="container-fluid px-4">

  <!-- stat cards -->
  <div class="cards" id="stat-cards"></div>

  <!-- tabs -->
  <ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist"></ul>
  <div class="tab-content" id="mainContent"></div>

</div><!-- /container -->

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

<script>
const D = __DATA__;

// ── header ───────────────────────────────────────────────────────────────
document.getElementById('h-name').textContent = D.meta.name;
document.getElementById('h-sub').textContent =
  `Run ${D.meta.run_date} · ${D.meta.n_days} trading days · ${D.meta.start_date} → ${D.meta.end_date}`;

// ── stat cards ────────────────────────────────────────────────────────────
const retCls = D.meta.total_return_pct >= 0 ? 'pos' : 'neg';
const cards = [
  ['Starting capital',  fmt(D.meta.starting_capital, 0), ''],
  ['Final value',       fmt(D.meta.final_value, 0),      ''],
  ['Total return',      fmtPct(D.meta.total_return_pct), retCls],
];
const cardEl = document.getElementById('stat-cards');
cards.forEach(([lbl, val, cls]) => {
  cardEl.innerHTML +=
    `<div class="card p-3"><div class="cl">${lbl}</div>` +
    `<div class="cv ${cls}">${val}</div></div>`;
});

// ── tabs ──────────────────────────────────────────────────────────────────
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

// ── chart helpers ─────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true,
  animation: false,
  plugins: { legend: { display: false } },
  elements: { point: { radius: 0 } },
};

function timeAxis(unit) {
  return { type: 'time', time: { unit }, ticks: { maxTicksLimit: 12 } };
}

function currencyAxis(label) {
  return {
    title: { display: !!label, text: label || '' },
    ticks: { callback: v => fmt(v, 0) },
  };
}

function makeLineChart(canvasId, datasets, yAxes, xUnit) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      ...CHART_DEFAULTS,
      scales: { x: timeAxis(xUnit || 'month'), ...yAxes },
    },
  });
}

// ── 1. Portfolio tab ──────────────────────────────────────────────────────
addTab('portfolio', 'Portfolio',
  `<div class="p-3"><canvas id="chart-portfolio"></canvas></div>`,
  true);

makeLineChart('chart-portfolio',
  [{ label: 'Portfolio value', data: D.equity,
     borderColor: '#2563eb', borderWidth: 1.5, fill: false }],
  { y: currencyAxis('Value') });

// ── 2. Per-instrument tabs ────────────────────────────────────────────────
D.instruments.forEach(inst => {
  addTab(`inst-${inst}`, inst,
    `<div class="p-3">
       <canvas id="chart-price-${inst}"></canvas>
       <div class="mt-3"><canvas id="chart-pos-${inst}"></canvas></div>
     </div>`,
    false);

  makeLineChart(`chart-price-${inst}`,
    [{ label: 'Price (back-adj)', data: D.prices[inst],
       borderColor: '#7c3aed', borderWidth: 1.2, fill: false }],
    { y: { title: { display: true, text: 'Price' } } });

  makeLineChart(`chart-pos-${inst}`,
    [{ label: 'Notional position', data: D.positions[inst],
       borderColor: '#0891b2', borderWidth: 1.2, fill: true,
       backgroundColor: 'rgba(8,145,178,0.08)' }],
    { y: { title: { display: true, text: 'Contracts' } } });
});

// ── 3. Weekly values tab ─────────────────────────────────────────────────
(function() {
  let rows = '';
  D.weekly.forEach(r => {
    const pnlCls = r.pnl === null ? '' : (r.pnl >= 0 ? 'pos' : 'neg');
    rows +=
      `<tr>
         <td>${r.date}</td>
         <td class="text-end">${r.value === null ? '—' : fmt(r.value, 2)}</td>
         <td class="text-end ${pnlCls}">${r.pnl === null ? '—' : fmtSigned(r.pnl, 2)}</td>
         <td class="text-end ${pnlCls}">${r.pct  === null ? '—' : fmtPct(r.pct)}</td>
       </tr>`;
  });
  addTab('weekly', 'Weekly values',
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead><tr>
           <th>Week ending</th><th class="text-end">Portfolio value</th>
           <th class="text-end">Weekly P&amp;L</th>
           <th class="text-end">Weekly return</th>
         </tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`,
    false);
})();

// ── 4. Trades tab ─────────────────────────────────────────────────────────
(function() {
  let rows = '';
  D.trades.forEach(t => {
    const cls = t.direction === 'Buy' ? 'pos' : 'neg';
    const noteBadge = t.note === 'open'
      ? '<span class="badge bg-secondary ms-1" style="font-size:.65rem">open</span>'
      : '';
    rows +=
      `<tr>
         <td>${t.date}</td>
         <td>${t.instrument}</td>
         <td class="${cls}">${t.direction}${noteBadge}</td>
         <td class="text-end">${t.qty.toLocaleString(undefined,
             {minimumFractionDigits:2, maximumFractionDigits:4})}</td>
       </tr>`;
  });
  addTab('trades', `Trades <span class="badge bg-secondary ms-1">${D.trades.length}</span>`,
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead><tr>
           <th>Date</th><th>Instrument</th>
           <th>Direction</th><th class="text-end">Quantity</th>
         </tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`,
    false);
})();

// ── 5. Holdings tab ───────────────────────────────────────────────────────
(function() {
  const h = D.holdings;
  const instCols = h.instruments ? Object.keys(h.instruments) : [];
  let hdr = '<th>Week ending</th>' +
    instCols.map(i => `<th class="text-end">${i}</th>`).join('');
  let rows = '';
  h.dates.forEach((dt, ri) => {
    const cells = instCols.map(inst => {
      const v = h.instruments[inst][ri];
      return `<td class="text-end">${v === null ? '—' :
        v.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</td>`;
    }).join('');
    rows += `<tr><td>${dt}</td>${cells}</tr>`;
  });
  addTab('holdings', 'Holdings',
    `<div class="p-3 tscroll">
       <table class="table table-sm table-hover mb-0">
         <thead><tr>${hdr}</tr></thead>
         <tbody>${rows}</tbody>
       </table>
     </div>`,
    false);
})();

// ── utilities ─────────────────────────────────────────────────────────────
function fmt(v, dp) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString(undefined,
    {minimumFractionDigits: dp, maximumFractionDigits: dp});
}
function fmtSigned(v, dp) {
  const s = fmt(Math.abs(v), dp);
  return (v >= 0 ? '+' : '−') + s;
}
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  const s = Math.abs(v).toFixed(2) + '%';
  return (v >= 0 ? '+' : '−') + s;
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
