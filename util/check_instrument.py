"""
Instrument health check.

For a given instrument verifies that all data required for live trading is
present, current, and free of business-day gaps:

  * Adjusted prices
  * Multiple prices  (also checks current price/carry/forward contracts make
    sense given today's date and the instrument's roll parameters)
  * Roll parameters  (config) and roll calendar (CSV)
  * FX conversion prices for the instrument's settlement currency

A "gap" is any business day (Monday–Friday) in the span [first observation,
today] that has no observation, excluding the hard-coded market-closed days
below.

Usage::

    python -m util.check_instrument EUR_micro
    python -m util.check_instrument                # interactive prompt
    python -m util.check_instrument --no-color
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from syscore.constants import arg_not_supplied
from syscore.exceptions import missingData
from sysdata.data_blob import dataBlob
from sysobjects.contract_dates_and_expiries import contractDate
from sysobjects.rolls import contractDateWithRollParameters
from sysproduction.data.contracts import dataContracts
from sysproduction.data.currency_data import dataCurrency
from sysproduction.data.instruments import diagInstruments
from sysproduction.data.prices import diagPrices

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ROLL_CALENDAR_DIR = _PROJECT_ROOT / "data" / "futures" / "roll_calendars_csv"

# ---------------------------------------------------------------------------
# Status constants & colour helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_RESET = "\033[0m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"

_COLOR_MAP = {PASS: _GREEN, WARN: _YELLOW, FAIL: _RED}
_USE_COLOR = True  # overridden by --no-color


def _colored(text: str, color: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def _status_str(status: str) -> str:
    return _colored(f"[{status}]", _COLOR_MAP.get(status, ""))



# ---------------------------------------------------------------------------
# Market-closed days
# ---------------------------------------------------------------------------

# Hard-coded known exchange closures which are acceptable missing observations.
# The base list covers the historical EUROSTX/STOXX futures data shipped/used
# here: New Year's Day, Good Friday, Easter Monday, May Day, Christmas Eve,
# Christmas Day, Boxing Day, New Year's Eve, plus two observed Whit Monday
# closures.  Additional dates below are known acceptable gaps observed in other
# production price series.
MARKET_CLOSED_DAYS = (
    "1999-01-01", "1999-04-02", "1999-04-05", "1999-12-24", "1999-12-31", "2000-04-21",
    "2000-04-24", "2000-05-01", "2000-12-25", "2000-12-26", "2001-01-01", "2001-04-13",
    "2001-04-16", "2001-05-01", "2001-12-24", "2001-12-25", "2001-12-26", "2001-12-31",
    "2002-01-01", "2002-03-29", "2002-04-01", "2002-05-01", "2002-12-24", "2002-12-25",
    "2002-12-26", "2002-12-31", "2003-01-01", "2003-04-18", "2003-04-21", "2003-05-01",
    "2003-12-24", "2003-12-25", "2003-12-26", "2003-12-31", "2004-01-01", "2004-04-09",
    "2004-04-12", "2004-12-24", "2004-12-31", "2005-03-25", "2005-03-28", "2005-12-26",
    "2006-04-14", "2006-04-17", "2006-05-01", "2006-12-25", "2006-12-26", "2007-01-01",
    "2007-04-06", "2007-04-09", "2007-05-01", "2007-05-28", "2007-12-24", "2007-12-25",
    "2007-12-26", "2007-12-31", "2008-01-01", "2008-03-21", "2008-03-24", "2008-05-01",
    "2008-12-24", "2008-12-25", "2008-12-26", "2008-12-31", "2009-01-01", "2009-04-10",
    "2009-04-13", "2009-05-01", "2009-12-24", "2009-12-25", "2009-12-31", "2010-01-01",
    "2010-04-02", "2010-04-05", "2010-12-24", "2010-12-31", "2011-04-22", "2011-04-25",
    "2011-12-26", "2012-04-06", "2012-04-09", "2012-05-01", "2012-12-24", "2012-12-25",
    "2012-12-26", "2012-12-31", "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-01",
    "2013-12-24", "2013-12-25", "2013-12-26", "2013-12-31", "2014-01-01", "2014-04-18",
    "2014-04-21", "2014-05-01", "2014-12-24", "2014-12-25", "2014-12-26", "2014-12-31",
    "2015-01-01", "2015-04-03", "2015-04-06", "2015-05-01", "2015-05-25", "2015-12-24",
    "2015-12-25", "2015-12-31", "2016-01-01", "2016-03-25", "2016-03-28", "2016-12-26",
    "2017-04-14", "2017-04-17", "2017-05-01", "2017-12-25", "2017-12-26", "2018-01-01",
    "2018-03-30", "2018-04-02", "2018-05-01", "2018-12-24", "2018-12-25", "2018-12-26",
    "2018-12-31", "2019-01-01", "2019-04-19", "2019-04-22", "2019-05-01", "2019-12-24",
    "2019-12-25", "2019-12-26", "2019-12-31", "2020-01-01", "2020-04-10", "2020-04-13",
    "2020-05-01", "2020-12-24", "2020-12-25", "2020-12-31", "2021-01-01", "2021-04-02",
    "2021-04-05", "2021-12-24", "2021-12-31", "2022-04-15", "2022-04-18", "2022-12-26",
    "2023-04-07", "2023-04-10", "2023-05-01", "2023-12-25", "2023-12-26", "2024-01-01",
    "2024-03-29", "2024-04-01", "2024-05-01", "2024-12-24", "2024-12-25", "2024-12-26",
    "2024-12-31", "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-12-24",
    "2025-12-25", "2025-12-26", "2025-12-31", "2026-01-01", "2026-04-03", "2026-04-06",
    "2026-05-01", "2026-12-24", "2026-12-25", "2026-12-31",
)

ADDITIONAL_MARKET_CLOSED_DAYS = (
    # New Year's Day observed when Jan 1 falls on a Sunday.
    "2006-01-02",
    "2012-01-02",
    "2017-01-02",
    "2023-01-02",
    # Known two-day November closure/gap in 2017 price histories.
    "2017-11-16",
    "2017-11-17",
)
_MARKET_CLOSED_DAY_INDEX = pd.DatetimeIndex(
    MARKET_CLOSED_DAYS + ADDITIONAL_MARKET_CLOSED_DAYS
)

# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def _business_day_gaps(
    series: pd.Series, end: Optional[pd.Timestamp] = None
) -> Tuple[int, List[Tuple[pd.Timestamp, pd.Timestamp]], int]:
    """Return (missing_bdays, gap_ranges, ignored_market_closed_bdays).

    Gaps are contiguous runs of missing business days after removing acceptable
    market-closed dates.  ``end`` defaults to today so we also flag if the
    series has not been updated recently.
    """
    clean = series.dropna()
    if clean.empty:
        return 0, [], 0

    # Cast to DatetimeIndex so Pyright sees .normalize() / floor-day ops.
    dti = pd.DatetimeIndex(clean.index)
    start: pd.Timestamp = dti.normalize().min()  # type: ignore[assignment]
    end = pd.Timestamp(end or datetime.date.today())

    all_bdays: pd.DatetimeIndex = pd.bdate_range(start, end)
    observed: pd.DatetimeIndex = dti.normalize().unique()
    missing_including_market_closed: pd.DatetimeIndex = all_bdays.difference(
        observed
    )
    missing: pd.DatetimeIndex = missing_including_market_closed.difference(
        _MARKET_CLOSED_DAY_INDEX
    )
    ignored_market_closed = len(missing_including_market_closed) - len(missing)

    if missing.empty:
        return 0, [], ignored_market_closed

    # Group into contiguous runs
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start: pd.Timestamp = missing[0]  # type: ignore[assignment]
    prev: pd.Timestamp = missing[0]  # type: ignore[assignment]
    for _d in missing[1:]:
        d: pd.Timestamp = _d  # type: ignore[assignment]
        # Jump of more than one business day → new gap
        if (d - prev).days > 3:
            gaps.append((run_start, prev))
            run_start = d
        prev = d
    gaps.append((run_start, prev))

    return len(missing), gaps, ignored_market_closed


# ---------------------------------------------------------------------------
# Individual check results
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str          # PASS / WARN / FAIL
    detail: str = ""
    notes: List[str] = field(default_factory=list)

    def print(self, indent: int = 2) -> None:
        pad = " " * indent
        print(f"{pad}{_status_str(self.status)}  {_BOLD if _USE_COLOR else ''}{self.name}{_RESET if _USE_COLOR else ''}")
        if self.detail:
            print(f"{pad}       {self.detail}")
        for note in self.notes:
            print(f"{pad}       {_colored('!', _YELLOW)} {note}")


# ---------------------------------------------------------------------------
# Per-series gap check helper
# ---------------------------------------------------------------------------

def _gap_check(
    series: pd.Series,
    label: str,
    today: pd.Timestamp,
    stale_warn_days: int = 5,
    allow_missing_today: bool = False,
) -> CheckResult:
    """Standard gap + staleness check for any daily price series."""
    if series is None or series.dropna().empty:
        return CheckResult(label, FAIL, "no data found")

    clean = series.dropna()
    dti = pd.DatetimeIndex(clean.index)
    first: pd.Timestamp = dti.min()  # type: ignore[assignment]
    last: pd.Timestamp = dti.max()  # type: ignore[assignment]
    n = len(clean)

    staleness = (today - last).days
    stale_note = (
        f"last observation {last.date()} is {staleness} calendar days ago"
        if staleness > stale_warn_days
        else None
    )

    gap_end = today - pd.Timedelta(days=1) if allow_missing_today else today
    missing_count, gaps, ignored_market_closed = _business_day_gaps(clean, end=gap_end)

    status = PASS
    notes: List[str] = []

    if stale_note:
        status = WARN
        notes.append(stale_note)

    if missing_count > 0:
        status = FAIL
        notes.append(f"{missing_count} missing business day(s)")
        # Show up to 5 gap examples (most recent first)
        recent_gaps = list(reversed(gaps))
        for gs, ge in recent_gaps[:5]:
            if gs == ge:
                notes.append(f"  gap: {gs.date()}")
            else:
                notes.append(f"  gap: {gs.date()} → {ge.date()}")
        if len(gaps) > 5:
            notes.append(f"  … ({len(gaps) - 5} more gaps)")
    detail = f"{n:,} obs  {first.date()} → {last.date()}"
    if ignored_market_closed:
        detail = f"{detail}; {ignored_market_closed} market-closed day(s) ignored"
    return CheckResult(label, status, detail, notes)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_adjusted_prices(
    diag: diagPrices, instrument_code: str, today: pd.Timestamp
) -> CheckResult:
    try:
        prices = diag.get_adjusted_prices(instrument_code)
    except Exception as exc:
        return CheckResult("Adjusted prices", FAIL, str(exc))

    return _gap_check(prices, "Adjusted prices", today)


def check_multiple_prices(
    diag: diagPrices, instrument_code: str, today: pd.Timestamp
) -> CheckResult:
    try:
        mp = diag.get_multiple_prices(instrument_code)
    except Exception as exc:
        return CheckResult("Multiple prices", FAIL, str(exc))

    if mp is None or mp.empty:
        return CheckResult("Multiple prices", FAIL, "no data found")

    # Use the PRICE column as the series to gap-check
    price_col = mp["PRICE"] if "PRICE" in mp.columns else mp.iloc[:, 0]
    result = _gap_check(price_col, "Multiple prices", today)

    # Show current contracts from the last row
    try:
        contracts = mp.current_contract_dict()
        result.notes.insert(
            0,
            f"current contracts — price: {contracts.price}  "
            f"carry: {contracts.carry}  forward: {contracts.forward}",
        )
    except (missingData, Exception):
        result.notes.insert(0, "could not read current contract dict")

    return result


def check_roll_parameters_and_calendar(
    data_contracts: dataContracts, instrument_code: str, today: pd.Timestamp
) -> CheckResult:
    notes: List[str] = []
    status = PASS

    # 1. Roll parameters
    try:
        rp = data_contracts.get_roll_parameters(instrument_code)
        rp_detail = (
            f"hold={rp.hold_rollcycle}  "
            f"priced={rp.priced_rollcycle}  "
            f"offset={rp.roll_offset_day}d  "
            f"carry_offset={rp.carry_offset}"
        )
    except Exception as exc:
        return CheckResult("Roll parameters / calendar", FAIL, f"roll parameters: {exc}")

    # 2. Roll calendar CSV
    calendar_path = _ROLL_CALENDAR_DIR / f"{instrument_code}.csv"
    if not calendar_path.exists():
        status = WARN
        notes.append(f"roll calendar CSV not found at {calendar_path}")
    else:
        try:
            rc_df = pd.read_csv(calendar_path, index_col=0)
            if rc_df.empty:
                status = WARN
                notes.append("roll calendar CSV is empty")
            else:
                notes.append(f"roll calendar CSV: {len(rc_df)} roll dates")
        except Exception as exc:
            status = WARN
            notes.append(f"could not read roll calendar CSV: {exc}")

    return CheckResult("Roll params / calendar", status, rp_detail, notes)


def check_expected_contracts(
    diag: diagPrices,
    data_contracts: dataContracts,
    instrument_code: str,
    today: pd.Timestamp,
) -> CheckResult:
    """Infer what the priced and forward contracts *should* be given today's
    date and compare against the latest multiple-prices row."""
    notes: List[str] = []
    status = PASS

    try:
        mp = diag.get_multiple_prices(instrument_code)
        current = mp.current_contract_dict()
        priced_id = current.price        # e.g. "202503"
        forward_id = current.forward
    except Exception as exc:
        return CheckResult("Contract roll status", FAIL, str(exc))

    try:
        rp = data_contracts.get_roll_parameters(instrument_code)
    except Exception as exc:
        return CheckResult("Contract roll status", FAIL, f"roll parameters: {exc}")

    # Build contractDateWithRollParameters for the current priced contract
    try:
        priced_contract = data_contracts.get_contract_from_db_given_code_and_id(
            instrument_code, priced_id
        )
        cd = priced_contract.contract_date
        cd_with_rp = contractDateWithRollParameters(cd, rp)
        roll_date = cd_with_rp.desired_roll_date
    except Exception as exc:
        return CheckResult(
            "Contract roll status",
            WARN,
            f"priced={priced_id}  fwd={forward_id}",
            [f"could not compute roll date: {exc}"],
        )

    # Expected next (forward) contract
    try:
        expected_forward_cd = cd_with_rp.next_held_contract()
        expected_forward_id = expected_forward_cd.date_str
    except Exception:
        expected_forward_id = None

    days_to_roll = (roll_date - today.to_pydatetime()).days

    if days_to_roll < 0:
        status = WARN
        notes.append(
            f"desired roll date was {roll_date.date()} "
            f"({-days_to_roll}d ago) — consider rolling"
        )
    else:
        notes.append(f"desired roll date: {roll_date.date()} ({days_to_roll}d away)")

    if expected_forward_id is not None and forward_id != expected_forward_id:
        status = WARN
        notes.append(
            f"forward contract mismatch: multiple prices has {forward_id}, "
            f"expected {expected_forward_id} by hold cycle"
        )

    detail = f"priced={priced_id}  carry={current.carry}  forward={forward_id}"
    return CheckResult("Contract roll status", status, detail, notes)


def check_fx_prices(
    diag_instr: diagInstruments,
    data_currency: dataCurrency,
    instrument_code: str,
    today: pd.Timestamp,
) -> CheckResult:
    try:
        currency = diag_instr.get_currency(instrument_code)
    except Exception as exc:
        return CheckResult("FX prices", FAIL, f"cannot read instrument currency: {exc}")

    base = data_currency.get_base_currency()

    if currency == base:
        return CheckResult(
            "FX prices",
            PASS,
            f"instrument currency is base ({base}) — no FX conversion needed",
        )

    fx_pair = f"{currency}{base}"

    try:
        fx = data_currency.get_fx_prices(fx_pair)
    except Exception as exc:
        return CheckResult("FX prices", FAIL, f"{fx_pair}: {exc}")

    if fx is None or fx.dropna().empty:
        # Try the inverse pair
        inv_pair = f"{base}{currency}"
        try:
            fx = data_currency.get_fx_prices(inv_pair)
            if fx is not None and not fx.dropna().empty:
                fx_pair = f"{inv_pair} (inverse)"
        except Exception:
            return CheckResult("FX prices", FAIL, f"no data for {fx_pair} or {inv_pair}")

    return _gap_check(fx, f"FX prices ({fx_pair})", today, allow_missing_today=True)


# ---------------------------------------------------------------------------
# Universe membership check
# ---------------------------------------------------------------------------

def check_universe_membership(
    diag: diagPrices, instrument_code: str
) -> CheckResult:
    """Check if the instrument appears in the active multiple-prices universe."""
    try:
        active = diag.get_list_of_instruments_in_multiple_prices(ignore_stale=True)
        stale = diag.get_stale_instruments()
    except Exception as exc:
        return CheckResult("Universe membership", WARN, str(exc))

    if instrument_code in stale:
        return CheckResult(
            "Universe membership",
            WARN,
            "instrument is marked STALE in production config",
        )
    if instrument_code not in active:
        return CheckResult(
            "Universe membership",
            WARN,
            "not in active multiple-prices universe (not traded)",
        )
    from syscore.universe import get_universe
    return CheckResult("Universe membership", PASS, f"in active universe ({get_universe().value})")


# ---------------------------------------------------------------------------
# Contract prices (per-contract raw data)
# ---------------------------------------------------------------------------

def check_contract_prices(
    diag: diagPrices, instrument_code: str, today: pd.Timestamp
) -> CheckResult:
    """Check that the currently-active price/forward contracts have recent data."""
    notes: List[str] = []
    status = PASS

    try:
        mp = diag.get_multiple_prices(instrument_code)
        current = mp.current_contract_dict()
        contracts_to_check = {
            "price": current.price,
            "forward": current.forward,
        }
    except Exception as exc:
        return CheckResult("Contract prices", FAIL, str(exc))

    for role, date_str in contracts_to_check.items():
        try:
            from sysobjects.contracts import futuresContract
            contract = futuresContract(instrument_code, date_str)
            prices = diag.get_merged_prices_for_contract_object(contract)
            final = prices.return_final_prices().dropna()
            if final.empty:
                status = FAIL
                notes.append(f"{role} contract {date_str}: no FINAL prices")
                continue
            last = final.index.max()
            staleness = (today - last).days
            obs = len(final)
            if staleness > 5:
                status = WARN if status == PASS else status
                notes.append(
                    f"{role} contract {date_str}: {obs:,} obs, "
                    f"last {last.date()} ({staleness}d ago)"
                )
            else:
                notes.append(
                    f"{role} contract {date_str}: {obs:,} obs, last {last.date()}"
                )
        except Exception as exc:
            status = FAIL
            notes.append(f"{role} contract {date_str}: {exc}")

    return CheckResult("Contract prices", status, "", notes)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def check_instrument(instrument_code: str) -> bool:
    """Run all checks for ``instrument_code``.  Returns True if all PASS/WARN."""
    today = pd.Timestamp(datetime.date.today())

    print()
    print(_colored(f"  Instrument: {_BOLD}{instrument_code}", _BOLD) + _RESET if _USE_COLOR else f"  Instrument: {instrument_code}")
    print(f"  Date:       {today.date()}")
    print()

    overall_ok = True
    all_results: List[CheckResult] = []

    with dataBlob(log_name="check_instrument") as data:
        diag = diagPrices(data)
        diag_instr = diagInstruments(data)
        data_contracts = dataContracts(data)
        data_currency = dataCurrency(data)

        checks = [
            check_universe_membership(diag, instrument_code),
            check_adjusted_prices(diag, instrument_code, today),
            check_multiple_prices(diag, instrument_code, today),
            check_roll_parameters_and_calendar(data_contracts, instrument_code, today),
            check_expected_contracts(diag, data_contracts, instrument_code, today),
            check_contract_prices(diag, instrument_code, today),
            check_fx_prices(diag_instr, data_currency, instrument_code, today),
        ]

    for r in checks:
        r.print()
        all_results.append(r)
        if r.status == FAIL:
            overall_ok = False

    print()
    fail_count = sum(1 for r in all_results if r.status == FAIL)
    warn_count = sum(1 for r in all_results if r.status == WARN)

    if fail_count:
        summary = _colored(
            f"  FAIL — {fail_count} failure(s), {warn_count} warning(s)", _RED
        )
    elif warn_count:
        summary = _colored(f"  WARN — {warn_count} warning(s)", _YELLOW)
    else:
        summary = _colored("  ALL CHECKS PASSED", _GREEN)

    print(summary)
    print()
    return overall_ok


# ---------------------------------------------------------------------------
# Interactive instrument selection
# ---------------------------------------------------------------------------

def _prompt_instrument(data: dataBlob) -> str:
    diag = diagPrices(data)
    try:
        instruments = sorted(diag.get_list_of_instruments_in_multiple_prices(ignore_stale=False))
    except Exception:
        instruments = []

    if not instruments:
        return input("Instrument code: ").strip()

    print("\nInstruments with multiple prices:")
    for i, code in enumerate(instruments, 1):
        print(f"  {i:3}.  {code}")
    print()

    raw = input("Instrument code (or number): ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(instruments):
            return instruments[idx]
    return raw


# ---------------------------------------------------------------------------
# System instrument list loader
# ---------------------------------------------------------------------------

def _instruments_for_system(system_path: str) -> List[str]:
    """Load instrument list from a system config YAML.

    ``system_path`` is a dotted module path whose final component is the YAML
    stem, e.g. ``hdenman.production`` resolves to
    ``systems/hdenman/production/system.yaml`` via the standard
    ``<prefix>.system.yaml`` convention, or falls back to
    ``<prefix>.<stem>.yaml`` when the path already ends in a filename stem.

    Instruments are taken from ``instrument_weights`` keys first, then
    ``instrument_list``.
    """
    from sysdata.config.configdata import Config

    # Try ``<system_path>.system.yaml`` first (the production convention), then
    # ``<system_path>.yaml`` for configs whose last component is the stem.
    for yaml_ref in (
        f"{system_path}.system.yaml",
        f"{system_path}.yaml",
    ):
        try:
            config = Config(yaml_ref)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(
            f"Could not locate a system YAML for {system_path!r}.\n"
            f"Tried: {system_path}.system.yaml and {system_path}.yaml"
        )

    iw: dict = config.get_element_or_default("instrument_weights", {})
    if iw:
        return sorted(iw.keys())

    il: list = config.get_element_or_default("instrument_list", [])
    if il:
        return sorted(il)

    raise ValueError(
        f"System config {yaml_ref!r} has neither instrument_weights nor instrument_list"
    )


# ---------------------------------------------------------------------------
# Interactive instrument selection
# ---------------------------------------------------------------------------

def _prompt_instrument(data: dataBlob) -> str:
    diag = diagPrices(data)
    try:
        instruments = sorted(diag.get_list_of_instruments_in_multiple_prices(ignore_stale=False))
    except Exception:
        instruments = []

    if not instruments:
        return input("Instrument code: ").strip()

    print("\nInstruments with multiple prices:")
    for i, code in enumerate(instruments, 1):
        print(f"  {i:3}.  {code}")
    print()

    raw = input("Instrument code (or number): ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(instruments):
            return instruments[idx]
    return raw


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that all expected data is present for an instrument.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "instrument",
        nargs="?",
        default=None,
        help="Instrument code, e.g. EUR_micro.  Omit for interactive selection.",
    )
    parser.add_argument(
        "--system",
        default=None,
        metavar="SYSTEM",
        help=(
            "Dotted path to a system config, e.g. hdenman.production.  "
            "Checks every instrument defined in that system's instrument_weights "
            "(or instrument_list).  Mutually exclusive with a positional instrument."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI colour output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global _USE_COLOR
    args = _parse_args(argv)
    if args.no_color:
        _USE_COLOR = False

    if args.system and args.instrument:
        print("error: --system and a positional instrument are mutually exclusive.", file=sys.stderr)
        return 2

    if args.system:
        try:
            instruments = _instruments_for_system(args.system)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(
            _colored(f"\nSystem: {args.system}  ({len(instruments)} instruments)", _BOLD)
        )
        all_ok = True
        for code in instruments:
            ok = check_instrument(code)
            if not ok:
                all_ok = False
        return 0 if all_ok else 1

    instrument_code = args.instrument
    if not instrument_code:
        with dataBlob(log_name="check_instrument") as data:
            instrument_code = _prompt_instrument(data)

    instrument_code = instrument_code.strip()
    if not instrument_code:
        print("No instrument specified.", file=sys.stderr)
        return 2

    ok = check_instrument(instrument_code)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
