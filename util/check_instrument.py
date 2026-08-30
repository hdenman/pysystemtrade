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
today] that has no observation.  Public holidays are *not* excluded, so an
occasional 1-day gap around major holidays is expected.

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
# Gap detection
# ---------------------------------------------------------------------------

def _business_day_gaps(
    series: pd.Series, end: Optional[pd.Timestamp] = None
) -> Tuple[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Return (total_missing_bdays, [(gap_start, gap_end), ...]).

    Gaps are contiguous runs of missing business days.  ``end`` defaults to
    today so we also flag if the series has not been updated recently.
    """
    clean = series.dropna()
    if clean.empty:
        return 0, []

    # Cast to DatetimeIndex so Pyright sees .normalize() / floor-day ops.
    dti = pd.DatetimeIndex(clean.index)
    start: pd.Timestamp = dti.normalize().min()  # type: ignore[assignment]
    end = pd.Timestamp(end or datetime.date.today())

    all_bdays: pd.DatetimeIndex = pd.bdate_range(start, end)
    observed: pd.DatetimeIndex = dti.normalize().unique()
    missing: pd.DatetimeIndex = all_bdays.difference(observed)

    if missing.empty:
        return 0, []

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

    return len(missing), gaps


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

    missing_count, gaps = _business_day_gaps(clean, end=today)

    status = PASS
    notes: List[str] = []

    if stale_note:
        status = WARN
        notes.append(stale_note)

    if missing_count > 0:
        status = FAIL
        notes.append(f"{missing_count} missing business day(s)")
        # Show up to 5 gap examples
        for gs, ge in gaps[:5]:
            if gs == ge:
                notes.append(f"  gap: {gs.date()}")
            else:
                notes.append(f"  gap: {gs.date()} → {ge.date()}")
        if len(gaps) > 5:
            notes.append(f"  … ({len(gaps) - 5} more gaps)")

    detail = f"{n:,} obs  {first.date()} → {last.date()}"
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

    return _gap_check(fx, f"FX prices ({fx_pair})", today)


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
