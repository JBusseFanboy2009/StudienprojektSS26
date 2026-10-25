import streamlit as st
import calendar
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

from yahooquery import Ticker as YQTicker
from yahooquery import search as yahoo_search

APP_VERSION = "3.3 – What-if-Vergleich mehrerer Wertpapiere"

# ==========================================
# 1. SETUP & SESSION STATE
# ==========================================
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

if "trades" not in st.session_state:
    st.session_state.trades = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "1. Portfolio Übersicht"
if "detail_ticker" not in st.session_state:
    st.session_state.detail_ticker = None

# ==========================================
# 2. HILFS- & NORMALISIERUNGSFUNKTIONEN
# ==========================================
def normalize_weight_pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    if 0 < abs(v) <= 1.0:
        return v * 100.0
    return v


def humanize_sector_key(key: str) -> str:
    mapping = {
        "realestate": "Immobilien",
        "technology": "Technologie",
        "financialservices": "Finanzdienstleistungen",
        "consumercyclical": "Zyklische Konsumgüter",
        "consumerdefensive": "Defensive Konsumgüter",
        "healthcare": "Gesundheitswesen",
        "utilities": "Versorger",
        "energy": "Energie",
        "industrials": "Industrie",
        "basicmaterials": "Grundstoffe",
        "communicationservices": "Kommunikation",
    }
    k = (key or "").strip().lower().replace(" ", "").replace("_", "")
    return mapping.get(k, key.replace("_", " ").title() if key else "Unbekannt")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def as_trade_date(value: Any) -> date:
    """Convert stored dates from the current or older session-state format."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return pd.to_datetime(value).date()
        except Exception:
            pass
    return date.today()


def currency_symbol(currency: str) -> str:
    symbols = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "CHF": "CHF",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
    }
    return symbols.get(currency.upper(), currency.upper())


def format_money(value: float, currency: str = "EUR") -> str:
    return f"{value:,.2f} {currency_symbol(currency)}"


def format_eur(value: float) -> str:
    return f"{value:,.2f} €"


def ensure_trade_ids() -> None:
    """Give every stored trade a stable id so individual rows can be deleted safely."""
    changed = False
    for trade in st.session_state.trades:
        if not trade.get("Trade ID"):
            trade["Trade ID"] = uuid4().hex
            changed = True
    if changed:
        st.session_state.trades = list(st.session_state.trades)


def delete_trade_by_id(trade_id: str) -> None:
    st.session_state.trades = [
        trade for trade in st.session_state.trades
        if trade.get("Trade ID") != trade_id
    ]


def delete_all_trades_for_ticker(ticker: str) -> None:
    ticker = ticker.upper().strip()
    st.session_state.trades = [
        trade for trade in st.session_state.trades
        if str(trade.get("Ticker", "")).upper().strip() != ticker
    ]

ensure_trade_ids()

# ==========================================
# 3. DATEN-ABFRAGEN
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker(query: str) -> list[str]:
    if not query or len(query.strip()) < 2:
        return []
    try:
        raw = yahoo_search(query)
        quotes = (raw or {}).get("quotes", [])[:10]
        results = []
        for q in quotes:
            symbol = q.get("symbol")
            if symbol:
                name = q.get("shortname") or q.get("longname") or "N/A"
                exch = q.get("exchDisp") or ""
                results.append(f"{symbol} | {name} ({exch})")
        return results
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_instrument_currency(ticker: str) -> str:
    """Detect the trading currency of a single Yahoo Finance instrument."""
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        try:
            fast = t.fast_info
            currency = fast.get("currency")
            if currency:
                return str(currency).upper()
        except Exception:
            pass

        info = t.info
        currency = info.get("currency")
        if currency:
            return str(currency).upper()
    except Exception:
        pass

    # Last-resort fallback. EUR keeps the dashboard usable even if metadata
    # is temporarily unavailable. A visible warning is shown at portfolio level.
    return "EUR"


@st.cache_data(ttl=600, show_spinner=False)
def get_purchase_default_price(ticker: str, requested_date: str) -> float:
    """Return the latest available Close on or before requested_date.

    For today this naturally falls back to the latest available market close when
    no current intraday price is available.
    """
    ticker = ticker.upper().strip()
    target = pd.Timestamp(requested_date).normalize()
    start = (target - pd.Timedelta(days=10)).date().isoformat()
    end = (target + pd.Timedelta(days=2)).date().isoformat()
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
        close_s = extract_close_series(df, ticker).dropna()
        if not close_s.empty:
            close_s.index = pd.to_datetime(close_s.index).tz_localize(None).normalize()
            close_s = close_s[close_s.index <= target]
            if not close_s.empty:
                return safe_float(close_s.iloc[-1])
    except Exception:
        pass

    # Fallback for today / temporary historical download issues.
    price, _ = get_current_price_and_trend(ticker)
    return safe_float(price)


@st.cache_data(ttl=300, show_spinner=False)
def get_current_price_and_trend(ticker: str) -> tuple[float, float]:
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        price = fast.get("lastPrice") or fast.get("regularMarketPrice")
        prev_close = fast.get("previousClose")
        if price is not None and prev_close is not None and safe_float(prev_close) > 0:
            price_f = safe_float(price)
            prev_f = safe_float(prev_close)
            trend_pct = ((price_f - prev_f) / prev_f) * 100.0
            return price_f, trend_pct
    except Exception:
        pass

    try:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
        close_s = extract_close_series(df, ticker)
        close_s = close_s.dropna()
        if len(close_s) >= 2:
            p, prev = float(close_s.iloc[-1]), float(close_s.iloc[-2])
            return p, ((p - prev) / prev) * 100.0
        if len(close_s) == 1:
            return float(close_s.iloc[0]), 0.0
    except Exception:
        pass
    return 0.0, 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def get_current_fx_rate(currency: str) -> float | None:
    """Return units of EUR per one unit of foreign currency.

    Example: if EUR/USD = 1.17, USD -> EUR is 1 / 1.17.
    """
    currency = currency.upper().strip()
    if currency == "EUR":
        return 1.0

    direct_pair = f"EUR{currency}=X"      # foreign currency per EUR
    inverse_pair = f"{currency}EUR=X"     # EUR per foreign currency

    for pair, inverse in ((direct_pair, True), (inverse_pair, False)):
        try:
            df = yf.download(pair, period="5d", progress=False, auto_adjust=False)
            close_s = extract_close_series(df, pair).dropna()
            if close_s.empty:
                continue
            rate = safe_float(close_s.iloc[-1])
            if rate > 0:
                return (1.0 / rate) if inverse else rate
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_fx_rate(currency: str, target_date: str) -> float | None:
    """Return EUR value of one unit of foreign currency on/around target_date."""
    currency = currency.upper().strip()
    if currency == "EUR":
        return 1.0

    target = pd.Timestamp(target_date).normalize()
    start = (target - pd.Timedelta(days=5)).date().isoformat()
    end = (target + pd.Timedelta(days=2)).date().isoformat()

    direct_pair = f"EUR{currency}=X"
    inverse_pair = f"{currency}EUR=X"

    for pair, inverse in ((direct_pair, True), (inverse_pair, False)):
        try:
            df = yf.download(
                pair,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
            )
            close_s = extract_close_series(df, pair).dropna()
            if close_s.empty:
                continue
            close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
            close_s = close_s[close_s.index <= target]
            if close_s.empty:
                continue
            rate = safe_float(close_s.iloc[-1])
            if rate > 0:
                return (1.0 / rate) if inverse else rate
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_fx_series(currency: str, start_date: str, end_date: str) -> pd.Series:
    """Return a daily series of EUR per unit of foreign currency."""
    currency = currency.upper().strip()
    idx = pd.date_range(start=start_date, end=end_date, freq="B")
    if currency == "EUR":
        return pd.Series(1.0, index=idx, name=currency)

    direct_pair = f"EUR{currency}=X"
    inverse_pair = f"{currency}EUR=X"
    request_start = (pd.Timestamp(start_date) - pd.Timedelta(days=5)).date().isoformat()
    request_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).date().isoformat()

    for pair, inverse in ((direct_pair, True), (inverse_pair, False)):
        try:
            df = yf.download(
                pair,
                start=request_start,
                end=request_end,
                progress=False,
                auto_adjust=False,
            )
            close_s = extract_close_series(df, pair).dropna()
            if close_s.empty:
                continue
            close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
            close_s = close_s[~close_s.index.duplicated(keep="last")]
            if inverse:
                close_s = 1.0 / close_s
            result = close_s.reindex(idx).ffill().bfill()
            result.name = currency
            return result
        except Exception:
            continue

    return pd.Series(np.nan, index=idx, name=currency)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals_yq(ticker: str) -> dict[str, Any]:
    res = {
        "pe": "N/A",
        "div_yield": "N/A",
        "market_cap": "N/A",
        "total_assets": "N/A",
        "expense_ratio": "N/A",
        "dist_policy": "N/A",
        "quote_type": "EQUITY",
        "long_name": ticker,
    }
    try:
        yq = YQTicker(ticker)
        summary = yq.summary_detail.get(ticker, {}) if isinstance(yq.summary_detail, dict) else {}
        qtype_dict = yq.quote_type.get(ticker, {}) if isinstance(yq.quote_type, dict) else {}
        fund_prof = yq.fund_profile.get(ticker, {}) if isinstance(yq.fund_profile, dict) else {}

        if isinstance(qtype_dict, dict):
            res["quote_type"] = qtype_dict.get("quoteType", "EQUITY")
            res["long_name"] = qtype_dict.get("shortName") or qtype_dict.get("longName") or ticker

        if isinstance(summary, dict):
            pe = summary.get("trailingPE")
            if pe is not None and isinstance(pe, (int, float)):
                res["pe"] = f"{pe:.2f}"

            dy = summary.get("dividendYield") or summary.get("yield")
            if dy is not None and isinstance(dy, (int, float)):
                res["div_yield"] = f"{dy * 100:.2f}%" if dy <= 1.0 else f"{dy:.2f}%"

            mc = summary.get("marketCap")
            if mc is not None and isinstance(mc, (int, float)):
                res["market_cap"] = f"{mc / 1e9:.2f} Mrd."

            ta = summary.get("totalAssets")
            if ta is not None and isinstance(ta, (int, float)):
                res["total_assets"] = f"{ta / 1e6:.2f} Mio."

        if isinstance(fund_prof, dict):
            fees = fund_prof.get("feesExpensesInvestment", {})
            if isinstance(fees, dict):
                er = fees.get("annualReportExpenseRatio")
                if er is not None:
                    res["expense_ratio"] = f"{er * 100:.2f}%"

        if res["div_yield"] != "N/A":
            res["dist_policy"] = "Ausschüttend (Distributing)"
        else:
            name = res["long_name"].upper()
            if "ACC" in name or " C " in name:
                res["dist_policy"] = "Thesaurierend (Accumulating)"
            elif "DIST" in name or " D " in name or "INC" in name:
                res["dist_policy"] = "Ausschüttend (Distributing)"
            elif res["quote_type"] == "ETF":
                res["dist_policy"] = "Thesaurierend (Accumulating)"
    except Exception:
        pass
    return res


@st.cache_data(ttl=86400, show_spinner=False)
def get_etf_breakdown_yq(ticker: str) -> tuple[dict[str, float], dict[str, float]]:
    sectors, top_holdings = {}, {}
    try:
        yq = YQTicker(ticker)
        raw = getattr(yq, "fund_holding_info", None)
        if isinstance(raw, dict) and ticker in raw:
            fund_data = raw[ticker]
            if isinstance(fund_data, dict):
                sec_raw = fund_data.get("sectorWeightings")
                if isinstance(sec_raw, list):
                    for item in sec_raw:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                val = normalize_weight_pct(v)
                                if val is not None:
                                    sectors[humanize_sector_key(str(k))] = val
                elif isinstance(sec_raw, dict):
                    for k, v in sec_raw.items():
                        val = normalize_weight_pct(v)
                        if val is not None:
                            sectors[humanize_sector_key(str(k))] = val

                holdings_raw = fund_data.get("holdings")
                if isinstance(holdings_raw, list):
                    for row in holdings_raw:
                        if isinstance(row, dict):
                            name = row.get("holdingName") or row.get("symbol") or "Holding"
                            pct = row.get("holdingPercent") or row.get("pctOfPortfolio")
                            val = normalize_weight_pct(pct)
                            if val is not None:
                                top_holdings[str(name)] = val
    except Exception:
        pass
    return sectors, top_holdings


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(ticker: str, days: int) -> pd.DataFrame:
    start_date = datetime.now() - timedelta(days=days)
    try:
        df = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
        )
        close_s = extract_close_series(df, ticker).dropna()
        if close_s.empty:
            return pd.DataFrame()
        return pd.DataFrame({"Datum": close_s.index, "Kurs": close_s.values})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_history_series(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Daily raw Close prices for portfolio-history calculations."""
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=2)).date().isoformat(),
            progress=False,
            auto_adjust=False,
        )
        close_s = extract_close_series(df, ticker).dropna()
        if close_s.empty:
            return pd.Series(dtype=float, name=ticker)
        close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
        close_s = close_s[~close_s.index.duplicated(keep="last")]
        return close_s.rename(ticker)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def get_what_if_history(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Adjusted daily prices for simulations and return heatmaps.

    A small lookback before start_date is kept so the first selected-period return
    can be calculated against the previous available close. auto_adjust=True makes
    the historical series split- and dividend-adjusted.
    """
    ticker = ticker.upper().strip()
    query_start = (pd.Timestamp(start_date).normalize() - pd.Timedelta(days=15)).date().isoformat()
    query_end = (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=2)).date().isoformat()
    try:
        df = yf.download(
            ticker,
            start=query_start,
            end=query_end,
            progress=False,
            auto_adjust=True,
        )
        close_s = extract_close_series(df, ticker).dropna()
        if close_s.empty:
            return pd.Series(dtype=float, name=ticker)
        index = pd.DatetimeIndex(pd.to_datetime(close_s.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        close_s.index = index.normalize()
        close_s = close_s[~close_s.index.duplicated(keep="last")]
        close_s = close_s.sort_index()
        close_s = pd.to_numeric(close_s, errors="coerce").dropna()
        return close_s.rename(ticker)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_investment_comparison(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    investment_eur: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compare the EUR value path of the same initial EUR investment in each ticker.

    Each instrument is purchased once at its first available adjusted close on or
    after start_date. Daily values include historical FX conversion to EUR.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        return pd.DataFrame(), pd.DataFrame(), ["Das Startdatum darf nicht nach dem Enddatum liegen."]
    if investment_eur <= 0:
        return pd.DataFrame(), pd.DataFrame(), ["Der Vergleichsbetrag muss größer als 0 € sein."]

    available_values: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    series_last_dates: dict[str, pd.Timestamp] = {}

    for raw_ticker in tickers:
        ticker = str(raw_ticker).upper().strip()
        prices = get_what_if_history(ticker, start.date().isoformat(), end.date().isoformat())
        prices = prices[(prices.index >= start) & (prices.index <= end)].dropna()
        prices = prices[prices > 0]
        if prices.empty:
            warnings.append(f"{ticker}: Keine Kursdaten im gewählten Zeitraum verfügbar.")
            continue

        currency = get_instrument_currency(ticker)
        fx = get_historical_fx_series(
            currency, prices.index.min().date().isoformat(), prices.index.max().date().isoformat()
        )
        fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).normalize()
        fx = fx.reindex(prices.index).ffill().bfill()
        if currency == "EUR":
            fx = fx.fillna(1.0)
        if fx.empty or fx.isna().all() or (currency != "EUR" and not np.isfinite(fx.to_numpy(dtype=float)).any()):
            warnings.append(f"{ticker}: Kein verlässlicher historischer Wechselkurs für {currency}; Wertpapier übersprungen.")
            continue

        # Buy the same EUR amount at the first available close on/after the selected start date.
        first_price = safe_float(prices.iloc[0], np.nan)
        first_fx = safe_float(fx.iloc[0], np.nan)
        if not np.isfinite(first_price) or first_price <= 0 or not np.isfinite(first_fx) or first_fx <= 0:
            warnings.append(f"{ticker}: Kaufkurs oder Wechselkurs am ersten Handelstag fehlt; Wertpapier übersprungen.")
            continue

        units = float(investment_eur) / (first_price * first_fx)
        values = (prices * fx * units).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            warnings.append(f"{ticker}: Der Wertverlauf konnte nicht berechnet werden.")
            continue

        available_values[ticker] = values.rename(ticker)
        series_last_dates[ticker] = pd.Timestamp(values.index[-1]).normalize()
        final_value = safe_float(values.iloc[-1])
        return_eur = final_value - float(investment_eur)
        summary_rows.append({
            "Ticker": ticker,
            "Handelswährung": currency,
            "Startkapital (€)": float(investment_eur),
            "Endwert (€)": final_value,
            "Rendite (€)": return_eur,
            "Rendite (%)": (return_eur / float(investment_eur)) * 100.0,
            "Kaufdatum": pd.Timestamp(values.index[0]).date(),
            "Letzter Kurs": pd.Timestamp(values.index[-1]).date(),
        })

    if not available_values:
        return pd.DataFrame(), pd.DataFrame(), warnings or ["Für die ausgewählten Wertpapiere sind keine Daten verfügbar."]

    # Use a common business-day axis for comparison. Values carry forward over each
    # instrument's market holidays, but are not extended beyond its last actual quote.
    common_index = pd.date_range(start=start, end=max(series_last_dates.values()), freq="B", name="Datum")
    comparison = pd.DataFrame(index=common_index)
    for ticker, values in available_values.items():
        aligned = values.reindex(common_index).ffill()
        aligned.loc[common_index < values.index.min()] = np.nan
        aligned.loc[common_index > values.index.max()] = np.nan
        comparison[ticker] = aligned

    summary = pd.DataFrame(summary_rows).sort_values("Ticker").reset_index(drop=True)
    return comparison, summary, warnings


def monthly_schedule(start_date: date, end_date: date) -> list[date]:
    """Create monthly dates on the starting day (or month-end if unavailable)."""
    if start_date > end_date:
        return []
    scheduled = []
    year, month = start_date.year, start_date.month
    while True:
        day = min(start_date.day, calendar.monthrange(year, month)[1])
        current = date(year, month, day)
        if current > end_date:
            break
        scheduled.append(current)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return scheduled


def investment_schedule(start_date: date, end_date: date, frequency: str) -> list[date]:
    """Return planned calendar dates for a daily, weekly, or monthly savings plan.

    The first contribution is always scheduled on the selected start date. Daily
    schedules use calendar days; non-trading dates are rolled to the next available
    market close when the simulation executes trades.
    """
    if start_date > end_date:
        return []
    if frequency == "Täglich":
        return [ts.date() for ts in pd.date_range(start=start_date, end=end_date, freq="D")]
    if frequency == "Wöchentlich":
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=7)
        return dates
    return monthly_schedule(start_date, end_date)


def describe_what_if_schedule(start_date: date, mode: str, frequency: str = "Monatlich") -> str:
    """Explain precisely when simulated contributions are planned and executed."""
    if mode == "Einzelkauf":
        return (
            f"Kaufzeitpunkt: einmalig am Startdatum {start_date.strftime('%d.%m.%Y')} (Tag X). "
            "Ausgeführt wird zum ersten verfügbaren Schlusskurs am oder nach diesem Datum."
        )
    if frequency == "Täglich":
        return (
            f"Kaufzeitpunkte: täglich an jedem Kalendertag ab einschließlich {start_date.strftime('%d.%m.%Y')} "
            "bis heute. Fällt ein Termin auf einen Nicht-Handelstag, wird zum ersten verfügbaren Schlusskurs "
            "am oder nach diesem Termin gekauft. Dadurch können mehrere Tagesraten – etwa vom Wochenende – "
            "am selben Handelstag ausgeführt werden."
        )
    if frequency == "Wöchentlich":
        weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        weekday = weekdays[start_date.weekday()]
        return (
            f"Kaufzeitpunkte: wöchentlich jeden {weekday}, erstmals am Startdatum "
            f"{start_date.strftime('%d.%m.%Y')}, danach alle 7 Tage bis heute. "
            "Fällt ein Termin auf einen Nicht-Handelstag, wird zum ersten verfügbaren Schlusskurs "
            "am oder nach diesem Termin gekauft."
        )
    return (
        f"Kaufzeitpunkte: monatlich jeweils am {start_date.day}. Kalendertag, erstmals am Startdatum "
        f"{start_date.strftime('%d.%m.%Y')}. Gibt es diesen Tag in einem Monat nicht, gilt dessen letzter Tag "
        "(z. B. 28./29. Februar). Fällt ein Termin auf einen Nicht-Handelstag, wird zum ersten verfügbaren "
        "Schlusskurs am oder nach diesem Termin gekauft."
    )


def calculate_what_if_simulation(
    ticker: str,
    start_date: date,
    mode: str,
    contribution_eur: float,
    frequency: str = "Monatlich",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct an EUR-denominated one-off purchase or configurable savings plan."""
    today = date.today()
    prices = get_what_if_history(ticker, start_date.isoformat(), today.isoformat())
    prices = prices[prices.index <= pd.Timestamp(today)].dropna()
    if prices.empty:
        return pd.DataFrame(), {"error": "Für diesen Ticker sind keine historischen Kurse verfügbar."}

    currency = get_instrument_currency(ticker)
    fx = get_historical_fx_series(
        currency,
        prices.index.min().date().isoformat(),
        prices.index.max().date().isoformat(),
    )
    fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).normalize()
    fx = fx.reindex(prices.index).ffill().bfill()
    if currency != "EUR" and (fx.dropna().empty or not np.isfinite(safe_float(fx.iloc[-1], np.nan))):
        return pd.DataFrame(), {
            "error": f"Kein verlässlicher EUR-Wechselkurs für {currency} verfügbar; die EUR-Bewertung wurde nicht geschätzt."
        }
    fx = fx.fillna(1.0 if currency == "EUR" else np.nan)
    if fx.isna().all():
        return pd.DataFrame(), {"error": f"Keine historischen Wechselkurse für {currency} verfügbar."}

    if mode == "Einzelkauf":
        planned_dates = [start_date]
    else:
        planned_dates = investment_schedule(start_date, today, frequency)

    share_additions = pd.Series(0.0, index=prices.index)
    cash_flows = pd.Series(0.0, index=prices.index)
    executed_dates: list[pd.Timestamp] = []
    missed_dates: list[date] = []
    for planned_date in planned_dates:
        target = pd.Timestamp(planned_date).normalize()
        pos = int(prices.index.searchsorted(target, side="left"))
        if pos >= len(prices):
            # If the latest market close is only a few days old, use it for a
            # contribution scheduled for today/a non-trading day at the end.
            last_day = prices.index[-1]
            if planned_date <= today and (target - last_day).days <= 4:
                pos = len(prices) - 1
            else:
                missed_dates.append(planned_date)
                continue
        execution_date = prices.index[pos]
        local_price = safe_float(prices.iloc[pos])
        fx_rate = safe_float(fx.iloc[pos], np.nan)
        if local_price <= 0 or not np.isfinite(fx_rate) or fx_rate <= 0:
            missed_dates.append(planned_date)
            continue
        # contribution_eur is converted at the historical FX rate; then units
        # are valued with the local price and FX rate of each subsequent day.
        units = float(contribution_eur) / (local_price * fx_rate)
        share_additions.loc[execution_date] += units
        cash_flows.loc[execution_date] += float(contribution_eur)
        executed_dates.append(execution_date)

    if not executed_dates:
        return pd.DataFrame(), {
            "error": "Im gewählten Zeitraum konnte kein Kauf zu einem verfügbaren Kurs ausgeführt werden."
        }

    units_held = share_additions.cumsum()
    invested = cash_flows.cumsum()
    value_eur = units_held * prices * fx
    history = pd.DataFrame({
        "Datum": prices.index,
        "Investiertes Kapital (€)": invested.values,
        "Aktueller Wert (€)": value_eur.values,
    })
    # Normally the chart begins on start_date. If the start date is today and
    # Yahoo has not published a newer close yet, retain the latest close as proxy.
    chart_start = min(pd.Timestamp(start_date).normalize(), prices.index[-1])
    history = history[history["Datum"] >= chart_start].copy()
    history = history.replace([np.inf, -np.inf], np.nan).dropna(subset=["Aktueller Wert (€)"])
    if history.empty:
        return pd.DataFrame(), {"error": "Die Verlaufskurve konnte nicht erstellt werden."}

    total_invested = safe_float(history["Investiertes Kapital (€)"].iloc[-1])
    current_value = safe_float(history["Aktueller Wert (€)"].iloc[-1])
    absolute_return = current_value - total_invested
    performance_pct = (absolute_return / total_invested * 100.0) if total_invested > 0 else 0.0
    metrics = {
        "currency": currency,
        "total_invested": total_invested,
        "current_value": current_value,
        "absolute_return": absolute_return,
        "performance_pct": performance_pct,
        "transactions": len(executed_dates),
        "planned_transactions": len(planned_dates),
        "frequency": frequency if mode == "Sparplan" else None,
        "schedule_description": describe_what_if_schedule(start_date, mode, frequency),
        "first_execution_date": min(executed_dates).date(),
        "last_execution_date": max(executed_dates).date(),
        "missed_dates": missed_dates,
        "last_price_date": prices.index[-1].date(),
    }
    return history, metrics


def build_return_heatmap_matrix(daily_returns: pd.Series, aggregation: str) -> pd.DataFrame:
    """Arrange daily returns as a year/month, year/week, year/day or annual matrix."""
    returns = daily_returns.dropna().copy()
    if returns.empty:
        return pd.DataFrame()
    returns.index = pd.DatetimeIndex(pd.to_datetime(returns.index)).normalize()
    returns = returns[~returns.index.duplicated(keep="last")].sort_index()
    frame = pd.DataFrame({"Return": returns})

    def compound(values: pd.Series) -> float:
        valid = pd.to_numeric(values, errors="coerce").dropna()
        return float((1.0 + valid).prod() - 1.0) if not valid.empty else np.nan

    if aggregation == "Jahre":
        annual = frame.groupby(frame.index.year)["Return"].apply(compound)
        return pd.DataFrame([annual.values], index=["Gesamt"], columns=[str(int(y)) for y in annual.index])

    if aggregation == "Monate":
        frame["Jahr"] = frame.index.year
        frame["Monat"] = frame.index.month
        grouped = frame.groupby(["Jahr", "Monat"])["Return"].apply(compound)
        years = sorted(frame["Jahr"].unique())
        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        matrix = pd.DataFrame(np.nan, index=years, columns=month_names, dtype=float)
        for (year, month), value in grouped.items():
            matrix.loc[int(year), month_names[int(month) - 1]] = value
        return matrix

    if aggregation == "Wochen":
        iso = frame.index.isocalendar()
        frame["ISO-Jahr"] = iso["year"].to_numpy()
        frame["KW"] = iso["week"].to_numpy()
        grouped = frame.groupby(["ISO-Jahr", "KW"])["Return"].apply(compound)
        years = sorted(int(y) for y in frame["ISO-Jahr"].unique())
        week_names = [f"KW {w:02d}" for w in range(1, 54)]
        matrix = pd.DataFrame(np.nan, index=years, columns=week_names, dtype=float)
        for (year, week), value in grouped.items():
            matrix.loc[int(year), week_names[int(week) - 1]] = value
        return matrix

    # Daily view: calendar-aligned (month/day) columns, with one row per year.
    years = sorted(int(y) for y in frame.index.year.unique())
    reference_dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    day_names = [dt.strftime("%d.%m") for dt in reference_dates]
    matrix = pd.DataFrame(np.nan, index=years, columns=day_names, dtype=float)
    for dt, value in frame["Return"].items():
        key = dt.strftime("%d.%m")
        if key in matrix.columns:
            matrix.loc[int(dt.year), key] = float(value)
    return matrix


# ==========================================
# 4. DATAFRAME / HISTORICAL PORTFOLIO ENGINE
# ==========================================
def extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract a single Close series from both single- and multi-level yfinance output."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    ticker = ticker.upper().strip()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # Typical current yfinance layout: first level is price field.
            if "Close" in df.columns.get_level_values(0):
                close_df = df["Close"]
                if isinstance(close_df, pd.DataFrame):
                    if ticker in close_df.columns:
                        return close_df[ticker]
                    return close_df.iloc[:, 0]
                return close_df

            # Fallback for the alternative orientation.
            if "Close" in df.columns.get_level_values(-1):
                close_df = df.xs("Close", axis=1, level=-1)
                if isinstance(close_df, pd.DataFrame):
                    if ticker in close_df.columns:
                        return close_df[ticker]
                    return close_df.iloc[:, 0]
                return close_df

        if "Close" in df.columns:
            return df["Close"]
        return df.iloc[:, 0]
    except Exception:
        return pd.Series(dtype=float)


def normalize_trades(raw_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Migrate old session-state trades and guarantee Buy Date for all rows."""
    normalized = []
    for trade in raw_trades:
        row = dict(trade)
        row["Ticker"] = str(row.get("Ticker", "")).upper().strip()
        row["Buy Price"] = safe_float(row.get("Buy Price"))
        row["Quantity"] = safe_float(row.get("Quantity"))
        # Existing projects used Date. Use it as fallback so old sessions keep working.
        row["Buy Date"] = as_trade_date(row.get("Buy Date") or row.get("Date"))
        row["Trade ID"] = str(row.get("Trade ID") or uuid4().hex)
        normalized.append(row)
    return normalized


@st.cache_data(ttl=3600, show_spinner=False)
def get_trade_purchase_fx(currency: str, buy_date_iso: str) -> float | None:
    return get_historical_fx_rate(currency, buy_date_iso)


def build_portfolio_dataframe(df_trades: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if df_trades.empty:
        return pd.DataFrame(), []

    tickers = df_trades["Ticker"].dropna().astype(str).str.upper().unique().tolist()
    currency_map = {ticker: get_instrument_currency(ticker) for ticker in tickers}
    warnings = []

    working = df_trades.copy()
    working["Currency"] = working["Ticker"].map(currency_map)
    working["Investierter Wert"] = working["Quantity"] * working["Buy Price"]

    # The EUR purchase basis uses the FX rate on the actual purchase date.
    purchase_fx = []
    for _, row in working.iterrows():
        buy_date_value = as_trade_date(row["Buy Date"])
        fx = get_trade_purchase_fx(row["Currency"], buy_date_value.isoformat())
        if fx is None:
            if row["Currency"] != "EUR":
                warnings.append(
                    f"Kein historischer FX-Kurs für {row['Currency']} am {buy_date_value.strftime('%d.%m.%Y')}; "
                    "der Kaufwert dieser Tranche wird vorläufig nicht in die EUR-Kaufbasis einbezogen."
                )
            purchase_fx.append(np.nan)
        else:
            purchase_fx.append(fx)

    working["Purchase FX EUR"] = purchase_fx
    working["Investierter Wert EUR"] = working["Investierter Wert"] * working["Purchase FX EUR"]

    grouped = working.groupby("Ticker").agg(
        Quantity=("Quantity", "sum"),
        **{
            "Investierter Wert": ("Investierter Wert", "sum"),
            "Investierter Wert EUR": ("Investierter Wert EUR", "sum"),
        },
    ).reset_index()

    grouped["Buy Price"] = np.where(
        grouped["Quantity"] > 0,
        grouped["Investierter Wert"] / grouped["Quantity"],
        0.0,
    )
    grouped["Currency"] = grouped["Ticker"].map(currency_map)

    current_prices = []
    trends = []
    current_fx = []
    for ticker in grouped["Ticker"]:
        price, trend = get_current_price_and_trend(ticker)
        current_prices.append(price)
        trends.append(trend)
        ccy = currency_map[ticker]
        fx = get_current_fx_rate(ccy)
        current_fx.append(fx)
        if fx is None and ccy != "EUR":
            warnings.append(f"Kein aktueller EUR-Wechselkurs für {ticker} ({ccy}) verfügbar.")

    grouped["Aktueller Preis"] = current_prices
    grouped["Tagestrend (%)"] = trends
    grouped["EUR je Einheit Fremdwährung"] = current_fx
    grouped["Aktueller Wert"] = grouped["Quantity"] * grouped["Aktueller Preis"]
    grouped["Aktueller Wert EUR"] = grouped["Aktueller Wert"] * grouped["EUR je Einheit Fremdwährung"]
    grouped["Performance (%)"] = np.where(
        grouped["Investierter Wert EUR"].fillna(0) > 0,
        ((grouped["Aktueller Wert EUR"] - grouped["Investierter Wert EUR"]) / grouped["Investierter Wert EUR"]) * 100,
        np.nan,
    )
    grouped["Performance EUR"] = grouped["Aktueller Wert EUR"] - grouped["Investierter Wert EUR"]

    return grouped, sorted(set(warnings))


def build_portfolio_history(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct daily EUR portfolio value from transaction dates, prices and FX."""
    if df_trades.empty:
        return pd.DataFrame()

    min_date = df_trades["Buy Date"].min()
    start = pd.Timestamp(min_date).normalize()
    end = pd.Timestamp(date.today()).normalize()
    if start > end:
        return pd.DataFrame()

    portfolio_index = pd.date_range(start=start, end=end, freq="B")
    if len(portfolio_index) == 0:
        return pd.DataFrame()

    tickers = df_trades["Ticker"].unique().tolist()
    currencies = {ticker: get_instrument_currency(ticker) for ticker in tickers}

    history_parts = []
    for ticker in tickers:
        price_s = get_history_series(
            ticker,
            start.date().isoformat(),
            end.date().isoformat(),
        )
        if price_s.empty:
            continue
        price_s.index = pd.to_datetime(price_s.index).tz_localize(None)
        price_s = price_s.reindex(portfolio_index).ffill()

        currency = currencies[ticker]
        fx_s = get_historical_fx_series(
            currency,
            start.date().isoformat(),
            end.date().isoformat(),
        ).reindex(portfolio_index).ffill().bfill()

        ticker_trades = df_trades[df_trades["Ticker"] == ticker].copy()
        ticker_trades["Buy Date"] = pd.to_datetime(ticker_trades["Buy Date"]).dt.normalize()

        quantities = pd.Series(0.0, index=portfolio_index)
        for _, trade in ticker_trades.iterrows():
            # Once a trade has happened, its shares remain in the portfolio.
            trade_date = pd.Timestamp(as_trade_date(trade["Buy Date"])).normalize()
            quantities.loc[quantities.index >= trade_date] += safe_float(trade["Quantity"])

        values_eur = quantities * price_s * fx_s
        history_parts.append(values_eur.rename(ticker))

    if not history_parts:
        return pd.DataFrame()

    history_df = pd.concat(history_parts, axis=1)
    history_df["Portfolio Wert EUR"] = history_df.sum(axis=1, min_count=1)
    history_df = history_df[["Portfolio Wert EUR"]].dropna()
    return history_df.reset_index(names="Datum")


def filter_portfolio_history_by_interval(history_df: pd.DataFrame, interval: str, first_buy_date: date) -> pd.DataFrame:
    """Filter portfolio history using the same interval labels as Detailansicht.

    If the requested interval starts before the first purchase, the chart starts at
    the first purchase instead, which is effectively the user's requested MAX behavior.
    """
    if history_df.empty:
        return history_df

    now = pd.Timestamp(date.today()).normalize()
    starts = {
        "1W": now - pd.Timedelta(days=7),
        "1M": now - pd.Timedelta(days=30),
        "6M": now - pd.Timedelta(days=180),
        "YTD": pd.Timestamp(now.year, 1, 1),
        "1Y": now - pd.Timedelta(days=365),
        "3Y": now - pd.Timedelta(days=1095),
        "Max": pd.Timestamp(first_buy_date).normalize(),
    }
    requested_start = starts.get(interval, starts["Max"])
    effective_start = max(requested_start, pd.Timestamp(first_buy_date).normalize())
    result = history_df[history_df["Datum"] >= effective_start].copy()
    return result


# ==========================================
# 5. NAVIGATION & DATENAUFBEREITUNG
# ==========================================
st.sidebar.title("Navigation")
st.sidebar.caption(APP_VERSION)
NAV_OPTIONS = [
    "1. Portfolio Übersicht",
    "2. Meine Positionen",
    "3. Detailansicht",
    "4. Risk-O-Meter",
    "5. What if",
]

# Das Radio hat absichtlich KEINEN Schlüssel, der programmgesteuert verändert wird.
# Dadurch kann ein Button aus Seite 2 sicher auf die Detailseite wechseln, ohne
# StreamlitAPIException wegen einer nachträglichen Änderung des Widget-Zustands.
current_page = st.session_state.current_page
try:
    current_index = NAV_OPTIONS.index(current_page)
except ValueError:
    current_index = 0

selected_page = st.sidebar.radio(
    "Gehe zu",
    NAV_OPTIONS,
    index=current_index,
)

# Manuelle Auswahl im Radio wird als separater App-Zustand gespeichert.
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    if selected_page != "3. Detailansicht":
        st.session_state.detail_ticker = None

page = st.session_state.current_page

normalized_trades = normalize_trades(st.session_state.trades)
df_trades = pd.DataFrame(normalized_trades)
df_portfolio = pd.DataFrame()
portfolio_warnings: list[str] = []

if not df_trades.empty:
    df_portfolio, portfolio_warnings = build_portfolio_dataframe(df_trades)

# ==========================================
# SEITE 1: PORTFOLIO ÜBERSICHT
# ==========================================
if page == "1. Portfolio Übersicht":
    st.title(" Portfolio Übersicht")

    if df_portfolio.empty:
        st.info("Dein Portfolio ist leer. Gehe zu 'Meine Positionen', um Assets hinzuzufügen.")
    else:
        if portfolio_warnings:
            with st.expander(" Hinweise zur Währungsumrechnung"):
                for warning in portfolio_warnings:
                    st.warning(warning)

        total_value_eur = df_portfolio["Aktueller Wert EUR"].fillna(0).sum()
        total_invested_eur = df_portfolio["Investierter Wert EUR"].fillna(0).sum()
        total_perf_eur = total_value_eur - total_invested_eur
        total_perf_pct = (total_perf_eur / total_invested_eur) * 100 if total_invested_eur > 0 else 0.0

        st.metric(
            label="Gesamtwert des Portfolios",
            value=format_eur(total_value_eur),
            delta=f"{total_perf_pct:.2f}% (All-Time in EUR)",
        )
        st.caption(
            "Alle Portfolio-Werte dieser Seite sind in EUR umgerechnet. "
            "Die Kaufbasis nutzt den historischen EUR-Wechselkurs am jeweiligen Kaufdatum; der aktuelle Wert nutzt den aktuellen EUR-Wechselkurs."
        )

        # Historischer Portfolioverlauf
        st.subheader(" Portfolioverlauf")
        interval_labels = ["1W", "1M", "6M", "YTD", "1Y", "3Y", "Max"]
        selected_portfolio_interval = st.radio(
            "Zeitraum",
            interval_labels,
            horizontal=True,
            index=6,
            key="portfolio_history_interval",
        )
        with st.spinner("Rekonstruiere historischen Portfolioverlauf aus Käufen, Kursen und Wechselkursen ..."):
            portfolio_history = build_portfolio_history(df_trades)

        if not portfolio_history.empty:
            first_buy_date = min(as_trade_date(x) for x in df_trades["Buy Date"])
            filtered_history = filter_portfolio_history_by_interval(
                portfolio_history,
                selected_portfolio_interval,
                first_buy_date,
            )
            fig_portfolio = px.line(
                filtered_history,
                x="Datum",
                y="Portfolio Wert EUR",
                title=f"Historischer Gesamtwert des Portfolios ({selected_portfolio_interval})",
            )
            fig_portfolio.update_traces(hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f} €<extra></extra>")
            fig_portfolio.update_layout(
                xaxis_title="Datum",
                yaxis_title="Portfolio-Wert (€)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)
            st.caption(
                "Der Verlauf wird aus den tatsächlichen Kauftranchen und den historischen Schlusskursen rekonstruiert. "
                "Dividenden, Gebühren, Steuern und automatische Änderungen der Stückzahl durch Splits werden nicht als eigene Cashflows modelliert."
            )
        else:
            st.warning("Der historische Portfolioverlauf konnte nicht aufgebaut werden. Bitte prüfe deine Kurs- oder Währungsdaten.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Aktuelle Allokation (Wert in EUR)")
            current_alloc = df_portfolio[["Ticker", "Aktueller Wert EUR"]].copy()
            current_alloc = current_alloc.rename(columns={"Aktueller Wert EUR": "Wert (€)"})
            fig_pie_curr = px.pie(
                current_alloc,
                values="Wert (€)",
                names="Ticker",
                hole=0.4,
                hover_data=["Wert (€)"],
            )
            fig_pie_curr.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_curr, use_container_width=True)

        with col2:
            st.subheader("Investitions-Allokation (Kaufbasis in EUR)")
            invest_alloc = df_portfolio[["Ticker", "Investierter Wert EUR"]].copy()
            invest_alloc = invest_alloc.rename(columns={"Investierter Wert EUR": "Kaufwert (€)"})
            fig_pie_inv = px.pie(
                invest_alloc,
                values="Kaufwert (€)",
                names="Ticker",
                hole=0.4,
                hover_data=["Kaufwert (€)"],
            )
            fig_pie_inv.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_inv, use_container_width=True)

        st.subheader("Positionen in EUR")
        overview_table = df_portfolio[
            ["Ticker", "Currency", "Quantity", "Investierter Wert EUR", "Aktueller Wert EUR", "Performance EUR", "Performance (%)"]
        ].copy()
        overview_table["Investition (€)"] = overview_table.pop("Investierter Wert EUR")
        overview_table["Wert aktuell (€)"] = overview_table.pop("Aktueller Wert EUR")
        overview_table["P&L (€)"] = overview_table.pop("Performance EUR")
        overview_table = overview_table.rename(columns={
            "Ticker": "Ticker",
            "Currency": "Handelswährung",
            "Quantity": "Stück",
            "Performance (%)": "Performance (%)",
        })
        st.dataframe(
            overview_table[
                ["Ticker", "Handelswährung", "Stück", "Investition (€)", "Wert aktuell (€)", "P&L (€)", "Performance (%)"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stück": st.column_config.NumberColumn(format="%.4g"),
                "Investition (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Wert aktuell (€)": st.column_config.NumberColumn(format="%.2f €"),
                "P&L (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Performance (%)": st.column_config.NumberColumn(format="%.2f %%"),
            },
        )

# ==========================================
# SEITE 2: MEINE POSITIONEN
# ==========================================
elif page == "2. Meine Positionen":
    st.title("Meine Positionen")

    with st.expander("Neue Position / Tranche hinzufügen", expanded=True):
        col1, col2, col3, col4, col5 = st.columns([2.2, 2.8, 1.5, 1.5, 1.3])
        with col1:
            search_query = st.text_input("Suche (z.B. 'Microsoft', 'MSFT')")

        search_results = search_ticker(search_query) if search_query else []

        with col2:
            selected_ticker_str = st.selectbox(
                "Ergebnis auswählen",
                search_results,
                index=0 if search_results else None,
                placeholder="Ticker auswählen ...",
            )
        with col3:
            buy_date = st.date_input("Kaufdatum", value=date.today(), max_value=date.today())
        clean_ticker_for_default = (selected_ticker_str.split(" | ")[0].strip().upper() if selected_ticker_str else "")
        default_buy_price = (
            get_purchase_default_price(clean_ticker_for_default, buy_date.isoformat())
            if clean_ticker_for_default
            else 0.0
        )
        with col4:
            buy_price = st.number_input(
                "Kaufkurs",
                min_value=0.0,
                value=float(default_buy_price),
                format="%.2f",
                step=1.0,
                key=f"buy_price_input_{clean_ticker_for_default}_{buy_date.isoformat()}",
            )
        with col5:
            qty = st.number_input("Stückzahl", min_value=0.0, format="%.4f", step=1.0)

        if clean_ticker_for_default and default_buy_price > 0:
            st.caption(
                f"Vorgeschlagener Kaufkurs: {format_money(default_buy_price, get_instrument_currency(clean_ticker_for_default))} "
                f"(letzter verfügbarer Schlusskurs am/bis {buy_date.strftime('%d.%m.%Y')}). Du kannst ihn überschreiben."
            )

        if st.button("Kauf eintragen"):
            if selected_ticker_str and qty > 0 and buy_price > 0 and buy_date <= date.today():
                clean_ticker = selected_ticker_str.split(" | ")[0].strip().upper()
                currency = get_instrument_currency(clean_ticker)
                st.session_state.trades.append({
                    "Ticker": clean_ticker,
                    "Buy Price": buy_price,
                    "Quantity": qty,
                    "Buy Date": buy_date.isoformat(),
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Currency": currency,
                    "Trade ID": uuid4().hex,
                })
                st.success(f"Tranche für {clean_ticker} erfolgreich hinzugefügt ({buy_date.strftime('%d.%m.%Y')}).")
                st.rerun()
            else:
                st.error("Bitte wähle einen Ticker, gib Kaufkurs und Menge > 0 ein und setze ein gültiges Kaufdatum.")

    st.divider()

    if not df_portfolio.empty:
        st.subheader("Aktuelle Bestände")
        for _, row in df_portfolio.iterrows():
            ticker = row["Ticker"]
            c1, c2, c3, c4, c5 = st.columns([2.5, 2, 2.2, 2.2, 0.5])

            c1.markdown(
                f"**{ticker}**<br>{row['Quantity']:.4g} Stück<br><small>{row['Currency']}</small>",
                unsafe_allow_html=True,
            )
            c2.metric("Ø Kaufkurs", format_money(row["Buy Price"], row["Currency"]))
            c3.metric(
                "Aktueller Kurs",
                format_money(row["Aktueller Preis"], row["Currency"]),
                f"{row['Tagestrend (%)']:.2f}% (Tagesrendite)",
            )
            c4.metric(
                "Gesamtwert",
                format_money(row["Aktueller Wert"], row["Currency"]),
                f"{row['Performance (%)']:.2f}% (Gesamt in EUR)",
            )

            with c5:
                st.write("")

            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                if st.button(" Detail", key=f"nav_{ticker}", help="Zur Detailansicht"):
                    st.session_state.current_page = "3. Detailansicht"
                    st.session_state.detail_ticker = ticker
                    st.rerun()
            with action_col2:
                if st.button(" Alle Käufe löschen", key=f"delete_all_{ticker}", help=f"Alle Käufe von {ticker} löschen"):
                    delete_all_trades_for_ticker(ticker)
                    st.rerun()

            with st.expander(f"Käufe / Tranchen von {ticker} anzeigen"):
                ticker_trades = df_trades[df_trades["Ticker"] == ticker].copy()
                ticker_trades = ticker_trades.sort_values(["Buy Date", "Date"])
                for _, t_row in ticker_trades.iterrows():
                    t_perf = 0.0
                    if t_row["Buy Price"] > 0:
                        t_perf = ((row["Aktueller Preis"] - t_row["Buy Price"]) / t_row["Buy Price"]) * 100
                    purchase_date_display = as_trade_date(t_row["Buy Date"]).strftime("%d.%m.%Y")
                    trade_col1, trade_col2 = st.columns([8, 1])
                    with trade_col1:
                        st.markdown(
                            f"- **{t_row['Quantity']:.4g} Stück** zu je {format_money(t_row['Buy Price'], row['Currency'])} "
                            f"| Kaufdatum: **{purchase_date_display}** "
                            f"| Performance seit Kauf: **{t_perf:.2f}%**"
                        )
                    with trade_col2:
                        if st.button("Delete", key=f"delete_trade_{t_row['Trade ID']}", help="Diesen Kauf löschen"):
                            delete_trade_by_id(str(t_row["Trade ID"]))
                            st.rerun()

            st.markdown("---")
    else:
        st.info("Noch keine Positionen vorhanden.")

# ==========================================
# SEITE 3: DETAILANSICHT
# ==========================================
elif page == "3. Detailansicht":
    st.title(" Detailansicht")

    if df_portfolio.empty:
        st.warning("Füge zuerst Positionen im Menü 'Meine Positionen' hinzu.")
    else:
        ticker_list = df_portfolio["Ticker"].unique().tolist()
        default_idx = 0
        if st.session_state.detail_ticker in ticker_list:
            default_idx = ticker_list.index(st.session_state.detail_ticker)

        selected_asset = st.selectbox("Wähle eine Position", ticker_list, index=default_idx)
        st.session_state.detail_ticker = selected_asset

        if selected_asset:
            fund_data = get_fundamentals_yq(selected_asset)
            sectors, top_holdings = get_etf_breakdown_yq(selected_asset)
            is_etf = fund_data["quote_type"] == "ETF" or len(sectors) > 0 or len(top_holdings) > 0
            asset_currency = get_instrument_currency(selected_asset)

            st.caption(
                f"**Name:** {fund_data['long_name']} ({selected_asset}) | "
                f"**Typ:** {'ETF/Fonds' if is_etf else 'Aktie'} | **Währung:** {asset_currency}"
            )

            time_map = {
                "7 Tage": 7,
                "1 Monat": 30,
                "6 Monate": 180,
                "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                "1 Jahr": 365,
                "3 Jahre": 1095,
                "Max": 20000,
            }
            selected_tf = st.radio("Zeitraum", list(time_map.keys()), horizontal=True, index=4)

            with st.spinner("Lade Chart ..."):
                hist_data = get_history(selected_asset, time_map[selected_tf])
                if not hist_data.empty:
                    fig_line = px.line(
                        hist_data,
                        x="Datum",
                        y="Kurs",
                        title=f"Kursverlauf {selected_asset} ({selected_tf})",
                    )
                    fig_line.update_layout(xaxis_title="Datum", yaxis_title=f"Kurs ({asset_currency})")
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.warning("Keine Verlaufsdaten verfügbar.")

            st.subheader("Fundamentaldaten")
            c1, c2, c3 = st.columns(3)

            if is_etf:
                c1.metric("Volumen (AUM)", fund_data["total_assets"])
                c2.metric("Laufende Kosten (TER)", fund_data["expense_ratio"])
                c3.metric("Ertragsverwendung", fund_data["dist_policy"])
            else:
                c1.metric("KGV (P/E)", fund_data["pe"])
                c2.metric("Div. Rendite", fund_data["div_yield"])
                c3.metric("Marktkapitalisierung", fund_data["market_cap"])

            if is_etf:
                st.divider()
                st.subheader("ETF Struktur (Branchen & Top Holdings)")
                col_etf1, col_etf2 = st.columns(2)

                with col_etf1:
                    if sectors:
                        sec_df = pd.DataFrame(list(sectors.items()), columns=["Branche", "Gewichtung (%)"])
                        fig_sec = px.pie(
                            sec_df,
                            values="Gewichtung (%)",
                            names="Branche",
                            title="Branchenaufteilung",
                        )
                        st.plotly_chart(fig_sec, use_container_width=True)
                    else:
                        st.info("Keine Branchenstruktur für dieses Instrument verfügbar.")

                with col_etf2:
                    if top_holdings:
                        holdings_df = pd.DataFrame(
                            list(top_holdings.items()),
                            columns=["Holding", "Anteil am ETF (%)"],
                        )
                        fig_holdings = px.bar(
                            holdings_df.head(10),
                            x="Anteil am ETF (%)",
                            y="Holding",
                            orientation="h",
                            title="Top Holdings (in % des GESAMTEN ETFs)",
                        )
                        fig_holdings.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_holdings, use_container_width=True)
                    else:
                        st.info("Keine Top-Holdings für dieses Instrument verfügbar.")

# ==========================================
# SEITE 4: RISK-O-METER
# ==========================================
elif page == "4. Risk-O-Meter":
    st.title(" Risk-O-Meter")

    if len(df_portfolio) < 2:
        st.warning("Für eine Risiko- & Korrelationsbewertung werden mindestens 2 verschiedene Positionen benötigt.")
    else:
        with st.spinner("Analysiere Portfoliorisiko (252 Handelstage) ..."):
            tickers = df_portfolio["Ticker"].tolist()
            # Use current EUR market values so weights are comparable across currencies.
            current_values_eur = df_portfolio["Aktueller Wert EUR"].fillna(0).values
            total_eur = current_values_eur.sum()
            weights = current_values_eur / total_eur if total_eur > 0 else np.zeros(len(tickers))

            data = yf.download(tickers, period="1y", progress=False, auto_adjust=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"]
                else:
                    closes = data["Close"] if "Close" in data.columns else data

                returns = closes.pct_change().dropna()

                st.subheader("Korrelations-Heatmap")
                st.markdown("Zeigt, wie stark sich deine Assets synchron bewegen (1 = identisch, -1 = entgegengesetzt).")
                corr_matrix = returns.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                st.subheader("Portfolio Gesamtrisiko (Volatilität)")
                cov_matrix = returns.cov() * 252
                try:
                    cov_values = cov_matrix.loc[tickers, tickers].values
                    port_variance = np.dot(weights.T, np.dot(cov_values, weights))
                except Exception:
                    # Align defensively if Yahoo returns an incomplete matrix.
                    aligned = returns.reindex(columns=tickers).dropna(how="all")
                    cov_values = aligned.cov().fillna(0).values
                    port_variance = np.dot(weights.T, np.dot(cov_values, weights))

                port_volatility = np.sqrt(max(0, port_variance)) * 100.0

                risk_profile = "Sehr Avers (Konservativ)"
                color = "green"
                if port_volatility > 12:
                    risk_profile, color = "Moderat", "orange"
                if port_volatility > 20:
                    risk_profile, color = "Risikoreich", "red"
                if port_volatility > 35:
                    risk_profile, color = "Hoch Spekulativ", "darkred"

                st.markdown(
                    f"**Annualisierte Volatilität:** <span style='color:{color}; font-size:24px'>**{port_volatility:.2f}%**</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Risikoeinstufung:** {risk_profile}")
                st.divider()

                st.subheader("Risikobeitrag pro Position")
                st.markdown("Anteil der einzelnen Assets an der Gesamtschwankung des Portfolios:")

                port_vol = np.sqrt(max(0, port_variance))
                if port_vol > 0:
                    marginal_contrib = np.dot(cov_values, weights) / port_vol
                    component_contrib = weights * marginal_contrib
                    component_contrib_pct = (component_contrib / port_vol) * 100.0
                else:
                    component_contrib_pct = np.zeros(len(tickers))

                risk_df = pd.DataFrame({
                    "Ticker": tickers,
                    "Gewichtung im Portfolio (%)": weights * 100.0,
                    "Beitrag zum Risiko (%)": component_contrib_pct,
                })

                fig_risk = px.bar(
                    risk_df,
                    x="Ticker",
                    y="Beitrag zum Risiko (%)",
                    color="Beitrag zum Risiko (%)",
                    color_continuous_scale="Reds",
                    title="Prozentualer Anteil am Gesamtrisiko",
                )
                st.plotly_chart(fig_risk, use_container_width=True)

                if port_vol > 0:
                    max_risk_asset = risk_df.loc[risk_df["Beitrag zum Risiko (%)"].idxmax()]
                    st.info(
                        f" **Das risikoreichste Asset** in deinem Portfolio ist aktuell **{max_risk_asset['Ticker']}**. "
                        f"Obwohl es {max_risk_asset['Gewichtung im Portfolio (%)']:.1f}% des Kapitals ausmacht, "
                        f"verursacht es {max_risk_asset['Beitrag zum Risiko (%)']:.1f}% der Gesamtschwankungen."
                    )


# ==========================================
# SEITE 5: WHAT IF
# ==========================================
elif page == "5. What if":
    st.title("What if")
    st.caption(
        "Simuliere historische Käufe und vergleiche Renditen über Jahre, Monate, Wochen oder Tage. "
        "Die Berechnungen verwenden angepasste Yahoo-Schlusskurse; Gebühren, Steuern und individuelle Orderkosten sind nicht enthalten."
    )

    st.divider()
    st.header("1. What-if-Investmentrechner")
    st.markdown("Wähle ein Wertpapier über dieselbe Yahoo-Query-Suche wie unter **Meine Positionen**.")
    inv_col1, inv_col2 = st.columns([1.2, 1])
    with inv_col1:
        investment_search = st.text_input(
            "Aktie oder ETF suchen",
            key="whatif_investment_search",
            placeholder="z. B. Microsoft, MSFT oder MSCI World",
        )
    investment_results = search_ticker(investment_search) if investment_search else []
    with inv_col2:
        investment_selection = st.selectbox(
            "Yahoo-Suchergebnis auswählen",
            investment_results,
            index=0 if investment_results else None,
            placeholder="Ticker auswählen ...",
            key="whatif_investment_selection",
        )

    inv_ticker = investment_selection.split(" | ")[0].strip().upper() if investment_selection else ""
    mode_col, date_col, amount_col = st.columns([1.2, 1, 1])
    with mode_col:
        investment_mode = st.radio(
            "Investitionsmodus",
            ["Einzelkauf", "Sparplan"],
            horizontal=False,
            key="whatif_investment_mode",
        )
        investment_frequency = "Monatlich"
        if investment_mode == "Sparplan":
            investment_frequency = st.selectbox(
                "Sparplan-Häufigkeit",
                ["Täglich", "Wöchentlich", "Monatlich"],
                index=2,
                key="whatif_investment_frequency",
                help="Die erste Rate ist für das Startdatum geplant. Weitere Raten richten sich nach der gewählten Häufigkeit.",
            )
    with date_col:
        investment_start = st.date_input(
            "Startdatum (erster geplanter Kauf / Tag X)",
            value=date(2020, 1, 1),
            min_value=date(1970, 1, 1),
            max_value=date.today(),
            key="whatif_investment_start",
        )
    with amount_col:
        if investment_mode == "Einzelkauf":
            investment_amount = st.number_input(
                "Einmalbetrag (€)", min_value=1.0, value=1000.0, step=100.0,
                format="%.2f", key="whatif_single_amount",
            )
        else:
            investment_amount = st.number_input(
                "Betrag je Ausführung (€)", min_value=1.0, value=100.0, step=25.0,
                format="%.2f", key="whatif_contribution_amount",
            )

    # Always make the assumed contribution calendar explicit before running the simulation.
    st.info(describe_what_if_schedule(investment_start, investment_mode, investment_frequency))

    if inv_ticker and st.button("Investment simulieren", type="primary", key="whatif_run_simulation"):
        with st.spinner("Lade historische Kurse und rekonstruiere die Käufe ..."):
            simulation_history, simulation_metrics = calculate_what_if_simulation(
                inv_ticker,
                investment_start,
                investment_mode,
                float(investment_amount),
                investment_frequency,
            )
        st.session_state["whatif_simulation_history"] = simulation_history
        st.session_state["whatif_simulation_metrics"] = simulation_metrics
        st.session_state["whatif_simulation_ticker"] = inv_ticker
        st.session_state["whatif_simulation_mode"] = investment_mode
        st.session_state["whatif_simulation_frequency"] = investment_frequency
        st.session_state["whatif_simulation_start"] = investment_start.isoformat()
        st.session_state["whatif_simulation_amount"] = float(investment_amount)

    saved_simulation = st.session_state.get("whatif_simulation_history", pd.DataFrame())
    saved_metrics = st.session_state.get("whatif_simulation_metrics", {})
    saved_ticker = st.session_state.get("whatif_simulation_ticker", "")
    saved_start = st.session_state.get("whatif_simulation_start", "")
    saved_frequency = st.session_state.get("whatif_simulation_frequency", "Monatlich")
    saved_amount = st.session_state.get("whatif_simulation_amount")
    simulation_inputs_changed = (
        saved_ticker != inv_ticker
        or st.session_state.get("whatif_simulation_mode", investment_mode) != investment_mode
        or saved_frequency != investment_frequency
        or saved_start != investment_start.isoformat()
        or saved_amount != float(investment_amount)
    )
    if saved_ticker and simulation_inputs_changed:
        st.caption("Die Eingaben wurden geändert. Starte die Simulation erneut, um das Ergebnis zu aktualisieren.")
    elif saved_ticker and saved_metrics:
        if saved_metrics.get("error"):
            st.warning(saved_metrics["error"])
        elif not saved_simulation.empty:
            currency_text = saved_metrics.get("currency", "EUR")
            st.subheader(f"Ergebnis: {saved_ticker} · {st.session_state.get('whatif_simulation_mode', investment_mode)}")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Investiertes Gesamtkapital", format_eur(saved_metrics["total_invested"]))
            metric_cols[1].metric("Aktueller Gesamtwert", format_eur(saved_metrics["current_value"]))
            metric_cols[2].metric("Absolute Rendite", format_eur(saved_metrics["absolute_return"]))
            metric_cols[3].metric("Performance", f"{saved_metrics['performance_pct']:.2f}%")
            st.caption(
                f"{saved_metrics['transactions']} ausgeführte Käufe von {saved_metrics.get('planned_transactions', saved_metrics['transactions'])} geplanten Raten · "
                f"Handelswährung: {currency_text} · Bewertung zum letzten verfügbaren Kurs vom "
                f"{saved_metrics['last_price_date'].strftime('%d.%m.%Y')}. "
                f"Erster tatsächlicher Kauf: {saved_metrics['first_execution_date'].strftime('%d.%m.%Y')}; "
                f"letzter tatsächlicher Kauf: {saved_metrics['last_execution_date'].strftime('%d.%m.%Y')}."
            )
            sim_fig = px.line(
                saved_simulation,
                x="Datum",
                y=["Investiertes Kapital (€)", "Aktueller Wert (€)"],
                title="Investitions- und Wertverlauf",
            )
            sim_fig.update_layout(xaxis_title="Datum", yaxis_title="Wert (€)", hovermode="x unified")
            sim_fig.update_traces(hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f} €<extra></extra>")
            st.plotly_chart(sim_fig, use_container_width=True)
            if saved_metrics.get("missed_dates"):
                st.info(
                    f"{len(saved_metrics['missed_dates'])} geplante Rate(n) konnten wegen fehlender Kurs- oder FX-Daten nicht berücksichtigt werden."
                )
            st.caption(saved_metrics.get("schedule_description", ""))
            st.caption(
                "Hinweis zu den Kursdaten: Ist für einen Termin am aktuellen Datenende noch kein Schlusskurs verfügbar, "
                "verwendet die Simulation den letzten verfügbaren Schlusskurs als Näherung. Bei nicht aktuellen Terminen "
                "wird der Kauf auf den nächsten verfügbaren Handelstag verschoben. Bruchteile von Anteilen sind erlaubt; "
                "Dividenden werden über angepasste Kurse berücksichtigt, Gebühren und Steuern nicht."
            )

    st.divider()
    st.header("2. Performance-Heatmap")
    st.markdown("Die Heatmap zeigt die kumulierte Rendite je Periode. Grün steht für Gewinn, Rot für Verlust; nahe 0 % sind die Zellen hell.")
    heat_col1, heat_col2 = st.columns([1.2, 1])
    with heat_col1:
        heat_search = st.text_input(
            "Aktie oder ETF für die Heatmap suchen",
            key="whatif_heatmap_search",
            placeholder="z. B. Apple, AAPL oder FTSE All-World",
        )
    heat_results = search_ticker(heat_search) if heat_search else []
    with heat_col2:
        heat_selection = st.selectbox(
            "Yahoo-Suchergebnis für Heatmap auswählen",
            heat_results,
            index=0 if heat_results else None,
            placeholder="Ticker auswählen ...",
            key="whatif_heatmap_selection",
        )
    heat_ticker = heat_selection.split(" | ")[0].strip().upper() if heat_selection else ""

    default_heat_start = date.today() - timedelta(days=365 * 5)
    heat_dates = st.date_input(
        "Gesamtzeitraum der Heatmap",
        value=(default_heat_start, date.today()),
        min_value=date(1970, 1, 1),
        max_value=date.today(),
        key="whatif_heatmap_dates",
    )
    if isinstance(heat_dates, (tuple, list)) and len(heat_dates) == 2:
        heat_start, heat_end = heat_dates
    else:
        heat_start, heat_end = None, None
        st.info("Bitte wähle Start- und Enddatum des Heatmap-Zeitraums aus.")

    # Aggregation control belongs to the heatmap section, beside its own filters.
    heatmap_aggregation = st.radio(
        "Heatmap-Darstellung",
        ["Jahre", "Monate", "Wochen", "Tage"],
        horizontal=True,
        key="whatif_heatmap_aggregation",
        help="Legt fest, ob die Renditen pro Jahr, Monat, Kalenderwoche oder Kalendertag zusammengefasst werden.",
    )

    if heat_ticker and heat_start and heat_end and heat_start <= heat_end:
        with st.spinner("Berechne Renditen für die Heatmap ..."):
            heat_prices = get_what_if_history(heat_ticker, heat_start.isoformat(), heat_end.isoformat())
            heat_prices = heat_prices[heat_prices.index <= pd.Timestamp(heat_end)]
            heat_returns = heat_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            heat_returns = heat_returns[
                (heat_returns.index >= pd.Timestamp(heat_start)) &
                (heat_returns.index <= pd.Timestamp(heat_end))
            ]
            heat_matrix = build_return_heatmap_matrix(heat_returns, heatmap_aggregation)

        if heat_matrix.empty or not heat_matrix.notna().any().any():
            st.warning("Für den gewählten Ticker und Zeitraum sind keine ausreichenden historischen Renditedaten verfügbar.")
        else:
            heat_matrix_pct = heat_matrix * 100.0
            valid_returns = heat_matrix_pct.to_numpy(dtype=float)
            valid_returns = valid_returns[np.isfinite(valid_returns)]
            color_limit = float(np.max(np.abs(valid_returns))) if len(valid_returns) else 1.0
            if color_limit < 0.01:
                color_limit = 0.01
            show_text = heatmap_aggregation in ("Jahre", "Monate")
            fig_heat = px.imshow(
                heat_matrix_pct,
                aspect="auto",
                text_auto=".1f" if show_text else False,
                color_continuous_scale=[(0.0, "#b2182b"), (0.5, "#f7f7f7"), (1.0, "#1a9850")],
                zmin=-color_limit,
                zmax=color_limit,
                labels={"x": "Periode", "y": "Jahr", "color": "Rendite (%)"},
                title=f"{heat_ticker} – Periodenrenditen ({heatmap_aggregation})",
            )
            fig_heat.update_traces(hovertemplate="Jahr: %{y}<br>Periode: %{x}<br>Rendite: %{z:.2f}%<extra></extra>")
            if heatmap_aggregation == "Tage":
                fig_heat.update_layout(height=max(350, min(800, 120 + 30 * len(heat_matrix_pct.index))))
            elif heatmap_aggregation == "Wochen":
                fig_heat.update_layout(height=max(350, min(800, 120 + 30 * len(heat_matrix_pct.index))))
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                f"Zeitraum: {heat_start.strftime('%d.%m.%Y')}–{heat_end.strftime('%d.%m.%Y')}. "
                "Renditen basieren auf angepassten Yahoo-Schlusskursen in der Handelswährung. "
                "Die erste ausgewählte Periode bezieht den letzten verfügbaren Schlusskurs vor dem Startdatum als Ausgangsbasis ein."
            )
    elif heat_ticker and heat_start and heat_end and heat_start > heat_end:
        st.error("Das Startdatum darf nicht nach dem Enddatum liegen.")

    st.divider()
    st.header("3. Vergleich: Gleicher Startbetrag für mehrere Wertpapiere")
    st.markdown(
        "Füge beliebig viele Aktien oder ETFs hinzu und vergleiche sie im selben Diagramm. "
        "Für jedes Wertpapier wird derselbe Betrag in Euro am ersten verfügbaren Schlusskurs "
        "am oder nach dem Startdatum investiert. Der Verlauf zeigt den Wert dieses Startkapitals "
        "in Euro – nicht die Entwicklung einer einzelnen Aktie. Historische Wechselkurse werden berücksichtigt."
    )

    if "whatif_compare_tickers" not in st.session_state:
        st.session_state["whatif_compare_tickers"] = []

    compare_search_col, compare_result_col, compare_add_col = st.columns([1.2, 1.5, 0.8])
    with compare_search_col:
        compare_search = st.text_input(
            "Aktie oder ETF suchen",
            key="whatif_compare_search",
            placeholder="z. B. Nvidia, NVDA oder MSCI World",
        )
    compare_results = search_ticker(compare_search) if compare_search else []
    with compare_result_col:
        compare_selection = st.selectbox(
            "Yahoo-Suchergebnis auswählen",
            compare_results,
            index=0 if compare_results else None,
            placeholder="Ticker auswählen ...",
            key="whatif_compare_selection",
        )
    with compare_add_col:
        st.write("")
        st.write("")
        add_compare_ticker = st.button(
            "Hinzufügen",
            key="whatif_compare_add_ticker",
            disabled=not bool(compare_selection),
            use_container_width=True,
        )

    if add_compare_ticker and compare_selection:
        selected_compare_ticker = compare_selection.split(" | ")[0].strip().upper()
        if selected_compare_ticker in st.session_state["whatif_compare_tickers"]:
            st.warning(f"{selected_compare_ticker} ist bereits in der Vergleichsliste.")
        else:
            st.session_state["whatif_compare_tickers"].append(selected_compare_ticker)
            st.rerun()

    compare_tickers = st.session_state["whatif_compare_tickers"]
    if compare_tickers:
        st.caption("Ausgewählte Wertpapiere – entferne einzelne Werte bei Bedarf:")
        for compare_ticker in list(compare_tickers):
            ticker_col, remove_col = st.columns([5, 1])
            ticker_col.write(f"**{compare_ticker}**")
            if remove_col.button("Entfernen", key=f"whatif_compare_remove_{compare_ticker}"):
                st.session_state["whatif_compare_tickers"] = [
                    ticker for ticker in st.session_state["whatif_compare_tickers"] if ticker != compare_ticker
                ]
                st.rerun()
    else:
        st.info("Suche oben nach einem Wertpapier und füge es zur Vergleichsliste hinzu. Für einen Vergleich sind mindestens zwei Wertpapiere sinnvoll.")

    compare_col1, compare_col2, compare_col3 = st.columns([1.2, 1.2, 1])
    with compare_col1:
        default_compare_start = date.today() - timedelta(days=365 * 5)
        compare_dates = st.date_input(
            "Vergleichszeitraum",
            value=(default_compare_start, date.today()),
            min_value=date(1970, 1, 1),
            max_value=date.today(),
            key="whatif_compare_dates",
        )
    with compare_col2:
        compare_initial_eur = st.number_input(
            "Gleicher Startbetrag je Wertpapier (€)",
            min_value=1.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
            key="whatif_compare_initial_eur",
            help="Dieser Betrag wird für jedes ausgewählte Wertpapier separat als Einmalanlage angesetzt.",
        )
    with compare_col3:
        st.write("")
        run_comparison = st.button(
            "Vergleich berechnen",
            type="primary",
            key="whatif_compare_run",
            disabled=len(compare_tickers) == 0 or not isinstance(compare_dates, (tuple, list)) or len(compare_dates) != 2,
            use_container_width=True,
        )

    if isinstance(compare_dates, (tuple, list)) and len(compare_dates) == 2:
        compare_start, compare_end = compare_dates
    else:
        compare_start, compare_end = None, None

    if run_comparison and compare_start and compare_end:
        if compare_start > compare_end:
            st.error("Das Startdatum darf nicht nach dem Enddatum liegen.")
        else:
            with st.spinner("Lade historische Kurse und berechne den EUR-Wertverlauf ..."):
                comparison_history, comparison_summary, comparison_warnings = calculate_investment_comparison(
                    tuple(compare_tickers),
                    compare_start.isoformat(),
                    compare_end.isoformat(),
                    float(compare_initial_eur),
                )
            st.session_state["whatif_compare_history"] = comparison_history
            st.session_state["whatif_compare_summary"] = comparison_summary
            st.session_state["whatif_compare_warnings"] = comparison_warnings
            st.session_state["whatif_compare_inputs"] = {
                "tickers": tuple(compare_tickers),
                "start": compare_start.isoformat(),
                "end": compare_end.isoformat(),
                "amount": float(compare_initial_eur),
            }

    saved_compare_history = st.session_state.get("whatif_compare_history", pd.DataFrame())
    saved_compare_summary = st.session_state.get("whatif_compare_summary", pd.DataFrame())
    saved_compare_warnings = st.session_state.get("whatif_compare_warnings", [])
    saved_compare_inputs = st.session_state.get("whatif_compare_inputs", {})
    current_compare_inputs = {
        "tickers": tuple(compare_tickers),
        "start": compare_start.isoformat() if compare_start else "",
        "end": compare_end.isoformat() if compare_end else "",
        "amount": float(compare_initial_eur),
    }
    if saved_compare_inputs and saved_compare_inputs != current_compare_inputs:
        st.caption("Vergleichseingaben wurden geändert. Klicke auf **Vergleich berechnen**, um die Darstellung zu aktualisieren.")
    elif not saved_compare_history.empty and not saved_compare_summary.empty:
        if saved_compare_warnings:
            for compare_warning in saved_compare_warnings:
                st.warning(compare_warning)

        st.subheader(f"Wertentwicklung von jeweils {format_eur(saved_compare_inputs.get('amount', compare_initial_eur))} Startkapital")
        comparison_plot = saved_compare_history.rename_axis("Datum").reset_index().melt(
            id_vars="Datum", var_name="Ticker", value_name="Wert (€)"
        ).dropna(subset=["Wert (€)"])
        if not comparison_plot.empty:
            fig_compare = px.line(
                comparison_plot,
                x="Datum",
                y="Wert (€)",
                color="Ticker",
                title="Vergleichbare Wertentwicklung in Euro",
            )
            fig_compare.update_layout(
                xaxis_title="Datum",
                yaxis_title="Wert des Startkapitals (€)",
                hovermode="x unified",
                legend_title_text="Wertpapier",
            )
            fig_compare.update_traces(hovertemplate="%{x|%d.%m.%Y}<br>%{fullData.name}: %{y:,.2f} €<extra></extra>")
            st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("Kennzahlen im Vergleich")
        st.dataframe(
            saved_compare_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Startkapital (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Endwert (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Rendite (€)": st.column_config.NumberColumn(format="%.2f €"),
                "Rendite (%)": st.column_config.NumberColumn(format="%.2f %%"),
            },
        )
        st.caption(
            "Jede Kurve startet mit demselben Eurobetrag, der zum ersten verfügbaren Schlusskurs am oder nach dem Startdatum angelegt wird. "
            "Bewertet wird mit angepassten historischen Yahoo-Kursen und historischen Wechselkursen in EUR. "
            "Dividenden sind über angepasste Kurse berücksichtigt; Gebühren und Steuern nicht. "
            "Die angegebenen Kauf- und letzten Kursdaten zeigen, bis wann für jedes Wertpapier tatsächlich Kursdaten verfügbar waren."
        )

