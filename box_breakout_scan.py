# -*- coding: utf-8 -*-

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
import numpy as np


# =========================
# 清理代理
# =========================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


# =========================
# GitHub Actions / 本地通用路径
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODE_FILE = os.path.join(DATA_DIR, "a_share_codes_for_akshare.csv")
ST_FILE = os.path.join(DATA_DIR, "st_stocks.csv")
BELOW_8B_FILE = os.path.join(DATA_DIR, "a_share_below_8b.csv")


# =========================
# 批次参数（由 GitHub Actions 环境变量传入）
# =========================
BATCH_TOTAL = int(os.getenv("BATCH_TOTAL", "2"))
BATCH_INDEX = int(os.getenv("BATCH_INDEX", "0"))

# 调试时可手动设，比如 100；正式跑 GitHub Actions 时保持 None
TEST_LIMIT = None


# =========================
# 输出文件（按批次分别输出）
# =========================
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"box_breakout_batch_{BATCH_INDEX}.csv")
FAILED_FILE = os.path.join(OUTPUT_DIR, f"box_breakout_failed_batch_{BATCH_INDEX}.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, f"summary_batch_{BATCH_INDEX}.txt")

# 自动日期
END_DATE = pd.Timestamp.today().strftime("%Y%m%d")
TARGET_DATE = pd.Timestamp.today().normalize()
START_DATE = "20240101"


# =========================
# 运行参数
# =========================
MAX_WORKERS = 1
SLEEP_MIN = 0.3
SLEEP_MAX = 0.8


# =========================
# 箱体策略参数
# =========================
LOOKBACK_BARS = 260

# 更细的窗口，避免漏掉 45/55 这类箱体
MIN_BOX_DAYS = 30
BOX_WINDOWS = list(range(30, 150, 1))

MIN_BOX_WIDTH_PCT = 0.04
MAX_BOX_WIDTH_PCT = 0.20

MIN_INBOX_RATIO = 0.78
LOWER_BUFFER = 0.015
UPPER_BUFFER = 0.015

BREAKOUT_PCT = 0.005
YDAY_ALLOW_PCT = 0.008
NEW_HIGH_BUFFER = 0.002

MAX_BELOW_LOWER_COUNT = 3
MIN_TOUCH_UPPER_COUNT = 2
UPPER_TOUCH_BAND = 0.025
MIN_TOUCH_LOWER_COUNT = 1
LOWER_TOUCH_BAND = 0.025
RECENT_WEAK_DAYS = 7
MAX_RECENT_DRAWDOWN_FROM_UPPER = 0.07

MA_SHORT = 20
MA_LONG = 60


# =========================
# 工具函数
# =========================
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def read_csv_auto(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def standardize_hist_tx(df):
    rename_map = {}

    for col in df.columns:
        c = str(col).strip().lower()
        if c in ["date", "日期"]:
            rename_map[col] = "date"
        elif c in ["open", "开盘"]:
            rename_map[col] = "open"
        elif c in ["close", "收盘"]:
            rename_map[col] = "close"
        elif c in ["high", "最高"]:
            rename_map[col] = "high"
        elif c in ["low", "最低"]:
            rename_map[col] = "low"
        elif c in ["volume", "vol", "成交量"]:
            rename_map[col] = "volume"
        elif c in ["amount", "成交额"]:
            rename_map[col] = "amount"

    out = df.rename(columns=rename_map).copy()

    if "date" not in out.columns or "close" not in out.columns:
        raise ValueError(f"hist_tx 缺少 date/close 列, 实际列: {list(df.columns)}")

    out["date"] = safe_to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    for c in ["open", "high", "low", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "close"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def calc_ma(df, windows=(20, 60)):
    out = df.copy()
    for w in windows:
        out[f"ma{w}"] = out["close"].rolling(w).mean()
    return out


def load_st_symbols():
    df = read_csv_auto(ST_FILE)
    if "ticker" not in df.columns:
        raise ValueError(f"ST 文件里没有 ticker 列，实际列名: {list(df.columns)}")
    s = df["ticker"].astype(str).str.strip().str.extract(r"(\d{6})", expand=False)
    return set(s.dropna().str.zfill(6).tolist())


def load_below_8b_symbols():
    """
    从 data/a_share_below_8b.csv 读取 80 亿以下股票代码
    兼容常见列：ticker / symbol / secID / 代码
    最终统一为 6 位代码集合
    """
    if not os.path.exists(BELOW_8B_FILE):
        print(f"警告: 未找到 {BELOW_8B_FILE}，将不做 80 亿以下过滤")
        return set()

    df = read_csv_auto(BELOW_8B_FILE)

    code_col = None
    for c in ["ticker", "symbol", "secID", "代码", "stock_code", "证券代码", "股票代码"]:
        if c in df.columns:
            code_col = c
            break

    if code_col is None:
        raise ValueError(f"a_share_below_8b.csv 缺少代码列，当前列名: {list(df.columns)}")

    codes = (
        df[code_col]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .str.zfill(6)
        .tolist()
    )
    return set(codes)


def load_universe_from_csv():
    """
    读取本地股票池，并先后过滤：
    1. ST
    2. 市值小于 80 亿
    """
    df = read_csv_auto(CODE_FILE)

    if "symbol" not in df.columns:
        raise ValueError("代码文件里没有 symbol 列")

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out[out["symbol"].str.startswith(("sh", "sz"))].copy()
    out = out.drop_duplicates("symbol").reset_index(drop=True)

    if "secShortName" in out.columns:
        out["name"] = out["secShortName"].astype(str)
    else:
        out["name"] = out["symbol"]

    out["code6"] = out["symbol"].str[-6:]

    # 过滤 ST
    st_codes = load_st_symbols()
    before_st = len(out)
    out = out[~out["code6"].isin(st_codes)].copy()
    after_st = len(out)

    # 过滤 80 亿以下
    below_8b_codes = load_below_8b_symbols()
    before_cap = len(out)
    out = out[~out["code6"].isin(below_8b_codes)].copy()
    after_cap = len(out)

    print(f"过滤 ST 前: {before_st}，过滤后: {after_st}")
    print(f"过滤 80亿以下前: {before_cap}，过滤后: {after_cap}")

    if TEST_LIMIT is not None:
        out = out.head(TEST_LIMIT).copy()

    out = out[["symbol", "name"]].reset_index(drop=True)
    return out


def split_universe_for_batch(universe: pd.DataFrame) -> pd.DataFrame:
    if TEST_LIMIT is not None:
        return universe.reset_index(drop=True)

    idx = np.arange(len(universe))
    mask = (idx % BATCH_TOTAL) == BATCH_INDEX
    return universe.loc[mask].reset_index(drop=True)


def fetch_hist_tx_with_retry(symbol, start_date, end_date, max_retry=3):
    last_err = None
    for attempt in range(max_retry):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            if df is not None and not df.empty:
                return df
            last_err = "empty dataframe"
        except Exception as e:
            last_err = str(e)

        time.sleep(1.2 + attempt * 1.2 + random.uniform(0.2, 0.6))

    raise RuntimeError(str(last_err))


def calc_box_features(close_arr: np.ndarray):
    lower = np.quantile(close_arr, 0.10)
    upper = np.quantile(close_arr, 0.90)
    return lower, upper


def check_trend_filter(df):
    if len(df) < 65:
        return False, "bars_not_enough_for_ma"

    df = calc_ma(df, windows=(MA_SHORT, MA_LONG))
    row = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(row[f"ma{MA_SHORT}"]) or pd.isna(row[f"ma{MA_LONG}"]):
        return False, "ma_nan"

    if row["close"] <= row[f"ma{MA_SHORT}"]:
        return False, "close_below_ma20"
    if row[f"ma{MA_SHORT}"] <= row[f"ma{MA_LONG}"]:
        return False, "ma20_not_above_ma60"
    if row[f"ma{MA_SHORT}"] < prev[f"ma{MA_SHORT}"]:
        return False, "ma20_not_rising"

    return True, "ok"


def check_box_breakout(df):
    if df is None or df.empty or len(df) < max(BOX_WINDOWS) + 2:
        return {"matched": False, "reason": "bars_not_enough"}

    closes = df["close"].values
    last_close = closes[-1]
    prev_close = closes[-2]
    last_date = df["date"].iloc[-1]

    best = None

    for w in BOX_WINDOWS:
        if w < MIN_BOX_DAYS:
            continue
        if len(df) < w + 2:
            continue

        # 用昨天以前的 w 天识别箱体，今天作为突破日
        box_closes = closes[-(w + 1):-1]
        if len(box_closes) != w:
            continue

        lower, upper = calc_box_features(box_closes)
        if lower <= 0 or upper <= lower:
            continue

        width_pct = (upper - lower) / lower
        if width_pct < MIN_BOX_WIDTH_PCT or width_pct > MAX_BOX_WIDTH_PCT:
            continue

        inbox_low = lower * (1 - LOWER_BUFFER)
        inbox_high = upper * (1 + UPPER_BUFFER)
        inbox_mask = (box_closes >= inbox_low) & (box_closes <= inbox_high)
        inbox_ratio = float(inbox_mask.mean())
        if inbox_ratio < MIN_INBOX_RATIO:
            continue

        below_lower_count = int((box_closes < lower * (1 - LOWER_BUFFER)).sum())
        if below_lower_count > MAX_BELOW_LOWER_COUNT:
            continue

        touch_upper_count = int((box_closes >= upper * (1 - UPPER_TOUCH_BAND)).sum())
        if touch_upper_count < MIN_TOUCH_UPPER_COUNT:
            continue

        touch_lower_count = int((box_closes <= lower * (1 + LOWER_TOUCH_BAND)).sum())
        if touch_lower_count < MIN_TOUCH_LOWER_COUNT:
            continue

        recent = box_closes[-RECENT_WEAK_DAYS:]
        recent_max = np.max(recent)
        recent_gap = (upper - recent_max) / upper
        if recent_gap > MAX_RECENT_DRAWDOWN_FROM_UPPER:
            continue

        if not (last_close > upper * (1 + BREAKOUT_PCT)):
            continue

        if prev_close > upper * (1 + YDAY_ALLOW_PCT):
            continue

        box_max_close = np.max(box_closes)
        if last_close <= box_max_close * (1 + NEW_HIGH_BUFFER):
            continue

        score = (
            inbox_ratio * 100
            - width_pct * 100
            + touch_upper_count * 2
            + touch_lower_count * 1.5
            + ((last_close / upper) - 1) * 100
            + w * 0.25
        )

        result = {
            "matched": True,
            "window": w,
            "box_lower": round(lower, 4),
            "box_upper": round(upper, 4),
            "box_width_pct": round(width_pct * 100, 2),
            "inbox_ratio_pct": round(inbox_ratio * 100, 2),
            "touch_upper_count": touch_upper_count,
            "touch_lower_count": touch_lower_count,
            "below_lower_count": below_lower_count,
            "prev_close": round(prev_close, 4),
            "last_close": round(last_close, 4),
            "breakout_pct_vs_upper": round((last_close / upper - 1) * 100, 2),
            "box_max_close": round(box_max_close, 4),
            "date": pd.to_datetime(last_date).strftime("%Y-%m-%d"),
            "score": round(score, 4),
        }

        if (best is None) or (result["score"] > best["score"]):
            best = result

    if best is None:
        return {"matched": False, "reason": "no_valid_box_breakout"}

    return best


def evaluate_one_stock(symbol, name):
    try:
        df = fetch_hist_tx_with_retry(symbol, START_DATE, END_DATE, max_retry=3)
        df = standardize_hist_tx(df)
        df = df[df["date"] <= TARGET_DATE].copy()
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            return {"symbol": symbol, "name": name, "error": "no data before target"}

        if len(df) < 120:
            return {"symbol": symbol, "name": name, "error": "not enough bars"}

        trend_ok, trend_reason = check_trend_filter(df)
        if not trend_ok:
            return {"symbol": symbol, "name": name, "error": trend_reason}

        chk = check_box_breakout(df)
        if not chk.get("matched", False):
            return {"symbol": symbol, "name": name, "error": chk.get("reason", "not_matched")}

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        return {
            "symbol": symbol,
            "name": name,
            "date": chk["date"],
            "window": chk["window"],
            "box_lower": chk["box_lower"],
            "box_upper": chk["box_upper"],
            "box_width_pct": chk["box_width_pct"],
            "inbox_ratio_pct": chk["inbox_ratio_pct"],
            "touch_upper_count": chk["touch_upper_count"],
            "touch_lower_count": chk["touch_lower_count"],
            "below_lower_count": chk["below_lower_count"],
            "prev_close": chk["prev_close"],
            "last_close": chk["last_close"],
            "breakout_pct_vs_upper": chk["breakout_pct_vs_upper"],
            "box_max_close": chk["box_max_close"],
            "score": chk["score"],
            "pattern_note": "长箱体震荡后，今日收盘有效突破箱顶",
        }

    except Exception as e:
        return {"symbol": symbol, "name": name, "error": str(e)}


def main():
    start_time = time.perf_counter()

    print("1) 读取本地全市场股票列表（已过滤 ST 和 80亿以下）...")
    universe = load_universe_from_csv()
    universe = split_universe_for_batch(universe)

    print("股票总数:", len(universe))
    print(f"当前批次: {BATCH_INDEX + 1}/{BATCH_TOTAL}")
    print(universe.head())

    matched = []
    failed = []
    error_counter = {}

    print("\n2) 开始箱体突破扫描...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(evaluate_one_stock, row["symbol"], row["name"]): row["symbol"]
            for _, row in universe.iterrows()
        }

        total = len(future_map)
        for i, future in enumerate(as_completed(future_map), 1):
            result = future.result()

            if isinstance(result, dict) and "error" not in result:
                matched.append(result)
            else:
                failed.append(result)
                err = result.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            if i % 100 == 0 or i == total:
                elapsed_now = time.perf_counter() - start_time
                avg_per_stock = elapsed_now / i if i > 0 else 0
                est_total = avg_per_stock * total if total > 0 else 0
                remain = est_total - elapsed_now

                rh = int(max(remain, 0) // 3600)
                rm = int((max(remain, 0) % 3600) // 60)
                rs = max(remain, 0) % 60

                print(
                    f"进度: {i}/{total} | 命中: {len(matched)} | 未命中/失败: {len(failed)} "
                    f"| 预计剩余: {rh}小时 {rm}分钟 {rs:.1f}秒"
                )

    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(FAILED_FILE, index=False, encoding="utf-8-sig")
    print("\n失败/未命中记录已保存:", FAILED_FILE)

    print("\n失败原因统计：")
    for k, v in sorted(error_counter.items(), key=lambda x: -x[1]):
        print(k, v)

    if matched:
        out_df = pd.DataFrame(matched).sort_values(
            by=["window", "score", "breakout_pct_vs_upper", "inbox_ratio_pct"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n命中结果已保存:", OUTPUT_FILE)
        print("\n结果预览:")
        print(out_df.head(30))
    else:
        pd.DataFrame(columns=[
            "symbol", "name", "date", "window", "box_lower", "box_upper",
            "box_width_pct", "inbox_ratio_pct", "touch_upper_count",
            "touch_lower_count", "below_lower_count", "prev_close", "last_close",
            "breakout_pct_vs_upper", "box_max_close", "score", "pattern_note"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n没有筛到符合条件的股票。")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    summary_lines = [
        f"batch_index={BATCH_INDEX}",
        f"batch_total={BATCH_TOTAL}",
        f"target_date={TARGET_DATE.strftime('%Y-%m-%d')}",
        f"stocks_scanned={len(universe)}",
        f"matched={len(matched)}",
        f"failed={len(failed)}",
        f"elapsed_seconds={elapsed:.2f}",
        f"elapsed_hms={hours}小时 {minutes}分钟 {seconds:.2f}秒",
    ]
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    if len(universe) > 0:
        print(f"平均每只股票耗时: {elapsed / len(universe):.2f} 秒")


if __name__ == "__main__":
    main()
