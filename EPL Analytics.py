"""
EPL Analytics — Custom Metrics Builder
データ: vaastav/Fantasy-Premier-League (GitHub)
ライセンス: FPL data © Premier League, non-commercial personal use
"""

import io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import requests
import streamlit as st
from scipy.stats import pearsonr, rankdata
from scipy.stats import zscore as sp_zscore

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON  = "2024-25"
FULL_MIN = 3420.0
POS_MAP  = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

C = dict(
    pitch   = "#1a5c36",   # ピッチグリーン
    pitch_l = "#27834e",
    chalk   = "#1a1a2e",   # 本文テキスト（濃紺）← 白背景に対して高コントラスト
    amber   = "#c45c00",   # アンバー（濃いめ）
    sky     = "#0077aa",   # スカイブルー（濃いめ）
    card    = "#ffffff",
    muted   = "#444444",   # サブテキスト
    dark    = "#0f172a",   # ヘッダー背景
    neg     = "#cc2200",
    pos     = "#1a7a3a",
    bg      = "#f5f7fa",   # ページ背景（薄いグレー）
    sidebar = "#1e2d3d",   # サイドバー背景（濃紺）
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EPL Analytics", layout="wide", page_icon="⚽",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main,.main .block-container{background:#f5f7fa !important;}
p,span,li,td,th,label{color:#1a1a2e !important;}
.stMarkdown *,[data-testid="stMarkdownContainer"] *{color:#1a1a2e !important;}
h1,h2,h3,h4{color:#1a1a2e !important;}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{color:#1a1a2e !important;}
[data-testid="stSidebar"]{background:#1e2d3d !important;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
[class*="valueContainer"] *,[class*="singleValue"]{color:#1a1a2e !important;}
[data-baseweb="select"] span{color:#1a1a2e !important;}
input,textarea{color:#1a1a2e !important;background:#fff !important;}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{color:#1a1a2e !important;background:#fff !important;}
[data-baseweb="option"]{color:#1a1a2e !important;background:#fff !important;}
[data-baseweb="tag"] *{color:#fff !important;}
/* Streamlit selectbox 選択値テキスト */
[data-testid="stSelectbox"] *{color:#1a1a2e !important;}
[data-testid="stMultiSelect"] *{color:#1a1a2e !important;}
[data-testid="stSelectbox"] [data-baseweb="tag"] span{color:#fff !important;}
.stSelectbox [class^="css"],[class*=" css"]{color:#1a1a2e !important;}
div[data-baseweb="select"] > div{color:#1a1a2e !important;background:#fff !important;}
div[data-baseweb="select"] > div > div{color:#1a1a2e !important;}
.stTabs [data-baseweb="tab"]{background:#e8f0eb;color:#1a1a2e !important;font-weight:600;}
.stTabs [aria-selected="true"]{background:#1e3a5f !important;}
.stTabs [aria-selected="true"] *{color:#fff !important;}
.section-bar{height:4px;background:linear-gradient(90deg,#1a5c36,#c45c00,#0077aa,transparent);margin:.5rem 0 1rem;border-radius:2px;}
[data-testid="stDataFrame"],[data-testid="stDataFrame"] *{color:#1a1a2e !important;}
[data-testid="stNotification"] *{color:#1a1a2e !important;}
[data-testid="stSuccess"] *{color:#166534 !important;}
[data-testid="stError"] *{color:#991b1b !important;}
[data-testid="stWarning"] *{color:#92400e !important;}
/* sidebar toggle: controlled by Streamlit native UI */
</style>
""", unsafe_allow_html=True)

# ── Data fetch ────────────────────────────────────────────────────────────────

# =========================================================
# API-Football 統合（シュート・ポゼッション・コーナー等）
# =========================================================
APF_BASE   = "https://v3.football.api-sports.io"
EPL_LEAGUE = 39
SEASON_TO_APF = {"2025-26": 2025, "2024-25": 2024, "2023-24": 2023, "2022-23": 2022}

def _apf_get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """API-Football への単一リクエスト（エラー詳細をセッションに記録）"""
    hdrs = {"x-apisports-key": api_key, "Accept": "application/json"}
    try:
        r = requests.get(f"{APF_BASE}/{endpoint}", headers=hdrs,
                         params=params, timeout=15)
        if r.status_code == 200:
            d = r.json()
            # API側のエラーレスポンスを確認
            if d.get("errors"):
                st.session_state["apf_last_error"] = str(d["errors"])
                return None
            return d
        else:
            st.session_state["apf_last_error"] = f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        st.session_state["apf_last_error"] = str(e)
    return None


def check_apf_key(api_key: str) -> tuple:
    """APIキー確認と残リクエスト数を返す"""
    data = _apf_get("status", {}, api_key)
    if data and "response" in data:
        req = data["response"].get("requests", {})
        used = int(req.get("current", 0))
        limit = int(req.get("limit_day", 100))
        return True, used, limit - used
    return False, 0, 0


def fetch_finished_fixture_ids(season_str: str, api_key: str) -> list:
    """
    全試合IDを取得し終了済み(FT/AET/PEN)のみ返す（1リクエスト消費）。
    session_stateで簡易キャッシュ（ページ内で重複呼び出しを防ぐ）。
    """
    # セッションキャッシュから返す（同一セッション内で再利用）
    _fid_key = f"fixture_ids_{season_str}"
    if _fid_key in st.session_state and st.session_state[_fid_key]:
        return st.session_state[_fid_key]

    apf_season = SEASON_TO_APF.get(season_str)
    if not apf_season:
        return []

    data = _apf_get("fixtures",
                    {"league": EPL_LEAGUE, "season": apf_season},
                    api_key)
    if not data:
        return []

    finished_statuses = {"FT", "AET", "PEN"}
    ids = [
        f["fixture"]["id"]
        for f in data.get("response", [])
        if f.get("fixture", {}).get("status", {}).get("short") in finished_statuses
    ]

    if ids:
        st.session_state[_fid_key] = ids
    else:
        # デバッグ: APIレスポンスの内容をセッションに保存
        if data:
            _resp = data.get("response", [])
            _errs = data.get("errors", {})
            st.session_state["apf_debug"] = (
                f"response件数={len(_resp)}, errors={_errs}, "
                f"results={data.get('results',0)}"
            )
        else:
            st.session_state["apf_debug"] = "APIレスポンスがNone（接続エラー）"
    return ids


def _parse_stats(response: list) -> dict:
    """fixture/statistics レスポンスを {home:{...}, away:{...}} に変換"""
    result = {}
    for i, team_data in enumerate(response[:2]):
        side = "home" if i == 0 else "away"
        tname = team_data.get("team", {}).get("name", "")
        stats = {"team_name": tname}
        for stat in team_data.get("statistics", []):
            val = stat.get("value")
            if isinstance(val, str) and val.endswith("%"):
                try: val = float(val.rstrip("%"))
                except: val = None
            elif val is not None:
                try: val = float(val)
                except: pass
            stats[stat["type"]] = val
        result[side] = stats
    return result


# セッションステートが消えても残るようにキャッシュキーを管理
_APF_CACHE_KEY = "apf_stats_v1"

def _load_apf_cache() -> dict:
    """セッションから統計キャッシュを読む"""
    import json as _j
    raw = st.session_state.get(_APF_CACHE_KEY, "{}")
    try:
        return _j.loads(raw) if raw else {}
    except Exception:
        return {}

def _save_apf_cache(data: dict) -> None:
    """統計キャッシュをセッションに保存"""
    import json as _j
    st.session_state[_APF_CACHE_KEY] = _j.dumps(data)


def fetch_and_cache_stats(
    fixture_ids: list, api_key: str, max_per_run: int = 80
) -> dict:
    """
    未取得の試合だけAPIを叩いて session_state に保存。
    ボタン押下 → ページ再実行 → session_state は維持される。
    ページを閉じると消えるが、毎回ボタン押下で追加取得できる。
    """
    import time as _time

    cached  = _load_apf_cache()
    missing = [fid for fid in fixture_ids if str(fid) not in cached]
    to_fetch = missing[:max_per_run]

    if not to_fetch:
        return {int(k): v for k, v in cached.items() if int(k) in fixture_ids}

    prog = st.progress(0)
    msg  = st.empty()
    newly_fetched = 0

    for i, fid in enumerate(to_fetch):
        msg.caption(f"📡 取得中: {i+1}/{len(to_fetch)} 試合...")
        prog.progress((i+1) / len(to_fetch))
        data = _apf_get("fixtures/statistics", {"fixture": fid}, api_key)
        if data and data.get("response"):
            cached[str(fid)] = _parse_stats(data["response"])
            newly_fetched += 1
        _time.sleep(0.35)

    prog.empty()
    msg.empty()

    _save_apf_cache(cached)

    if newly_fetched > 0:
        st.sidebar.success(f"✅ {newly_fetched}試合を新たに取得しました")

    return {int(k): v for k, v in cached.items() if int(k) in fixture_ids}


# API-Football → vaastav チーム名マッピング
APF_TEAM_NAME_MAP = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man Utd",
    "Nottingham Forest":       "Nott'm Forest",
    "Newcastle United":        "Newcastle",
    "Brighton & Hove Albion":  "Brighton",
    "West Ham United":         "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur":       "Spurs",
    "Tottenham":               "Spurs",   # API-Footballの実際の表記
    "Leicester City":          "Leicester",
    "Ipswich Town":            "Ipswich",
    "Sheffield United":        "Sheffield Utd",
    "Luton Town":              "Luton",
    "Burnley":                 "Burnley",
}

def _normalize_team_name(name: str) -> str:
    """API-Footballのチーム名をvaastav形式に変換"""
    return APF_TEAM_NAME_MAP.get(name, name)


def build_team_timeseries(dg_raw, team_id_map: dict) -> "pd.DataFrame":
    """
    GW×チームの時系列データを構築。
    各GWの試合値と累積値を両方持つ。
    """
    dg = dg_raw.copy()
    for c in ["expected_goals","expected_goals_conceded","goals_scored","goals_conceded",
               "yellow_cards","red_cards","assists","saves","clean_sheets","bonus",
               "creativity","threat","influence","ict_index",
               "tackles","recoveries","clearances_blocks_interceptions","defensive_contribution"]:
        if c in dg.columns:
            dg[c] = pd.to_numeric(dg[c], errors="coerce").fillna(0)
    dg["was_home"] = dg["was_home"].fillna(False).astype(bool)
    dg["gf"] = np.where(dg["was_home"],
                         pd.to_numeric(dg["team_h_score"], errors="coerce"),
                         pd.to_numeric(dg["team_a_score"], errors="coerce"))
    dg["ga"] = np.where(dg["was_home"],
                         pd.to_numeric(dg["team_a_score"], errors="coerce"),
                         pd.to_numeric(dg["team_h_score"], errors="coerce"))
    dg["pts"] = np.where(dg["gf"] > dg["ga"], 3,
                np.where(dg["gf"] == dg["ga"], 1, 0))

    # merged_gw.csv の team列はチーム名文字列（vaastav 2025-26以降）
    # 古いシーズンでは数値IDの場合もあるため両方対応
    if "team" not in dg.columns:
        dg["team_name"] = "Unknown"
    else:
        _first = dg["team"].dropna().iloc[0] if len(dg["team"].dropna()) > 0 else 0
        try:
            int(_first)  # 数値IDの場合
            dg["team_name"] = pd.to_numeric(dg["team"], errors="coerce").map(team_id_map).fillna("Unknown")
        except (ValueError, TypeError):
            # 既にチーム名文字列の場合はそのまま使う
            dg["team_name"] = dg["team"].astype(str)

    _agg_cols = {c: "sum" for c in
                 ["expected_goals","expected_goals_conceded","goals_scored","goals_conceded",
                  "yellow_cards","red_cards","assists","saves","clean_sheets","bonus",
                  "creativity","threat","influence","ict_index",
                  "tackles","recoveries","clearances_blocks_interceptions","defensive_contribution"]
                 if c in dg.columns}
    _agg_cols.update({"gf": "first", "ga": "first", "pts": "first"})

    ts = dg.groupby(["team_name", "GW", "fixture"]).agg(**{k: (k, v) for k, v in _agg_cols.items()}).reset_index()
    ts = ts.rename(columns={
        "expected_goals": "xG", "expected_goals_conceded": "xGC",
        "goals_scored": "goals_for_sum", "goals_conceded": "goals_ag_sum",
        "clearances_blocks_interceptions": "cbi",
        "defensive_contribution": "def_contribution",
    })
    ts = ts.sort_values(["team_name", "GW"]).reset_index(drop=True)

    # 累積列を追加
    cum_cols = ["xG","xGC","pts","gf","ga","yellow_cards","creativity","threat","influence"]
    for col in cum_cols:
        if col in ts.columns:
            ts[f"{col}_cum"] = ts.groupby("team_name")[col].cumsum()

    return ts


def build_apf_team_stats(fixture_cache: dict, df_teams: "pd.DataFrame") -> "pd.DataFrame":
    """
    fixture_cache からチームごとにシュート・ポゼッション等を集計し
    df_teams に列として追加する。
    チーム名はAPI-Football→vaastav形式に正規化してマッチング。
    """
    rows = []
    for fid, sides in fixture_cache.items():
        for side, opp in [("home","away"), ("away","home")]:
            s = sides.get(side, {})
            o = sides.get(opp, {})
            if not s or not s.get("team_name"):
                continue
            rows.append({
                "team_name":        _normalize_team_name(s["team_name"]),
                "total_shots":      s.get("Total Shots"),
                "shots_on_target":  s.get("Shots on Goal"),
                "shots_inside_box": s.get("Shots insidebox"),
                "possession_pct":   s.get("Ball Possession"),
                "corners":          s.get("Corner Kicks"),
                "fouls":            s.get("Fouls"),
                "offsides":         s.get("Offsides"),
                "passes_total":     s.get("Total passes"),
                "pass_accuracy":    s.get("Passes %"),
                # 被シュート（対戦相手の攻撃指標）
                "shots_against":    o.get("Total Shots"),
                "shots_on_tgt_ag":  o.get("Shots on Goal"),
                "shots_inbox_ag":   o.get("Shots insidebox"),
            })

    if not rows:
        return df_teams

    df_apf = pd.DataFrame(rows)
    for c in df_apf.columns:
        if c != "team_name":
            df_apf[c] = pd.to_numeric(df_apf[c], errors="coerce")

    agg = df_apf.groupby("team_name").agg(
        shots_pm           = ("total_shots",       "mean"),
        shots_on_tgt_pm    = ("shots_on_target",   "mean"),
        shots_inbox_pm     = ("shots_inside_box",  "mean"),
        shots_against_pm   = ("shots_against",     "mean"),
        shots_on_tgt_ag_pm = ("shots_on_tgt_ag",   "mean"),
        shots_inbox_ag_pm  = ("shots_inbox_ag",     "mean"),
        possession         = ("possession_pct",    "mean"),
        corners_pm         = ("corners",           "mean"),
        fouls_pm           = ("fouls",             "mean"),
        offsides_pm        = ("offsides",          "mean"),
        pass_acc           = ("pass_accuracy",     "mean"),
        passes_pm          = ("passes_total",      "mean"),
        n_fixtures         = ("total_shots",       "count"),
    ).reset_index()

    merged = df_teams.merge(agg, on="team_name", how="left")

    # 計算指標
    merged["shot_conversion"]    = (merged["gf_per_match"] / merged["shots_pm"].replace(0, np.nan) * 100).round(1)
    merged["shots_on_tgt_rate"]  = (merged["shots_on_tgt_pm"] / merged["shots_pm"].replace(0, np.nan) * 100).round(1)
    merged["shots_on_tgt_ag_rate"] = (merged["shots_on_tgt_ag_pm"] / merged["shots_against_pm"].replace(0, np.nan) * 100).round(1)
    return merged

def _get(url, timeout=30):
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 Chrome/124"}, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))  # 指数バックオフ: 2, 4, 6, 8, 10秒
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def load_season(season):
    r_p = _get(f"{VAASTAV}/{season}/players_raw.csv")
    r_g = _get(f"{VAASTAV}/{season}/gws/merged_gw.csv")
    r_t = _get(f"{VAASTAV}/{season}/teams.csv")
    return (
        pd.read_csv(io.StringIO(r_p.text)) if r_p else None,
        pd.read_csv(io.StringIO(r_g.text)) if r_g else None,
        pd.read_csv(io.StringIO(r_t.text)) if r_t else None,
    )

# ── Data preparation ──────────────────────────────────────────────────────────
NUM_COLS_PLAYER = [
    "minutes","goals_scored","assists","clean_sheets","goals_conceded",
    "saves","yellow_cards","red_cards","bonus","bps","total_points","now_cost",
    "expected_goals","expected_assists","expected_goal_involvements",
    "expected_goals_conceded","influence","creativity","threat","ict_index",
    "own_goals","penalties_saved","penalties_missed","starts",
    "clean_sheets_per_90","expected_goals_per_90","expected_assists_per_90",
    "expected_goal_involvements_per_90","expected_goals_conceded_per_90",
    "goals_conceded_per_90","saves_per_90","starts_per_90",
]

NUM_COLS_GW = [
    "tackles","recoveries","clearances_blocks_interceptions","defensive_contribution",
    "goals_scored","assists","expected_goals","expected_assists",
    "expected_goal_involvements","expected_goals_conceded","goals_conceded",
    "saves","clean_sheets","yellow_cards","red_cards","bonus","minutes",
    "creativity","threat","influence","ict_index","team_a_score","team_h_score",
]

def prep_players(df_raw, team_id_map):
    df = df_raw.copy()
    df["player_name"] = df["web_name"] if "web_name" in df.columns else df.index.astype(str)
    if "id" not in df.columns and df.index.name == "id":
        df = df.reset_index()
    if "id" not in df.columns:
        df["id"] = range(len(df))  # fallback
    df["position"]    = (df["element_type"] if "element_type" in df.columns
                         else pd.Series(0, index=df.index)).map(POS_MAP).fillna("UNK")
    df["team_name"]   = (df["team"] if "team" in df.columns
                         else pd.Series(0, index=df.index)).map(team_id_map).fillna("Unknown")
    for c in NUM_COLS_PLAYER:
        df[c] = pd.to_numeric(df[c] if c in df.columns
                              else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["price_m"]      = df["now_cost"] / 10.0
    # 「名前 (チーム)」形式の表示名を追加（同名選手識別用）
    df["display_name"]  = df["player_name"] + " (" + df["team_name"] + ")"
    df["goal_luck"]    = df["goals_scored"] - df["expected_goals"]
    df["def_luck"]     = df["expected_goals_conceded"] - df["goals_conceded"]
    df["mins_p90"]     = (df["minutes"] / 90).clip(lower=0.1)
    df["xG_p90"]       = df["expected_goals"] / df["mins_p90"]
    df["xA_p90"]       = df["expected_assists"] / df["mins_p90"]
    df["xGI_p90"]      = df["expected_goal_involvements"] / df["mins_p90"]
    df["goals_p90"]    = df["goals_scored"] / df["mins_p90"]
    df["assists_p90"]  = df["assists"] / df["mins_p90"]
    df["saves_p90"]    = df["saves"] / df["mins_p90"]
    # 追加p90列（切り替え機能用）
    for _c in ["influence","creativity","threat","ict_index","goal_luck","def_luck",
               "clean_sheets","goals_conceded","yellow_cards","red_cards",
               "bonus","total_points"]:
        df[f"{_c}_p90"] = df[_c] / df["mins_p90"]
    # 守備指標（2025-26から追加。旧シーズン/他ポジは0）
    for _col, _p90col in [
        ("tackles",                        "tackles_p90"),
        ("recoveries",                     "recoveries_p90"),
        ("clearances_blocks_interceptions","cbi_p90"),
        ("defensive_contribution",         "def_contribution_p90"),
    ]:
        _series = df[_col] if _col in df.columns else pd.Series(0.0, index=df.index)
        df[_col]    = pd.to_numeric(_series, errors="coerce").fillna(0)
        df[_p90col] = df[_col] / df["mins_p90"]
    return df

def build_team_stats(dg_raw, team_id_map):
    dg = dg_raw.copy()
    for c in NUM_COLS_GW:
        dg[c] = pd.to_numeric(dg[c] if c in dg.columns
                              else pd.Series(0, index=dg.index), errors="coerce").fillna(0)
    dg["gf"] = np.where(dg["was_home"].fillna(False),
                        dg["team_h_score"], dg["team_a_score"])
    dg["ga"] = np.where(dg["was_home"].fillna(False),
                        dg["team_a_score"], dg["team_h_score"])

    # 得失点（マッチスコアから）
    ms = dg.dropna(subset=["gf","ga"]).groupby(["team","round"]).agg(
        gf=("gf","first"), ga=("ga","first")
    ).reset_index()
    # 勝ち点計算（勝=3 引=1 負=0）
    ms["pts"] = np.where(ms["gf"] > ms["ga"], 3,
                np.where(ms["gf"] == ms["ga"], 1, 0))
    ms["wins"]   = (ms["gf"] > ms["ga"]).astype(int)
    ms["draws"]  = (ms["gf"] == ms["ga"]).astype(int)
    ms["losses"] = (ms["gf"] < ms["ga"]).astype(int)
    scores = ms.groupby("team").agg(
        goals_scored =("gf",  "sum"),
        goals_conceded=("ga", "sum"),
        matches      =("round","nunique"),
        points       =("pts", "sum"),
        wins         =("wins","sum"),
        draws        =("draws","sum"),
        losses       =("losses","sum"),
    ).reset_index()

    # 攻撃指標（全選手集計）— 旧シーズンに存在しない列は0で補完
    _ALWAYS_COLS = {
        "expected_goals": "xG",
        "expected_assists": "xA",
        "creativity": "creativity",
        "threat": "threat",
        "influence": "influence",
        "yellow_cards": "yellow_cards",
        "red_cards": "red_cards",
        "bonus": "bonus",
        "assists": "assists",
    }
    _NEW_COLS = {   # 2025-26から追加。旧シーズンは0埋め
        "tackles": "tackles",
        "recoveries": "recoveries",
        "clearances_blocks_interceptions": "cbi",
        "defensive_contribution": "def_contribution",
    }
    # 存在しない列を0で補完してから集計
    for col in list(_ALWAYS_COLS.keys()) + list(_NEW_COLS.keys()):
        if col not in dg.columns:
            dg[col] = 0.0
    atk_spec = {v: (k, "sum") for k, v in {**_ALWAYS_COLS, **_NEW_COLS}.items()}
    atk = dg.groupby("team").agg(**atk_spec).reset_index()

    # GKからCS・xGC・Saves
    gk = dg[dg["position"]=="GK"].copy()
    gk_agg = gk.groupby("team").agg(
        xGC=("expected_goals_conceded","sum"),
        clean_sheets=("clean_sheets","sum"),
        saves=("saves","sum"),
    ).reset_index()

    team = scores.merge(atk, on="team").merge(gk_agg, on="team", how="left")
    team[["xGC","clean_sheets","saves"]] = team[["xGC","clean_sheets","saves"]].fillna(0)

    m = team["matches"].clip(lower=1)
    team["xG_per_match"]  = (team["xG"]  / m).round(2)
    team["xGC_per_match"] = (team["xGC"] / m).round(2)
    team["gf_per_match"]  = (team["goals_scored"] / m).round(2)
    team["pts_per_match"] = (team["points"] / m).round(2)
    team["ga_per_match"]  = (team["goals_conceded"] / m).round(2)
    team["xG_diff"]       = (team["xG"] - team["xGC"]).round(2)
    team["goal_diff"]     = team["goals_scored"] - team["goals_conceded"]
    team["goal_luck"]     = (team["goals_scored"] - team["xG"]).round(2)
    team["def_luck"]      = (team["xGC"] - team["goals_conceded"]).round(2)
    team["cs_per_match"]  = (team["clean_sheets"] / m).round(2)
    team["saves_per_match"] = (team["saves"] / m).round(2)

    return team.rename(columns={"team": "team_name"})

# ── Matplotlib style ──────────────────────────────────────────────────────────
def apply_dark_style(fig, axes=None):
    """白背景・濃いテキストの高コントラストスタイル"""
    fig.patch.set_facecolor("#f5f7fa")
    if axes is None:
        axes = fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#333333", labelsize=8)
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")
        ax.title.set_color("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, color="#e5e7eb", linewidth=0.5, alpha=0.8)
    return fig

PITCH_COLORS = [
    "#48cae4","#f4a261","#22c55e","#a78bfa","#f87171",
    "#34d399","#fb923c","#60a5fa","#e879f9","#fbbf24",
    "#4ade80","#38bdf8","#c084fc","#fb7185","#a3e635",
    "#2dd4bf","#f472b6","#818cf8","#facc15","#6ee7b7",
]

def team_color_map(teams):
    return {t: PITCH_COLORS[i % len(PITCH_COLORS)] for i, t in enumerate(sorted(teams))}

# ── Chart helpers ─────────────────────────────────────────────────────────────
def to_p90(col: str, df, mins_col: str = "mins_p90") -> tuple[str, str]:
    """
    列名を p90 版に変換する。
    既存の _p90 列があればそれを返す。なければその場で計算して返す。
    Returns: (p90_column_name, display_label)
    """
    p90_col = f"{col}_p90"
    if p90_col not in df.columns:
        if mins_col in df.columns and col in df.columns:
            df[p90_col] = df[col] / df[mins_col].clip(lower=0.1)
        else:
            return col, col  # fallback
    label = col.replace("_", " ").title() + " p90"
    return p90_col, label


def radar(df_sel, metrics, labels, title, z_pool=None):
    """Dark-themed percentile radar chart"""
    if z_pool is None:
        pool = df_sel
    else:
        pool = z_pool
    df_pct = df_sel[metrics].copy()
    for col in metrics:
        pv = pool[col].dropna()
        if len(pv) == 0:
            df_pct[col] = 0.0
            continue
        pv_nonzero = pv[pv > 0]
        pv_max     = pv.max()
        def _pct(v, _pv=pv, _pv_nz=pv_nonzero, _pv_max=pv_max):
            if v <= 0:
                # 値が0以下: プールに非ゼロが存在する場合は最下位(0.0)
                # 全員0の指標は 0.0 (意味なし)
                return 0.0
            # 値が正: 非ゼロプール内でのパーセンタイル
            if len(_pv_nz) == 0:
                return 0.0
            return float((_pv_nz <= v).mean())
        df_pct[col] = df_pct[col].apply(_pct)
    n      = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#ffffff")
    ax.spines["polar"].set_edgecolor("#cccccc")
    ax.grid(color="#e5e7eb", lw=.6)
    palette = PITCH_COLORS
    patches = []
    for i, (idx, row) in enumerate(df_pct.iterrows()):
        c    = palette[i % len(palette)]
        vals = row[metrics].tolist() + [row[metrics[0]]]
        name = str(idx)[:20]
        ax.plot(angles, vals, "o-", lw=2, color=c, alpha=.9)
        ax.fill(angles, vals, alpha=.12, color=c)
        patches.append(mpatches.Patch(color=c, label=name))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#1a1a2e", fontweight="bold")
    ax.set_ylim(0,1)
    ax.set_yticks([.25,.5,.75])
    ax.set_yticklabels(["25%","50%","75%"], size=7, color="#555555")
    ax.tick_params(pad=10)
    ax.set_title(title, size=10, color=C["pitch"], pad=20, fontweight="bold")
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.5, 1.15), fontsize=8,
              facecolor="#f5f7fa", edgecolor="#cccccc", labelcolor="#1a1a2e")
    plt.tight_layout()
    return fig

def scatter_2d(df, x_col, y_col, label_col, title, c_map=None):
    fig, ax = plt.subplots(figsize=(8,6))
    apply_dark_style(fig, ax)
    teams = df[label_col].tolist()
    colors = [c_map.get(t, C["sky"]) for t in teams] if c_map else [C["sky"]]*len(teams)
    ax.scatter(df[x_col], df[y_col], c=colors, s=100, alpha=.9, edgecolors="#374151", lw=.5, zorder=3)
    # 平均線
    ax.axhline(df[y_col].mean(), color="#374151", ls="--", lw=.8, zorder=1)
    ax.axvline(df[x_col].mean(), color="#374151", ls="--", lw=.8, zorder=1)
    # ラベル
    for _, row in df.iterrows():
        ax.annotate(row[label_col], (row[x_col], row[y_col]),
                    xytext=(5,5), textcoords="offset points",
                    fontsize=7.5, color=C["chalk"], fontweight="600")
    ax.set_xlabel(x_col.replace("_"," ").title(), color=C["muted"])
    ax.set_ylabel(y_col.replace("_"," ").title(), color=C["muted"])
    ax.set_title(title, color=C["chalk"], fontweight="bold")
    plt.tight_layout()
    return fig

def pca_plot(df, metrics, label_col, title, c_map=None):
    from numpy.linalg import svd
    X = df[metrics].fillna(0).values.astype(float)
    X_z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, Vt = svd(X_z, full_matrices=False)
    scores = X_z @ Vt[:2].T
    loadings = Vt[:2].T

    fig = plt.figure(figsize=(10, 5))
    apply_dark_style(fig)
    gs = GridSpec(1, 2, figure=fig, wspace=.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    apply_dark_style(fig, [ax1, ax2])

    # Scatterplot
    teams = df[label_col].tolist()
    colors = [c_map.get(t, C["sky"]) for t in teams] if c_map else [C["sky"]]*len(df)
    ax1.scatter(scores[:,0], scores[:,1], c=colors, s=90, alpha=.9,
                edgecolors="#374151", lw=.5, zorder=3)
    ax1.axhline(0, color="#374151", ls="--", lw=.7)
    ax1.axvline(0, color="#374151", ls="--", lw=.7)
    for i, t in enumerate(teams):
        ax1.annotate(t, (scores[i,0], scores[i,1]),
                     xytext=(5,5), textcoords="offset points",
                     fontsize=7, color=C["chalk"], fontweight="600")
    ax1.set_xlabel("PC 1", color=C["muted"])
    ax1.set_ylabel("PC 2", color=C["muted"])
    ax1.set_title(title, color=C["chalk"], fontweight="bold")

    # Loading bar chart  ← loadings shape: (n_metrics, 2)
    load_df = pd.DataFrame(loadings, index=metrics, columns=["PC1","PC2"]).reset_index()
    load_df.columns = ["metric","PC1","PC2"]
    x_pos = np.arange(len(metrics))
    w = .35
    ax2.barh(x_pos + w/2, load_df["PC1"], w, label="PC 1",
             color=C["amber"], alpha=.85)
    ax2.barh(x_pos - w/2, load_df["PC2"], w, label="PC 2",
             color=C["sky"], alpha=.85)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([m.replace("_"," ") for m in metrics],
                        fontsize=8, color=C["chalk"])
    ax2.axvline(0, color="#374151", lw=.7)
    ax2.set_xlabel("Loading", color=C["muted"])
    ax2.set_title("Loadings — which metrics drive each PC",
                  color=C["chalk"], fontweight="bold")
    ax2.legend(facecolor="#1f2937", edgecolor="#374151",
               labelcolor=C["chalk"], fontsize=9)
    plt.tight_layout()
    return fig, scores, loadings, Vt

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="padding:.8rem 0 .3rem">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
    color:{C['amber']};letter-spacing:.06em">⚽ EPL Analytics</span>
  <div style="color:{C['muted']};font-size:.72rem;margin-top:.1rem">
    Custom Metrics Builder
  </div>
</div>
""", unsafe_allow_html=True)

# 現在の日付から利用可能なシーズンを動的に生成
# EPLは8月開幕のため、8月以降は新シーズンを追加
import datetime as _dt
_now   = _dt.datetime.now()
_year  = _now.year
# 8月以降なら今年開幕のシーズンを最新に追加
_start = _year if _now.month >= 8 else _year - 1
# 2022-23から現在シーズンまでを生成（最大5シーズン）
_seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(_start, _start-5, -1)]
season = st.sidebar.selectbox("Season", _seasons)
page   = st.sidebar.radio("", ["🏟️ Team Analysis","👤 Player Analysis"], label_visibility="collapsed")
st.sidebar.markdown("---")
# API-Football データは事前取得済みJSONから読む
api_key_input = ""
apf_enabled   = False
apf_remain    = 0
st.sidebar.markdown("---")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df_p_raw, df_g_raw, df_t_raw = load_season(season)

# GW範囲フィルター（サイドバー）
if df_g_raw is not None and "GW" in df_g_raw.columns:
    _gw_col = "GW"
elif df_g_raw is not None and "round" in df_g_raw.columns:
    _gw_col = "round"
else:
    _gw_col = None

if _gw_col:
    _gw_vals = sorted(df_g_raw[_gw_col].dropna().unique().astype(int))
    _gw_min, _gw_max = int(min(_gw_vals)), int(max(_gw_vals))
    st.sidebar.markdown("**GW範囲フィルター**")
    if _gw_min == _gw_max:
        # GWが1節のみの場合はスライダーを表示しない
        st.sidebar.caption(f"取得済みGW: {_gw_min}節のみ")
        _gw_range = (_gw_min, _gw_max)
    else:
        _gw_range = st.sidebar.slider(
            "節（GW）", _gw_min, _gw_max, (_gw_min, _gw_max),
            help=f"取得済みGW: {_gw_min}〜{_gw_max}節。範囲を絞ると指定期間のデータのみで集計します。"
        )
    if _gw_range != (_gw_min, _gw_max):
        df_g_raw = df_g_raw[
            df_g_raw[_gw_col].between(_gw_range[0], _gw_range[1])
        ].copy()
        # players_raw は累積値なので、GW別データを使って累積を再計算
        _num_cols_agg = [
            "goals_scored","assists","expected_goals","expected_assists",
            "expected_goal_involvements","expected_goals_conceded","goals_conceded",
            "saves","clean_sheets","yellow_cards","red_cards","bonus",
            "minutes","tackles","recoveries","clearances_blocks_interceptions",
            "defensive_contribution",
        ]
        _agg_dict = {c: "sum" for c in _num_cols_agg if c in df_g_raw.columns}
        # influence/creativity/threat はGW別データにある
        for _ic in ["influence","creativity","threat","ict_index"]:
            if _ic in df_g_raw.columns:
                _agg_dict[_ic] = "sum"
        df_p_raw = (
            df_g_raw.groupby("element")
            .agg(_agg_dict)
            .reset_index()
            .rename(columns={"element": "id"})
        )
        # web_name / element_type / team をplayers_rawからマージ
        if df_p_raw is not None and "web_name" not in df_p_raw.columns:
            _orig = load_season(season)[0]
            if _orig is not None:
                _meta = _orig[["id","web_name","element_type","team","now_cost"]].copy()
                df_p_raw = df_p_raw.merge(_meta, on="id", how="left")
        st.sidebar.caption(f"GW {_gw_range[0]}〜{_gw_range[1]}節のデータで集計中")
    st.sidebar.markdown("---")

if df_p_raw is None or df_g_raw is None:
    st.error(f"""
**データ取得に失敗しました**

まず **ページをリロード（再読み込み）** してください。
GitHub へのアクセスが一時的に失敗することがあります。

▶ それでも失敗する場合は、シーズンを別のものに切り替えてみてください。  
▶ 左上の **>>** をタップするとシーズン選択が開きます。
""")
    st.info("💡 スマホの場合: 画面左上の **>>** をタップするとメニューが開きます")
    st.stop()

team_id_map = {}
if df_t_raw is not None and "id" in df_t_raw.columns and "name" in df_t_raw.columns:
    team_id_map = dict(zip(df_t_raw["id"], df_t_raw["name"]))

df_players  = prep_players(df_p_raw, team_id_map)
df_teams    = build_team_stats(df_g_raw, team_id_map)
df_ts       = build_team_timeseries(df_g_raw, team_id_map)

# API-Football データ: 事前取得済みJSONをGitHubから読み込む
# データ取得方法: fetch_api_stats.py を参照
def load_apf_json(season_str: str) -> dict:
    """GitHubに保存済みのAPI統計JSONを読み込む。session_stateでキャッシュ。"""
    cache_key = f"apf_json_{season_str}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        repo_user = st.secrets.get("GITHUB_USER", "")
        repo_name = st.secrets.get("GITHUB_REPO", "")
    except Exception:
        return {}
    if not repo_user or not repo_name:
        return {}

    url = (f"https://raw.githubusercontent.com/{repo_user}/{repo_name}"
           f"/main/api_stats_{season_str}.json")
    r = _get(url)
    if r:
        try:
            data = r.json()
            if data:
                st.session_state[cache_key] = data
            return data
        except Exception:
            return {}
    return {}

_apf_json = load_apf_json(season)
if _apf_json:
    df_teams = build_apf_team_stats(_apf_json, df_teams)
    st.sidebar.success(f"⚡ {len(_apf_json)}試合分スタッツ読込済")
else:
    st.sidebar.caption("⚡ APIスタッツ未設定 (fetch_api_stats.pyで取得→GitHubにコミット)")

tcmap       = team_color_map(df_teams["team_name"].tolist())

# ── Header ────────────────────────────────────────────────────────────────────
_view_label = "Team" if "Team" in page else "Player"
_subtitle   = f"{season}  ·  Custom Metrics Builder  ·  {_view_label} View"

# SVGヘッダー（CSSに依存しないため確実に表示）
_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="90" viewBox="0 0 900 90">
  <rect width="900" height="90" rx="10" fill="#0f172a"/>
  <rect width="6" height="90" rx="3" fill="#1a5c36"/>
  <text x="22" y="52" font-family="Arial Black,Arial,sans-serif"
        font-size="32" font-weight="900" letter-spacing="2"
        fill="#ffffff">EPL Analytics</text>
  <text x="22" y="74" font-family="Arial,sans-serif"
        font-size="11" font-weight="600" fill="#94a3b8">{_subtitle}</text>
</svg>"""
import base64 as _b64
_svg_b64 = _b64.b64encode(_svg.encode()).decode()
st.markdown(
    f'<img src="data:image/svg+xml;base64,{_svg_b64}" style="width:100%;display:block">',
    unsafe_allow_html=True
)
st.markdown("<div style='height:4px;background:linear-gradient(90deg,#1a5c36,#c45c00,#0077aa,transparent);border-radius:2px;margin-bottom:.5rem'></div>", unsafe_allow_html=True)
# スマホ向け案内（CSSメディアクエリで制御）
st.markdown("""
<style>
.mobile-hint{display:none}
@media(max-width:640px){
  .mobile-hint{
    display:block;
    background:#1e2d3d;
    color:#e2e8f0 !important;
    font-size:.82rem;
    padding:.5rem .9rem;
    border-radius:6px;
    margin-bottom:.5rem;
    border-left:3px solid #1a5c36;
  }
}
</style>
<div class="mobile-hint">
  ☰ 左上の <b>&gt;&gt;</b> をタップするとシーズン選択・設定が開きます
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if "Team" in page:

    # ── 利用可能指標一覧 ──────────────────────────────────────────────────────
    TEAM_METRICS = {
        # label: (col, description, category)
        "Points":             ("points",         "Total points (W=3 D=1 L=0)",       "Results"),
    "Wins":               ("wins",           "Number of wins",                    "Results"),
    "Draws":              ("draws",          "Number of draws",                   "Results"),
    "Losses":             ("losses",         "Number of losses",                  "Results"),
    "Points per Match":   ("pts_per_match",  "Points / matches played",           "Results"),
    "Goals Scored":       ("goals_scored",   "Total goals scored",               "Attack"),
        "Goals Conceded":     ("goals_conceded", "Total goals conceded",              "Defense"),
        "Goal Difference":    ("goal_diff",      "Goals scored − Goals conceded",     "Attack"),
        "xG (Total)":         ("xG",             "Expected goals scored",             "Attack"),
        "xGC (Total)":        ("xGC",            "Expected goals conceded",           "Defense"),
        "xG per Match":       ("xG_per_match",   "xG / matches played",               "Attack"),
        "xGC per Match":      ("xGC_per_match",  "xGC / matches played",              "Defense"),
        "xG Difference":      ("xG_diff",        "xG − xGC",                          "Attack"),
        "Goal Luck (Attack)": ("goal_luck",      "Goals − xG (positive = clinical)",  "Luck"),
        "Def Luck":           ("def_luck",       "xGC − GA (positive = fortunate)",   "Luck"),
        "Clean Sheets":       ("clean_sheets",   "Number of clean sheets",            "Defense"),
        "CS per Match":       ("cs_per_match",   "Clean sheets / matches",            "Defense"),
        "Saves":              ("saves",          "Total saves by GK",                 "Defense"),
        "Saves per Match":    ("saves_per_match","Saves / matches",                   "Defense"),
        "GF per Match":       ("gf_per_match",   "Goals scored / matches",            "Attack"),
        "GA per Match":       ("ga_per_match",   "Goals conceded / matches",          "Defense"),
        "Assists":            ("assists",        "Total assists",                      "Attack"),
        "xA (Total)":         ("xA",             "Expected assists",                   "Attack"),
        "Creativity":         ("creativity",     "FPL Creativity (chance creation)",   "Attack"),
        "Threat":             ("threat",         "FPL Threat (goal threat)",           "Attack"),
        "Yellow Cards":       ("yellow_cards",   "Yellow cards accumulated",           "Discipline"),
        "Red Cards":          ("red_cards",      "Red cards accumulated",              "Discipline"),
    "Tackles (team)":     ("tackles",        "Total team tackles",                 "Defense"),
    "Recoveries (team)":  ("recoveries",     "Total team ball recoveries",         "Defense"),
    "CBI (team)":         ("cbi",            "Team clearances+blocks+interceptions","Defense"),
    "Def Contribution":   ("def_contribution","Team defensive contribution score", "Defense"),
    # ── API-Football 取得指標（要APIキー）──────────────────────────────────────
    "Shots/Match ⚡":     ("shots_pm",        "Total shots per match (API-Football)", "Attack⚡"),
    "Shots on Target/M ⚡":("shots_on_tgt_pm","Shots on target per match",            "Attack⚡"),
    "Shots in Box/M ⚡":  ("shots_inbox_pm",  "Shots inside box per match",           "Attack⚡"),
    "Shot Conversion % ⚡":  ("shot_conversion",      "Goals / Shots × 100",                    "Attack⚡"),
    "Shots on Tgt Rate % ⚡": ("shots_on_tgt_rate",   "Shots on target / Total shots × 100",    "Attack⚡"),
    "Shots Against/M ⚡":     ("shots_against_pm",     "Shots conceded per match",                "Defense⚡"),
    "Shots on Tgt Ag/M ⚡":   ("shots_on_tgt_ag_pm",  "On-target shots conceded per match",      "Defense⚡"),
    "Shots in Box Ag/M ⚡":   ("shots_inbox_ag_pm",   "Inside-box shots conceded per match",     "Defense⚡"),
    "Shots on Tgt Ag Rate% ⚡":("shots_on_tgt_ag_rate","Opponent shots on target rate (%) ",     "Defense⚡"),
    "Possession % ⚡":    ("possession",       "Average ball possession (%)",          "Attack⚡"),
    "Corners/Match ⚡":   ("corners_pm",       "Corners per match",                   "Attack⚡"),
    "Fouls/Match ⚡":     ("fouls_pm",         "Fouls committed per match",           "Discipline⚡"),
    "Offsides/Match ⚡":  ("offsides_pm",      "Offsides per match",                  "Attack⚡"),
    "Pass Accuracy % ⚡": ("pass_acc",         "Pass completion rate (%)",             "Attack⚡"),
    "Passes/Match ⚡":    ("passes_pm",        "Total passes per match",               "Attack⚡"),
    }
    all_metric_labels = list(TEAM_METRICS.keys())
    metric_cols = {v[0]: k for k, v in TEAM_METRICS.items()}

    tab_overview, tab_rank_t, tab_trend, tab_radar, tab_scatter, tab_pca, tab_custom = st.tabs([
        "📋 Available Metrics",
        "🏆 Rankings",
        "📈 Time Series",
        "🕸️ Radar",
        "⊕ 2-Axis Plot",
        "📐 PCA",
        "🔧 Custom Metric",
    ])

    # ── Tab 0: Available Metrics ──────────────────────────────────────────────
    with tab_overview:
        st.markdown("## Available Team Metrics")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        # 指標の読み方ガイド
        with st.expander("📖 指標の読み方・使い方ガイド", expanded=False):
            st.markdown("""
            #### FPL独自指標（ICT）の意味
            | 指標 | 意味 | 高い選手の特徴 |
            |------|------|--------------|
            | **Creativity** | チャンスメイクの量と質。キーパス・クロス・スルーパスを評価 | トップ下、クリエイティブなMF（例: デ・ブライネ） |
            | **Threat** | ゴールへの脅威。シュート数・シュート位置・PA内行動を評価 | ストライカー、得点力が高いFW（例: サラー） |
            | **Influence** | 試合全体への関与度。ボールタッチ・デュエル・守備行動を包括的に評価 | キャプテン的存在、試合を動かすMF |
            | **ICT Index** | Influence + Creativity + Threat の合成スコア。FPLの総合評価 | 全能型MF・FW（例: サラー、デ・ブライネ） |
            | **xG (Expected Goals)** | そのシュートがゴールになる統計的確率の合計。「本来の得点力」を示す | — |
            | **xA (Expected Assists)** | アシストパスのxG合計。「パスの質」を示す | — |
            | **xGI** | xG + xA の合計。攻撃関与度の総合値 | — |
            | **xGC** | 被xG。出場中に相手に与えた得点期待値。低いほど守備が良い | — |

            #### 使い方のヒント
            - **レーダーチャート**: 複数チームを同じ指標で比較。標準化済みなので公平な比較ができます
            - **2軸散布図**: 例）「xG per Match」vs「xGC per Match」→ 攻守バランスの把握
            - **PCA**: 複数指標を入れると、隠れた「プレースタイル軸」を発見できます
            - **カスタム指標**: 例）「攻撃力 = xG×1.5 + Shots×0.5 - xGC×1.0」のように自由に設計
            - **⚡マーク指標**: API-Footballから取得。Streamlit SecretsにAPIキーを設定すると使えます
            """)

        cats = sorted(set(v[2] for v in TEAM_METRICS.values()))
        for cat in cats:
            st.markdown(f"### {cat}")
            rows = [(lbl, TEAM_METRICS[lbl][0], TEAM_METRICS[lbl][1])
                    for lbl in TEAM_METRICS if TEAM_METRICS[lbl][2] == cat]
            df_cat = pd.DataFrame(rows, columns=["Metric","Column","Description"])
            df_cat["Min"]  = df_cat["Column"].apply(lambda c: f"{df_teams[c].min():.2f}" if c in df_teams else "—")
            df_cat["Max"]  = df_cat["Column"].apply(lambda c: f"{df_teams[c].max():.2f}" if c in df_teams else "—")
            df_cat["Mean"] = df_cat["Column"].apply(lambda c: f"{df_teams[c].mean():.2f}" if c in df_teams else "—")
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

    # ── Tab 1: Radar ─────────────────────────────────────────────────────────
    with tab_rank_t:
        st.markdown("## Team Rankings")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_ra, col_rb = st.columns([1, 3])
        with col_ra:
            rank_metric_t = st.selectbox("指標を選択", all_metric_labels, key="t_rank_metric",
                                          index=all_metric_labels.index("Points") if "Points" in all_metric_labels else 0)
            _t_low_is_good = {
                "goals_conceded","ga_per_match","xGC","xGC_per_match",
                "yellow_cards","red_cards","losses","fouls_pm","offsides_pm",
                "goal_luck","def_luck",
            }
            _t_raw = TEAM_METRICS[rank_metric_t][0]
            _t_default_asc = _t_raw in _t_low_is_good
            sort_asc_t = st.toggle("低い順に表示", value=_t_default_asc, key="t_rank_asc",
                                    help="失点・被xG・イエローカードなど低い方が良い指標に使います")
            show_n_t   = st.radio("表示数", [10, 15, 20], horizontal=True, key="t_rank_n")

        with col_rb:
            t_col = TEAM_METRICS[rank_metric_t][0]
            if t_col not in df_teams.columns:
                st.warning(f"'{rank_metric_t}' はこのシーズンでは利用できません")
            else:
                df_trank = (df_teams[["team_name", t_col]]
                            .copy()
                            .sort_values(t_col, ascending=sort_asc_t)
                            .head(int(show_n_t))
                            .reset_index(drop=True))
                df_trank.index += 1

                # パーセンタイル
                _pool_t = df_teams[t_col].dropna()
                if sort_asc_t:
                    df_trank["Percentile"] = df_trank[t_col].apply(
                        lambda v: f"Bottom {int(((_pool_t <= v).mean())*100)}%"
                    )
                else:
                    df_trank["Percentile"] = df_trank[t_col].apply(
                        lambda v: f"Top {100 - int((_pool_t <= v).mean()*100)}%"
                    )

                df_trank_show = df_trank.rename(columns={t_col: rank_metric_t})

                # スタイル
                try:
                    styled = df_trank_show.style.background_gradient(
                        subset=[rank_metric_t], cmap="RdYlGn_r" if sort_asc_t else "RdYlGn"
                    ).format({rank_metric_t: "{:.3f}"})
                    st.dataframe(styled, use_container_width=True)
                except Exception:
                    st.dataframe(df_trank_show, use_container_width=True)

                # 横棒グラフ
                fig_tr, ax_tr = plt.subplots(figsize=(7, max(3, len(df_trank) * 0.45)))
                fig_tr.patch.set_facecolor("#ffffff")
                ax_tr.set_facecolor("#f8f9fa")
                ax_tr.grid(axis="x", color="#e0e0e0", lw=0.6, zorder=0)
                _vals  = df_trank[t_col].values
                _teams = df_trank["team_name"].values
                _colors = [tcmap.get(t, "#64748b") for t in _teams]
                bars = ax_tr.barh(range(len(_teams)), _vals, color=_colors, alpha=0.85, edgecolor="#374151", lw=0.4)
                ax_tr.set_yticks(range(len(_teams)))
                ax_tr.set_yticklabels(_teams, fontsize=10, color="#1a1a2e")
                ax_tr.set_xlabel(rank_metric_t, color="#333333")
                ax_tr.set_title(f"{rank_metric_t} Ranking", color="#1a1a2e", fontweight="bold")
                ax_tr.tick_params(colors="#333333")
                ax_tr.spines["bottom"].set_color("#cccccc")
                ax_tr.spines["left"].set_color("#cccccc")
                ax_tr.invert_yaxis()
                # 値ラベル
                for i, (bar, val) in enumerate(zip(bars, _vals)):
                    ax_tr.text(val, i, f" {val:.2f}", va="center", fontsize=8, color="#1a1a2e")
                plt.tight_layout()
                st.pyplot(fig_tr, use_container_width=True)

    with tab_trend:
        st.markdown("## Time Series")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        # 指標定義（時系列で使える指標のみ）
        # API-Football の時系列データを構築
        def _build_apf_ts(apf_json: dict) -> "pd.DataFrame":
            """api_stats JSONの_metaからGW×チームの時系列を構築"""
            rows = []
            for fid, sides in apf_json.items():
                meta = sides.get("_meta", {})
                gw   = meta.get("gw", 0)
                if not gw:
                    continue
                for side, opp in [("home","away"), ("away","home")]:
                    s = sides.get(side, {})
                    o = sides.get(opp, {})
                    if not s or not s.get("team_name"):
                        continue
                    rows.append({
                        "team_name":     _normalize_team_name(s["team_name"]),
                        "GW":            gw,
                        "shots":         s.get("Total Shots"),
                        "shots_on_tgt":  s.get("Shots on Goal"),
                        "shots_inbox":   s.get("Shots insidebox"),
                        "possession":    s.get("Ball Possession"),
                        "corners":       s.get("Corner Kicks"),
                        "fouls":         s.get("Fouls"),
                        "shots_ag":      o.get("Total Shots"),
                        "shots_on_tgt_ag": o.get("Shots on Goal"),
                    })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            for c in df.columns:
                if c not in ["team_name","GW"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.groupby(["team_name","GW"]).mean(numeric_only=True).reset_index()
            # 各比率を計算
            df["shots_on_tgt_rate"] = (df["shots_on_tgt"] / df["shots"].replace(0, np.nan) * 100).round(1)
            return df

        _apf_ts = _build_apf_ts(_apf_json) if _apf_json else pd.DataFrame()
        _has_apf_ts = not _apf_ts.empty

        TS_METRICS = {
            "xG (per match)":              ("xG",              False, "vaastav"),
            "xGC (per match)":             ("xGC",             False, "vaastav"),
            "Goals (per match)":           ("gf",              False, "vaastav"),
            "Goals Conceded (pm)":         ("ga",              False, "vaastav"),
            "Points (per match)":          ("pts",             False, "vaastav"),
            "Yellow Cards (pm)":           ("yellow_cards",    False, "vaastav"),
            "Creativity (pm)":             ("creativity",      False, "vaastav"),
            "Threat (pm)":                 ("threat",          False, "vaastav"),
            "Influence (pm)":              ("influence",       False, "vaastav"),
            "xG (cumulative)":             ("xG_cum",          True,  "vaastav"),
            "xGC (cumulative)":            ("xGC_cum",         True,  "vaastav"),
            "Goals (cumulative)":          ("gf_cum",          True,  "vaastav"),
            "Points (cumulative)":         ("pts_cum",         True,  "vaastav"),
            "Creativity (cumulative)":     ("creativity_cum",  True,  "vaastav"),
            "Threat (cumulative)":         ("threat_cum",      True,  "vaastav"),
            "Shots/Match ⚡":              ("shots",           False, "apf"),
            "Shots on Target/M ⚡":        ("shots_on_tgt",    False, "apf"),
            "Possession % ⚡":             ("possession",      False, "apf"),
            "Corners/Match ⚡":            ("corners",         False, "apf"),
            "Fouls/Match ⚡":              ("fouls",           False, "apf"),
            "Shots Against/M ⚡":          ("shots_ag",        False, "apf"),
            "Shots on Tgt Ag/M ⚡":        ("shots_on_tgt_ag", False, "apf"),
            "Shots on Tgt Rate % ⚡":      ("shots_on_tgt_rate",False,"apf"),
        }
        # APIデータがない場合は⚡指標をリストから除外
        if not _has_apf_ts:
            TS_METRICS = {k: v for k, v in TS_METRICS.items() if v[2] != "apf"}

        col_ta, col_tb = st.columns([1, 3])
        with col_ta:
            ts_metric   = st.selectbox("指標", list(TS_METRICS.keys()),
                                        index=list(TS_METRICS.keys()).index("Points (cumulative)"),
                                        key="ts_metric")
            ts_teams    = st.multiselect("チームを選択（空 = 全チーム）",
                                          sorted(df_ts["team_name"].unique().tolist()),
                                          key="ts_teams")
            ts_ma       = st.slider("移動平均（試合数、1=なし）", 1, 5, 1, key="ts_ma",
                                     help="直近N試合の平均を表示します（累積指標には不要）")
            ts_show_avg = st.toggle("リーグ平均を表示", value=True, key="ts_avg")

        with col_tb:
            ts_col, ts_is_cum, ts_src = TS_METRICS[ts_metric]
            _df_base = _apf_ts if ts_src == "apf" else df_ts
            if _df_base.empty or ts_col not in _df_base.columns:
                st.warning(f"'{ts_metric}' はこのシーズンでは利用できません。"
                           + (" API-Football JSONを再取得してGitHubを更新してください。" if ts_src=="apf" else ""))
            else:
                _df_plot = _df_base.copy()
                if ts_teams:
                    _df_plot = _df_plot[_df_plot["team_name"].isin(ts_teams)]

                # 移動平均（累積指標には適用しない）
                if ts_ma > 1 and not ts_is_cum:
                    _df_plot = _df_plot.copy()
                    _df_plot[ts_col] = (
                        _df_plot.groupby("team_name")[ts_col]
                        .transform(lambda x: x.rolling(ts_ma, min_periods=1).mean())
                    )

                # GW単位に集約
                if ts_col in _df_plot.columns:
                    _df_gw = (_df_plot.groupby(["team_name","GW"])[ts_col]
                              .mean().reset_index() if ts_src == "apf"
                              else (_df_plot.groupby(["team_name","GW"])[ts_col]
                                    .sum().reset_index() if not ts_is_cum
                                    else _df_plot.groupby(["team_name","GW"])[ts_col]
                                    .last().reset_index()))
                else:
                    st.warning(f"列 '{ts_col}' が見つかりません")
                    st.stop()

                fig_ts, ax_ts = plt.subplots(figsize=(9, 5))
                fig_ts.patch.set_facecolor("#ffffff")
                ax_ts.set_facecolor("#f8f9fa")
                ax_ts.grid(axis="both", color="#e0e0e0", lw=0.5, zorder=0)

                # チームごとに折れ線
                _all_teams = _df_gw["team_name"].unique()
                for tn in sorted(_all_teams):
                    _td = _df_gw[_df_gw["team_name"] == tn].sort_values("GW")
                    _c  = tcmap.get(tn, "#64748b")
                    ax_ts.plot(_td["GW"], _td[ts_col],
                               color=_c, lw=1.8, marker="o", markersize=3,
                               label=tn, alpha=0.85, zorder=3)

                # リーグ平均
                if ts_show_avg:
                    if ts_src == "apf":
                        _avg_df = _df_base if ts_col in _df_base.columns else pd.DataFrame()
                        _avg = _avg_df.groupby("GW")[ts_col].mean() if not _avg_df.empty else pd.Series(dtype=float)
                    elif ts_is_cum:
                        _avg = df_ts.groupby(["team_name","GW"])[ts_col].last().reset_index().groupby("GW")[ts_col].mean() if ts_col in df_ts.columns else pd.Series(dtype=float)
                    else:
                        _avg = df_ts.groupby("GW")[ts_col].mean() if ts_col in df_ts.columns else pd.Series(dtype=float)
                    ax_ts.plot(_avg.index, _avg.values,
                               color="#666666", lw=2, ls="--",
                               label="League Avg", alpha=0.7, zorder=4)

                ax_ts.set_xlabel("GW（節）", color="#333333", fontsize=10)
                ax_ts.set_ylabel(ts_metric, color="#333333", fontsize=10)
                ax_ts.set_title(ts_metric, color="#1a1a2e", fontweight="bold", fontsize=12)
                ax_ts.tick_params(colors="#333333")
                for spine in ax_ts.spines.values():
                    spine.set_color("#cccccc")

                # 凡例（チーム数が多い場合は右外に）
                _n_teams = len(_all_teams)
                if _n_teams <= 6:
                    ax_ts.legend(fontsize=8, facecolor="#ffffff",
                                 edgecolor="#cccccc", labelcolor="#1a1a2e")
                elif _n_teams <= 12:
                    ax_ts.legend(fontsize=7, facecolor="#ffffff", edgecolor="#cccccc",
                                 labelcolor="#1a1a2e", bbox_to_anchor=(1.01,1),
                                 loc="upper left", ncol=1)
                else:
                    ax_ts.legend(fontsize=6.5, facecolor="#ffffff", edgecolor="#cccccc",
                                 labelcolor="#1a1a2e", bbox_to_anchor=(1.01,1),
                                 loc="upper left", ncol=2)

                plt.tight_layout()
                st.pyplot(fig_ts, use_container_width=True)
                _ma_note = f"（{ts_ma}試合移動平均）" if ts_ma > 1 and not ts_is_cum else ""
                st.caption(f"横軸 = GW（節）　縦軸 = {ts_metric}{_ma_note}　破線 = リーグ平均")

    with tab_radar:
        st.markdown("## Radar Chart — Team Comparison")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,2])
        with col_a:
            sel_teams   = st.multiselect("Select Teams (2–6)",
                                          sorted(df_teams["team_name"]),
                                          default=sorted(df_teams["team_name"])[:6])
            sel_metrics = st.multiselect("Select Metrics (3–8)",
                                          all_metric_labels,
                                          default=["xG per Match","xGC per Match",
                                                   "CS per Match","Saves per Match",
                                                   "Goal Luck (Attack)","Def Luck"])
        with col_b:
            if len(sel_teams) >= 2 and len(sel_metrics) >= 3:
                cols_sel = [TEAM_METRICS[m][0] for m in sel_metrics]
                df_r = df_teams[df_teams["team_name"].isin(sel_teams)].set_index("team_name")
                fig_r = radar(df_r, cols_sel, sel_metrics,
                              "Team Comparison Radar", z_pool=df_teams.set_index("team_name"))
                st.pyplot(fig_r, use_container_width=True)
                st.caption("Outer = higher EPL-wide percentile. Axes show position relative to all 20 clubs.")
            else:
                st.info("Select at least 2 teams and 3 metrics.")

    # ── Tab 2: 2-Axis Scatter ─────────────────────────────────────────────────
    with tab_scatter:
        st.markdown("## 2-Axis Club Positioning")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,3])
        with col_a:
            x_label = st.selectbox("X Axis", all_metric_labels,
                                   index=all_metric_labels.index("xG per Match"))
            y_label = st.selectbox("Y Axis", all_metric_labels,
                                   index=all_metric_labels.index("xGC per Match"))
            show_regression = st.toggle("📈 回帰分析を表示", value=False,
                                         help="回帰直線・Pearson r・R²・Spearman ρを追加表示します")
        with col_b:
            x_col = TEAM_METRICS[x_label][0]
            y_col = TEAM_METRICS[y_label][0]

            if show_regression:
                # 回帰分析付き散布図
                from scipy.stats import pearsonr, spearmanr, linregress
                _x = df_teams[x_col].fillna(0).values.astype(float)
                _y = df_teams[y_col].fillna(0).values.astype(float)
                _mask = np.isfinite(_x) & np.isfinite(_y)
                _x, _y = _x[_mask], _y[_mask]
                _df_reg = df_teams[_mask].copy()

                slope, intercept, r_val, p_val, _ = linregress(_x, _y)
                r_sq   = r_val ** 2
                sp_r, sp_p = spearmanr(_x, _y)

                fig_s = scatter_2d(_df_reg, x_col, y_col, "team_name",
                                   f"{x_label}  vs  {y_label}", tcmap)
                ax_s  = fig_s.get_axes()[0]

                # 回帰直線
                _x_line = np.linspace(_x.min(), _x.max(), 100)
                _y_line = slope * _x_line + intercept
                ax_s.plot(_x_line, _y_line, color="#f4a261",
                          lw=2, ls="-", zorder=4, label="Regression line")

                # 外れ値（残差が大きいチーム）を強調
                _y_pred = slope * _x + intercept
                _resid  = _y - _y_pred
                _thresh = 1.5 * np.std(_resid)
                _df_reg = _df_reg.reset_index(drop=True)
                for _i, _row in _df_reg.iterrows():
                    if abs(_resid[_i]) >= _thresh:
                        ax_s.annotate(
                            f"← {_row['team_name']}",
                            (_x[_i], _y[_i]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color="#f4d03f", fontweight="bold"
                        )

                # 数式テキスト
                _eq = f"y = {slope:.3f}x + {intercept:.3f}"
                _stats = f"r = {r_val:.3f}  R² = {r_sq:.3f}  p = {p_val:.3f}"
                ax_s.text(0.02, 0.97, f"{_eq}\n{_stats}",
                          transform=ax_s.transAxes,
                          va="top", ha="left", fontsize=9,
                          color="#e2e8f0",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="#1f2937", ec="#374151", alpha=0.85))
                ax_s.legend(facecolor="#1f2937", edgecolor="#374151",
                            labelcolor="#e2e8f0", fontsize=9)
                st.pyplot(fig_s, use_container_width=True)

                # 統計サマリー
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Pearson r",   f"{r_val:+.3f}",
                              help="線形相関係数。±1に近いほど強い線形関係")
                col_r2.metric("R²",          f"{r_sq:.3f}",
                              help=f"決定係数。{x_label}が{y_label}の変動の{r_sq*100:.1f}%を説明")
                col_r3.metric("Spearman ρ",  f"{sp_r:+.3f}",
                              help="順位相関係数。外れ値に頑健")

                _sig = "p < 0.001" if p_val < 0.001 else (f"p = {p_val:.3f}")
                _strength = ("強い" if abs(r_val) > 0.7 else
                             "中程度の" if abs(r_val) > 0.4 else "弱い")
                _direction = "正" if r_val > 0 else "負"
                st.caption(
                    f"{_sig} | {x_label}と{y_label}の間に**{_strength}{_direction}の相関**があります。"
                    f"回帰式: {_eq}  |  黄色ラベル = 残差が大きい外れ値チーム"
                )
            else:
                fig_s = scatter_2d(df_teams, x_col, y_col, "team_name",
                                   f"{x_label}  vs  {y_label}", tcmap)
                st.pyplot(fig_s, use_container_width=True)
                st.caption("破線 = リーグ平均。📈 回帰分析トグルをONにすると回帰直線・相関係数を表示します。")

    # ── Tab 3: PCA ────────────────────────────────────────────────────────────
    with tab_pca:
        st.markdown("## Principal Component Analysis")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.info("ℹ️ EPLは20チームのみのため、投入指標数は6以下を推奨します。それ以上では偶然の構造を拾いやすくなります。")

        sel_pca = st.multiselect("Metrics for PCA (3–6 recommended)",
                                  all_metric_labels,
                                  default=["xG per Match","xGC per Match","GF per Match",
                                           "GA per Match","CS per Match","Goal Luck (Attack)"])
        if len(sel_pca) >= 3:
            pca_cols = [TEAM_METRICS[m][0] for m in sel_pca]
            fig_pca, scores, loadings, Vt = pca_plot(
                df_teams, pca_cols, "team_name",
                "Club Positioning — PC1 vs PC2", tcmap
            )
            st.pyplot(fig_pca, use_container_width=True)

            # PC寄与度サマリー
            col1, col2 = st.columns(2)
            load_df = pd.DataFrame({"Metric": sel_pca,
                                    "PC1 Loading": loadings[:,0].round(3),
                                    "PC2 Loading": loadings[:,1].round(3)})
            load_df["PC1 Abs"] = load_df["PC1 Loading"].abs()
            load_df["PC2 Abs"] = load_df["PC2 Loading"].abs()

            with col1:
                st.markdown("**PC 1 — Top contributing metrics**")
                st.caption("High PC1 score → right on scatter plot")
                st.dataframe(load_df.sort_values("PC1 Abs",ascending=False)
                             [["Metric","PC1 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",subset=["PC1 Loading"],vmin=-1,vmax=1)
                             .format({"PC1 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)
            with col2:
                st.markdown("**PC 2 — Top contributing metrics**")
                st.caption("High PC2 score → top on scatter plot")
                st.dataframe(load_df.sort_values("PC2 Abs",ascending=False)
                             [["Metric","PC2 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",subset=["PC2 Loading"],vmin=-1,vmax=1)
                             .format({"PC2 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)

            # PC1スコアをランキング表示
            df_pca_out = df_teams[["team_name"] + pca_cols].copy()
            df_pca_out["PC1_score"] = scores[:,0].round(3)
            df_pca_out["PC2_score"] = scores[:,1].round(3)
            st.markdown("**Club scores on each principal component**")
            st.dataframe(
                df_pca_out[["team_name","PC1_score","PC2_score"]]
                .sort_values("PC1_score", ascending=False)
                .style.background_gradient(subset=["PC1_score","PC2_score"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Select at least 3 metrics.")

    # ── Tab 4: Custom Metric ──────────────────────────────────────────────────
    with tab_custom:
        st.markdown("## 🔧 Build Your Own Metric")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.markdown("""
        **使い方:** 指標を選んで係数を設定するだけでオリジナル指標が作れます。
        係数は手動で設定するか、「Correlation with Points」ボタンで勝ち点との相関から自動提案できます。
        """)

        metric_name = st.text_input("Metric Name", value="My Custom Metric")

        # 指標選択テーブル
        n_rows = st.number_input("Number of metrics to combine", 1, 8, 3)
        components = []
        col_h1, col_h2, col_h3, col_h4 = st.columns([3,2,2,3])
        col_h1.markdown("**Metric**")
        col_h2.markdown("**Weight**")
        col_h3.markdown("**Direction**")
        col_h4.markdown("**Formula**")

        for i in range(int(n_rows)):
            c1,c2,c3,c4 = st.columns([3,2,2,3])
            with c1:
                m = st.selectbox(f"Metric {i+1}", all_metric_labels,
                                  key=f"cm_{i}",
                                  index=min(i, len(all_metric_labels)-1))
            with c2:
                w = st.number_input("Weight", value=1.0, step=0.1, key=f"cw_{i}")
            with c3:
                d = st.selectbox("", ["+", "−"], key=f"cd_{i}")
            with c4:
                sign = 1 if d == "+" else -1
                col = TEAM_METRICS[m][0]
                st.markdown(f"<span style='color:{C['muted']};font-size:.8rem'>"
                            f"`{'+' if sign>0 else '-'}{abs(w):.1f} × {col}`</span>",
                            unsafe_allow_html=True)
            components.append((m, col, w, sign))

        # 相関ベース自動係数
        if st.button("🔄 Auto-weight by correlation with xG Difference"):
            for i, (m, col, w, sign) in enumerate(components):
                if col in df_teams.columns:
                    r, _ = pearsonr(df_teams[col].fillna(0), df_teams["xG_diff"].fillna(0))
                    st.write(f"  {m}: r = {r:+.3f} → suggested weight = {abs(r):.2f} "
                             f"({'+'  if r > 0 else '−'})")

        # 計算・表示
        if st.button("▶  Calculate & Rank", type="primary"):
            df_custom = df_teams[["team_name"] + [c[1] for c in components]].copy()
            df_custom[metric_name] = sum(
                row[2] * row[3] * df_custom[row[1]].fillna(0)
                for row in components
            )
            df_custom = df_custom.sort_values(metric_name, ascending=False).reset_index(drop=True)
            df_custom.index += 1

            col_t, col_c = st.columns([2,3])
            with col_t:
                st.markdown(f"**Rankings: {metric_name}**")
                show_cols = ["team_name", metric_name] + [c[1] for c in components]
                st.dataframe(
                    df_custom[show_cols].style
                    .background_gradient(subset=[metric_name], cmap="RdYlGn")
                    .format({metric_name: "{:.3f}"}),
                    use_container_width=True
                )
            with col_c:
                # 簡易棒グラフ
                fig_bar, ax_bar = plt.subplots(figsize=(6,6))
                apply_dark_style(fig_bar, ax_bar)
                colors_bar = [tcmap.get(t, C["sky"]) for t in df_custom["team_name"]]
                ax_bar.barh(df_custom["team_name"][::-1],
                            df_custom[metric_name][::-1],
                            color=colors_bar[::-1], edgecolor="#374151", lw=.4)
                ax_bar.axvline(df_custom[metric_name].mean(),
                               color=C["amber"], ls="--", lw=1, label="Average")
                ax_bar.set_xlabel(metric_name, color=C["muted"])
                ax_bar.set_title(f"Club Ranking: {metric_name}",
                                 color=C["chalk"], fontweight="bold")
                ax_bar.legend(facecolor="#1f2937", edgecolor="#374151",
                              labelcolor=C["chalk"])
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    PLAYER_METRICS = {
        "Goals":          ("goals_scored",                 "Total goals",                "Attack"),
        "Assists":        ("assists",                      "Total assists",              "Attack"),
        "xG":             ("expected_goals",               "Expected goals",             "Attack"),
        "xA":             ("expected_assists",             "Expected assists",           "Attack"),
        "xGI":            ("expected_goal_involvements",   "xG + xA",                   "Attack"),
        "Threat":         ("threat",                       "FPL Threat score",                                  "FPL Threat score",           "Attack"),
        "Creativity":     ("creativity",                   "FPL Creativity score",                          "FPL Creativity score",       "Playmaking"),
        "Influence":      ("influence",                    "FPL Influence score",                            "FPL Influence score",        "Playmaking"),
        "ICT Index":      ("ict_index",                    "FPL ICT combined",           "Playmaking"),
        "xGC":            ("expected_goals_conceded",      "Expected goals conceded",    "Defense"),
        "Saves":          ("saves",                        "Total saves (GK)",           "Defense"),
        "Clean Sheets":   ("clean_sheets",                 "Clean sheets",               "Defense"),
        "Goals Conceded": ("goals_conceded",               "Goals conceded while on",    "Defense"),
        "Goal Luck":      ("goal_luck",                    "Goals − xG",                 "Luck"),
        "Def Luck":       ("def_luck",                     "xGC − Goals conceded",       "Luck"),
        "Minutes":        ("minutes",                      "Total minutes played",       "Availability"),
        "Starts":         ("starts",                       "Number of starts",           "Availability"),
        "Yellow Cards":   ("yellow_cards",                 "Yellow cards",               "Discipline"),
        "Red Cards":      ("red_cards",                    "Red cards",                  "Discipline"),
        "Tackles":        ("tackles",                      "Tackles (2025-26+)",          "Defense"),
        "Recoveries":     ("recoveries",                   "Ball recoveries (2025-26+)",                    "Ball recoveries (2025-26+)",  "Defense"),
        "CBI":            ("clearances_blocks_interceptions","Clearances+Blocks+Interceptions","Defense"),
        "Def Contribution":("defensive_contribution",      "Defensive contribution score","Defense"),
        "Bonus":          ("bonus",                        "FPL Bonus points",            "FPL"),
        "FPL Points":     ("total_points",                 "Total FPL points",            "FPL"),
        "Price (£M)":     ("price_m",                     "Current FPL price",           "FPL"),
    }
    all_player_labels = list(PLAYER_METRICS.keys())

    # サイドバーフィルター
    st.sidebar.markdown("**Player Filters**")
    pos_filter = st.sidebar.multiselect("Position", ["GK","DEF","MID","FWD"],
                                         default=["GK","DEF","MID","FWD"])
    team_filter = st.sidebar.multiselect("Team", sorted(df_players["team_name"].unique()),
                                          default=sorted(df_players["team_name"].unique()))
    min_min = st.sidebar.slider("Min minutes", 90, 3000, 450, 90)

    df_filt = df_players[
        df_players["position"].isin(pos_filter)
        & df_players["team_name"].isin(team_filter)
        & (df_players["minutes"] >= min_min)
    ].copy()
    st.sidebar.markdown(f"<div style='color:{C['muted']};font-size:.8rem'>"
                        f"Filtered players: <b style='color:{C['amber']}'>{len(df_filt)}</b></div>",
                        unsafe_allow_html=True)

    tab_avail, tab_top10, tab_prad, tab_scatter2, tab_pcap, tab_unit, tab_dream, tab_custp = st.tabs([
        "📋 Available Metrics",
        "🏆 Top 10 Rankings",
        "🕸️ Player Radar",
        "⊕ 2-Axis Plot",
        "📐 Play Style (PCA)",
        "👥 Unit Analysis",
        "⚽ Dream Team",
        "🔧 Custom Metric",
    ])

    with tab_avail:
        st.markdown("## Available Player Metrics")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        with st.expander("📖 指標の読み方・使い方ガイド", expanded=False):
            st.markdown("""
            #### 主要指標の説明
            | 指標 | 意味 | 参考 |
            |------|------|------|
            | **xG** | Expected Goals。シュートのゴール期待値の合計 | 得点数と比較するとLuck(運)がわかる |
            | **xA** | Expected Assists。アシストパスのxG合計 | — |
            | **xGI / xGI p90** | xG+xA。攻撃への総関与度 | p90は90分あたりの値（出場時間補正） |
            | **xGC** | Expected Goals Conceded。出場中の被期待得点（低いほど良い） | — |
            | **Goal Luck** | 実得点 − xG。プラスなら期待値より多く決めた | — |
            | **Def Luck** | xGC − 失点。プラスなら期待値より少ない失点 | — |
            | **CBI** | Clearances+Blocks+Interceptions の合計 | DF・守備的MFが高い |

            #### FPL独自指標（ICT）の詳細
            | 指標 | 詳細な意味 |
            |------|-----------|
            | **Creativity** | チャンスメイクの量と質を評価。キーパス・クロス・スルーパスが多いほど高い。トップ下やクリエイティブなMFが高い値を示す（例: デ・ブライネ） |
            | **Threat** | ゴールへの脅威を数値化。シュート数・シュート位置・PA内行動から算出。ストライカーや得点力の高い選手が高い（例: サラー） |
            | **Influence** | 試合全体への関与度の総合値。ボールタッチ・デュエル・守備行動を包括的に評価。試合を動かすMFやキャプテン的存在が高い |
            | **ICT Index** | Influence・Creativity・Threatの合成スコア。FPLの総合評価指標 |

            #### 2025-26から追加された守備指標
            | 指標 | 詳細な意味 |
            |------|-----------|
            | **Recoveries** | ルーズボール・こぼれ球の回収数。タックルやインターセプトとは別に、デュエル後のこぼれ球や相手クリアの拾い直しをカウント |
            | **Def Contribution** | FPL独自の守備貢献スコア。タックル・クリア・インターセプト・ブロック等を総合的に評価した複合スコア |

            #### 使い方のヒント
            - **Top 10 Rankings**: ポジションフィルターと組み合わせて使うと効果的です（例: MIDのみでxA p90ランキング）
            - **Radar**: 同ポジション・同チームの選手を比較すると分かりやすいです
            - **Play Style PCA**: MIDのみに絞り攻撃指標を入れると「ボックストゥボックス vs アンカー」の軸が出ます
            - **p90指標**: 出場時間が異なる選手を公平に比較できます（最低出場分数フィルターと併用推奨）
            - **カスタム指標例**: 「攻撃貢献 = xG p90 × 2 + xA p90 × 1.5 + Creativity × 0.01」
            """)

        cats = sorted(set(v[2] for v in PLAYER_METRICS.values()))
        for cat in cats:
            st.markdown(f"### {cat}")
            rows = [(lbl, PLAYER_METRICS[lbl][0], PLAYER_METRICS[lbl][1])
                    for lbl in PLAYER_METRICS if PLAYER_METRICS[lbl][2] == cat]
            df_cat = pd.DataFrame(rows, columns=["Metric","Column","Description"])
            for stat_col, stat_lbl in [("Min","min"),("Max","max"),("Mean","mean")]:
                df_cat[stat_col] = df_cat["Column"].apply(
                    lambda c: f"{getattr(df_filt[c], stat_lbl)():.2f}"
                    if c in df_filt.columns else "—"
                )
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

    with tab_top10:
        st.markdown("## Top 10 Rankings")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,3])
        with col_a:
            rank_metric = st.selectbox("Rank by", all_player_labels,
                                       index=all_player_labels.index("xGI") if "xGI" in all_player_labels else 0)
            use_p90_top = st.toggle("per 90分に変換", value=False, key="top10_p90",
                                     help="選択した指標を90分あたりの値に変換して比較します")
            show_n = st.radio("Show", [10, 20, 30], horizontal=True)
            # 低い値が良い指標は昇順にする
            _low_is_good_base = {
                "goals_conceded", "expected_goals_conceded",
                "yellow_cards", "red_cards", "price_m",
            }
            # p90変換後の列名も含めて判定
            _raw_for_check = PLAYER_METRICS.get(rank_metric, ("",))[0]
            # col_rはcol_bで定義されるためここではraw列名のみで判定
            _default_asc = (
                _raw_for_check in _low_is_good_base or
                any(_raw_for_check.startswith(b) for b in _low_is_good_base)
            )
            sort_asc = st.toggle(
                "低い順に表示",
                value=_default_asc,
                key="top10_asc",
                help="xGC・失点数・イエローカードなど、低い値が良い指標に使います"
            )
        with col_b:
            _col_r_raw = PLAYER_METRICS[rank_metric][0]
            _p90_skip_top = {"price_m","minutes","starts"}
            # p90変換
            if use_p90_top and _col_r_raw not in _p90_skip_top and not _col_r_raw.endswith("_p90"):
                col_r, _ = to_p90(_col_r_raw, df_filt)
                rank_metric_label = rank_metric + " p90"
            else:
                col_r = _col_r_raw
                rank_metric_label = rank_metric
            if col_r not in df_filt.columns:
                st.warning(f"Column '{col_r}' not available.")
            else:
                _base_cols = ["player_name","team_name","position","minutes"]
                _show_cols = _base_cols if col_r in _base_cols else _base_cols + [col_r]
                df_top = (df_filt[list(dict.fromkeys(_show_cols))]
                          .sort_values(col_r, ascending=sort_asc)
                          .head(int(show_n))
                          .reset_index(drop=True))
                df_top.index += 1

                # パーセンタイル追加
                pool_vals = df_filt[col_r].dropna()
                if sort_asc:
                    # 低い順: 最小値がTop 1%
                    df_top["Percentile"] = df_top[col_r].apply(
                        lambda v: f"Top {int((pool_vals>=v).mean()*100)+1}%"
                    )
                else:
                    df_top["Percentile"] = df_top[col_r].apply(
                        lambda v: f"Top {100-int((pool_vals<=v).mean()*100)}%"
                    )
                pos_color = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}
                def pos_style(v):
                    c = pos_color.get(v,"#6b7280")
                    return f"background:{c};color:white;font-weight:700;border-radius:4px;text-align:center"
                df_top_show = df_top.rename(columns={col_r: rank_metric_label})
                styled = (df_top_show.style
                          .background_gradient(subset=[rank_metric_label], cmap="RdYlGn")
                          .format({rank_metric_label: "{:.3f}"}))
                # pandas 3.x は .map()、旧版は .applymap()
                try:
                    styled = styled.map(pos_style, subset=["position"])
                except AttributeError:
                    try:
                        styled = styled.applymap(pos_style, subset=["position"])
                    except Exception:
                        pass
                st.dataframe(styled, use_container_width=True)

    with tab_prad:
        st.markdown("## Player Radar")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1,2])
        with col_a:
            all_p = sorted(df_filt["display_name"].tolist())
            sel_p = st.multiselect("Select Players (2-5)", all_p,
                                    default=all_p[:3] if len(all_p) >= 3 else all_p)
            sel_pm = st.multiselect("Metrics (3-7)", all_player_labels,
                                     default=["xG","xA","Creativity","Threat","Influence","Clean Sheets"])
            use_p90_radar = st.toggle("per 90分に変換", value=False, key="radar_p90",
                                       help="選んだ指標を90分あたりに変換します（既にp90の指標はそのまま）")
            compare_mode = st.radio(
                "比較モード",
                ["絶対評価", "相対評価"],
                help="絶対評価: フィルタ後の全選手を母集団として比較。EPL全体での位置がわかる。"
                     " / 相対評価: 選んだ選手だけを母集団として比較。"
                     "アタッカー同士など似たレベルの選手間の細かな差を見たいときに使う。"
            )
        with col_b:
            if len(sel_p) >= 2 and len(sel_pm) >= 3:
                # p90変換（トグルONかつ既にp90でない指標のみ）
                _p90_skip = {"price_m","minutes","starts",
                             "xG_p90","xA_p90","xGI_p90","goals_p90","assists_p90",
                             "saves_p90","tackles_p90","recoveries_p90","cbi_p90","def_contribution_p90"}
                pm_cols = []
                sel_pm_disp = []
                for m in sel_pm:
                    raw_col = PLAYER_METRICS[m][0]
                    if use_p90_radar and raw_col not in _p90_skip and not raw_col.endswith("_p90"):
                        p90c, _ = to_p90(raw_col, df_filt)
                        pm_cols.append(p90c)
                        sel_pm_disp.append(m + " p90" if not m.endswith("p90") else m)
                    else:
                        pm_cols.append(raw_col)
                        sel_pm_disp.append(m)
                df_pr = df_filt[df_filt["display_name"].isin(sel_p)].set_index("display_name")
                if compare_mode == "相対評価":
                    z_pool_pr  = df_pr
                    caption_pr = "相対評価モード: 選んだ選手同士の中での相対的な位置。外側 = このグループ内での上位。"
                else:
                    z_pool_pr  = df_filt.set_index("player_name")
                    caption_pr = "絶対評価モード: フィルタ後の全選手を母集団としたパーセンタイル。外側 = 全体での上位。"
                fig_pr = radar(df_pr, pm_cols, sel_pm_disp, "Player Radar", z_pool=z_pool_pr)
                st.pyplot(fig_pr, use_container_width=True)
                st.caption(caption_pr)
            else:
                st.info("Select at least 2 players and 3 metrics.")

    with tab_scatter2:
        st.markdown("## 2-Axis Player Plot")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.caption("選手数が多いため上位N名に絞って表示します。ポジション・チームフィルターと組み合わせてください。")
        col_a2, col_b2 = st.columns([1, 3])
        with col_a2:
            px_label = st.selectbox("X Axis", all_player_labels,
                                    index=all_player_labels.index("xG") if "xG" in all_player_labels else 0, key="px")
            py_label = st.selectbox("Y Axis", all_player_labels,
                                    index=all_player_labels.index("xA") if "xA" in all_player_labels else 1, key="py")
            use_p90_sc = st.toggle("per 90分に変換", value=False, key="sc2_p90",
                                    help="選んだ指標を90分あたりに変換します")
            show_n2  = st.slider("表示人数（上位N名）", 10, 100, 30, 5)
            color_by = st.selectbox("色分け", ["position", "team_name"], key="pcol")
        with col_b2:
            _p90_skip_sc = {"price_m","minutes","starts"}
            _px_raw = PLAYER_METRICS[px_label][0]
            _py_raw = PLAYER_METRICS[py_label][0]
            if use_p90_sc and _px_raw not in _p90_skip_sc:
                px_col, px_disp = to_p90(_px_raw, df_filt)
                px_label_disp = px_label + " p90"
            else:
                px_col, px_disp, px_label_disp = _px_raw, px_label, px_label
            if use_p90_sc and _py_raw not in _p90_skip_sc:
                py_col, py_disp = to_p90(_py_raw, df_filt)
                py_label_disp = py_label + " p90"
            else:
                py_col, py_disp, py_label_disp = _py_raw, py_label, py_label
            if px_col not in df_filt.columns or py_col not in df_filt.columns:
                st.warning("選択した指標がこのシーズンでは利用できません")
            else:
                # 両軸の合計スコアでTop N を選出
                _xs = (df_filt[px_col] - df_filt[px_col].min()) / (df_filt[px_col].max() - df_filt[px_col].min() + 1e-9)
                _ys = (df_filt[py_col] - df_filt[py_col].min()) / (df_filt[py_col].max() - df_filt[py_col].min() + 1e-9)
                df_s2 = df_filt.copy()
                df_s2["_rank_score"] = _xs + _ys
                df_s2 = df_s2.nlargest(int(show_n2), "_rank_score")

                pos_color2 = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}
                if color_by == "position":
                    colors2 = [pos_color2.get(p, "#64748b") for p in df_s2["position"]]
                else:
                    _teams  = df_s2["team_name"].tolist()
                    _cmap2  = team_color_map(_teams)
                    colors2 = [_cmap2.get(t, "#64748b") for t in _teams]

                fig_s2, ax_s2 = plt.subplots(figsize=(8, 6))
                apply_dark_style(fig_s2, ax_s2)
                ax_s2.scatter(df_s2[px_col], df_s2[py_col],
                              c=colors2, s=70, alpha=0.85, edgecolors="#374151", lw=0.4, zorder=3)
                ax_s2.axhline(df_filt[py_col].mean(), color="#555", ls="--", lw=0.8)
                ax_s2.axvline(df_filt[px_col].mean(), color="#555", ls="--", lw=0.8)
                for _, row in df_s2.iterrows():
                    ax_s2.annotate(row["player_name"][:12],
                                   (row[px_col], row[py_col]),
                                   xytext=(4, 4), textcoords="offset points",
                                   fontsize=6.5, color="#1a1a2e", alpha=0.9)
                ax_s2.set_xlabel(px_label_disp, color="#333333")
                ax_s2.set_ylabel(py_label_disp, color="#333333")
                ax_s2.set_title(f"{px_label_disp}  vs  {py_label_disp}  (Top {show_n2})",
                                color="#1a1a2e", fontweight="bold")
                # 凡例（ポジション別）
                if color_by == "position":
                    import matplotlib.patches as _mp
                    _handles = [_mp.Patch(color=c, label=p) for p, c in pos_color2.items()]
                    ax_s2.legend(handles=_handles, fontsize=8,
                                 facecolor="#f5f7fa", edgecolor="#cccccc", labelcolor="#1a1a2e")
                plt.tight_layout()
                st.pyplot(fig_s2, use_container_width=True)
                st.caption(f"破線 = フィルタ後の平均値。両軸スコア上位{show_n2}名を表示。")

    with tab_pcap:
        st.markdown("## Play Style Analysis (PCA)")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.info("**推奨:** 同ポジションで実施すると意味のある軸が出てきます。例: MFのみでPCA → ボックストゥボックス vs アンカー")

        pca_pos = st.multiselect("Position filter for PCA", ["GK","DEF","MID","FWD"],
                                  default=["MID"])
        df_pca_p = df_filt[df_filt["position"].isin(pca_pos)].copy()
        # PCAの選手名表示はチーム名なしで簡潔に
        df_pca_p["_plot_name"] = df_pca_p["player_name"]

        use_p90_pca = st.toggle("per 90分に変換", value=True, key="pca_p90",
                                     help="指標を90分あたりに変換して比較します")
        sel_pcap = st.multiselect("Metrics for PCA (4-8 recommended)", all_player_labels,
                                   default=["xG","xA","Creativity","Threat","Influence","Saves"])
        if len(df_pca_p) >= 5 and len(sel_pcap) >= 3:
            _p90_skip_pca = {"price_m","minutes","starts"}
            pcap_cols = []
            sel_pcap_disp = []
            for m in sel_pcap:
                raw_c = PLAYER_METRICS[m][0]
                if use_p90_pca and raw_c not in _p90_skip_pca and not raw_c.endswith("_p90"):
                    p90c, _ = to_p90(raw_c, df_pca_p)
                    pcap_cols.append(p90c)
                    sel_pcap_disp.append(m + " p90")
                else:
                    pcap_cols.append(raw_c)
                    sel_pcap_disp.append(m)
            # PCA実行
            X = df_pca_p[pcap_cols].fillna(0).values.astype(float)
            X_z = (X - X.mean(0)) / (X.std(0) + 1e-9)
            from numpy.linalg import svd as npsvd
            _, _, Vt_p = npsvd(X_z, full_matrices=False)
            scores_p = X_z @ Vt_p[:2].T
            loadings_p = Vt_p[:2].T

            df_pca_p = df_pca_p.reset_index(drop=True).copy()
            df_pca_p["PC1"] = scores_p[:,0]
            df_pca_p["PC2"] = scores_p[:,1]

            # 散布図（選手名オーバーレイ）
            fig_pp, ax_pp = plt.subplots(figsize=(10,7))
            apply_dark_style(fig_pp, ax_pp)
            pcmap = {p: PITCH_COLORS[i % len(PITCH_COLORS)]
                     for i, p in enumerate(df_pca_p["team_name"].unique())}
            clrs_p = [pcmap.get(t, C["sky"]) for t in df_pca_p["team_name"]]
            ax_pp.scatter(df_pca_p["PC1"], df_pca_p["PC2"],
                          c=clrs_p, s=60, alpha=.8, edgecolors="#374151", lw=.4, zorder=3)
            ax_pp.axhline(0, color="#374151", ls="--", lw=.7)
            ax_pp.axvline(0, color="#374151", ls="--", lw=.7)
            for _, row in df_pca_p.iterrows():
                ax_pp.annotate(row.get("_plot_name", row["player_name"])[:12],
                               (row["PC1"], row["PC2"]),
                               xytext=(3,3), textcoords="offset points",
                               fontsize=6.5, color=C["chalk"], alpha=.8)
            ax_pp.set_xlabel("PC 1", color=C["muted"])
            ax_pp.set_ylabel("PC 2", color=C["muted"])
            ax_pp.set_title(f"Play Style Map — {', '.join(pca_pos)}",
                            color=C["chalk"], fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pp, use_container_width=True)

            # 寄与度 → プレースタイルラベル提案
            col1, col2 = st.columns(2)
            load_df_p = pd.DataFrame({"Metric": sel_pcap_disp,
                                      "PC1 Loading": loadings_p[:,0].round(3),
                                      "PC2 Loading": loadings_p[:,1].round(3)})
            top_pc1 = load_df_p.reindex(load_df_p["PC1 Loading"].abs().sort_values(ascending=False).index).iloc[:3]
            top_pc2 = load_df_p.reindex(load_df_p["PC2 Loading"].abs().sort_values(ascending=False).index).iloc[:3]

            with col1:
                st.markdown("**PC1 — Style Axis Drivers**")
                st.dataframe(top_pc1[["Metric","PC1 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",
                                                        subset=["PC1 Loading"],vmin=-1,vmax=1)
                             .format({"PC1 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)
            with col2:
                st.markdown("**PC2 — Style Axis Drivers**")
                st.dataframe(top_pc2[["Metric","PC2 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",
                                                        subset=["PC2 Loading"],vmin=-1,vmax=1)
                             .format({"PC2 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)

            # TOP/BOT10
            st.markdown("**PC1 extremes (play style contrast)**")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"<span class='pill pill-amber'>High PC1 →</span>", unsafe_allow_html=True)
                st.dataframe(df_pca_p.nlargest(5,"PC1")[["player_name","team_name","PC1"]]
                             .style.format({"PC1":"{:+.2f}"}), hide_index=True)
            with c2:
                st.markdown(f"<span class='pill pill-sky'>Low PC1 →</span>", unsafe_allow_html=True)
                st.dataframe(df_pca_p.nsmallest(5,"PC1")[["player_name","team_name","PC1"]]
                             .style.format({"PC1":"{:+.2f}"}), hide_index=True)
        else:
            st.info("5人以上の選手と3指標以上を選択してください。")

    with tab_unit:
        st.markdown("## Unit Analysis")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.caption(
            "複数のユニット（選手の組み合わせ）を比較します。"
            "各ユニットの共出場試合に限定して指標を合算・平均して比較します。"
        )

        # ユニット定義
        n_units = st.number_input("比較するユニット数", 2, 4, 2, key="n_units")
        unit_defs = []
        for _u in range(int(n_units)):
            with st.expander(f"ユニット {_u+1}", expanded=True):
                _uname = st.text_input("ユニット名", value=f"Unit {_u+1}", key=f"uname_{_u}")
                _uplayers = st.multiselect(
                    "選手を選択（2〜5人）",
                    sorted(df_filt["display_name"].tolist()),
                    key=f"uplayers_{_u}"
                )
                unit_defs.append({"name": _uname, "players": _uplayers})

        sel_um = st.multiselect(
            "比較する指標",
            all_player_labels,
            default=["xGC","CBI","Recoveries","Tackles","Clean Sheets"],
            key="unit_metrics"
        )
        use_p90_unit = st.toggle("per 90分に変換", value=True, key="unit_p90",
                                  help="共出場時間で割った値で比較します")

        if st.button("▶ ユニット比較を実行", type="primary", key="run_unit"):
            _p90_skip_u = {"price_m","minutes","starts"}

            # GWデータを使って共出場試合を特定
            if df_g_raw is None or "element" not in df_g_raw.columns:
                st.warning("GWデータが読み込めていません")
            else:
                unit_results = []
                valid = True

                for _ud in unit_defs:
                    if len(_ud["players"]) < 2:
                        st.warning(f"{_ud['name']}: 選手を2人以上選んでください")
                        valid = False
                        break

                    # display_name → id マッピング（チーム名付きで一意に特定）
                    if "id" not in df_players.columns or "display_name" not in df_players.columns:
                        st.warning("選手IDが取得できません")
                        valid = False
                        break

                    _id_map = df_players[["display_name","id"]].drop_duplicates()
                    _ids = _id_map[_id_map["display_name"].isin(_ud["players"])]["id"].tolist()

                    # 全員が出場したGWを特定
                    _gw_col_u = "GW" if "GW" in df_g_raw.columns else "round"
                    # minutes > 0 の行のみ（登録されていても欠場の場合を除外）
                    _participated = df_g_raw[
                        df_g_raw["element"].isin(_ids) &
                        (pd.to_numeric(df_g_raw.get("minutes", 0), errors="coerce").fillna(0) > 0)
                    ].copy()
                    _gw_counts = _participated.groupby(_gw_col_u)["element"].nunique()
                    _shared_gws = _gw_counts[_gw_counts >= len(_ids)].index.tolist()

                    if not _shared_gws:
                        st.warning(f"{_ud['name']}: 全員が共出場した試合がありません")
                        valid = False
                        break

                    # 共出場試合のデータのみ抽出して合算
                    _shared_data = _participated[_participated[_gw_col_u].isin(_shared_gws)]
                    _total_min = _shared_data.groupby("element")["minutes"].sum().sum()

                    row = {"ユニット": _ud["name"],
                           "選手": ", ".join([p.split(" (")[0] for p in _ud["players"]]),  # チーム名を除いた表示
                           "共出場GW": len(_shared_gws),
                           "総出場分": int(_total_min)}

                    for _m in sel_um:
                        _raw_c = PLAYER_METRICS[_m][0]
                        if _raw_c in _shared_data.columns:
                            _val = pd.to_numeric(_shared_data[_raw_c], errors="coerce").sum()
                            if use_p90_unit and _raw_c not in _p90_skip_u:
                                _mins = max(_total_min / len(_ids), 1)
                                _val = _val / (_mins / 90)
                                _label = _m + " p90"
                            else:
                                _label = _m
                            row[_label] = round(float(_val), 3)

                    unit_results.append(row)

                if valid and unit_results:
                    df_unit = pd.DataFrame(unit_results)
                    metric_cols = [c for c in df_unit.columns
                                   if c not in ["ユニット","選手","共出場GW","総出場分"]]

                    st.markdown("**集計結果**")
                    st.dataframe(
                        df_unit[["ユニット","選手","共出場GW"] + metric_cols]
                        .style.background_gradient(subset=metric_cols, cmap="RdYlGn"),
                        use_container_width=True, hide_index=True
                    )

                    if len(metric_cols) >= 3:
                        # ① レーダーチャート（全体像）
                        st.markdown("**レーダーチャート（全体像）**")
                        df_u_radar = df_unit.set_index("ユニット")[metric_cols]
                        fig_ur = radar(df_u_radar, metric_cols, metric_cols,
                                       "Unit Comparison Radar",
                                       z_pool=df_u_radar)
                        st.pyplot(fig_ur, use_container_width=True)
                        st.caption("相対評価: 比較ユニット内でのパーセンタイル")

                    st.caption("共出場GW数が5節未満の場合は統計的信頼性が下がります。")

    with tab_dream:
        st.markdown("## ⚽ Dream Team Builder")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.caption("11人を選んでチームとしての総合スタッツを確認。フォーメーション配置図・レーダーチャート・数値テーブルで実在チームやリーグ平均と比較できます。")

        def make_positions(n_df, n_mf):
            """DF数・MF数からポジションリストを動的生成"""
            n_fw = 10 - n_df - n_mf
            positions = []
            row = 0

            # FW行
            fw_labels = {1:["ST"], 2:["LST","RST"], 3:["LW","CF","RW"],
                         4:["LW","LST","RST","RW"]}.get(n_fw, [f"FW{i+1}" for i in range(n_fw)])
            xs_fw = [0.15 + 0.7*(i/(max(n_fw-1,1))) for i in range(n_fw)]
            for i, lbl in enumerate(fw_labels):
                positions.append((lbl, row, xs_fw[i]))
            row += 1

            # MF行
            mf_labels = {3:["LCM","CM","RCM"], 4:["LM","LCM","RCM","RM"],
                         5:["LM","LCM","CM","RCM","RM"]}.get(n_mf, [f"MF{i+1}" for i in range(n_mf)])
            xs_mf = [0.1 + 0.8*(i/(max(n_mf-1,1))) for i in range(n_mf)]
            for i, lbl in enumerate(mf_labels):
                positions.append((lbl, row, xs_mf[i]))
            row += 1

            # DF行
            df_labels = {3:["LCB","CB","RCB"], 4:["LB","LCB","RCB","RB"],
                         5:["LB","LCB","CB","RCB","RB"]}.get(n_df, [f"DF{i+1}" for i in range(n_df)])
            xs_df = [0.08 + 0.84*(i/(max(n_df-1,1))) for i in range(n_df)]
            for i, lbl in enumerate(df_labels):
                positions.append((lbl, row, xs_df[i]))
            row += 1

            # GK
            positions.append(("GK", row, 0.5))
            return positions

        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            n_df = st.radio("DF人数", [3, 4, 5], index=1, horizontal=True, key="dt_df")
            n_mf = st.radio("MF人数", [3, 4, 5], index=0, horizontal=True, key="dt_mf")
            n_fw = 10 - n_df - n_mf
            if n_fw < 1:
                st.error("FWが0人以下になります。DFかMFを減らしてください。")
                n_mf = 3
                n_fw = 10 - n_df - n_mf
            formation = f"{n_df}-{n_mf}-{n_fw}"
            st.markdown(f"**フォーメーション: {formation}**")
            pos_list = make_positions(n_df, n_mf)

            # 各ポジションに選手を割り当て
            all_players_dt = sorted(df_filt["display_name"].tolist())
            assignments    = {}
            st.markdown("**選手を割り当て**")
            for pos_label, row, col_pct in pos_list:
                default_idx = 0
                assignments[pos_label] = st.selectbox(
                    pos_label, ["(未選択)"] + all_players_dt,
                    key=f"dt_{pos_label}", label_visibility="visible"
                )

            # 比較対象
            st.markdown("**比較対象**")
            compare_type = st.radio("比較対象", ["リーグ平均", "チームを選択"],
                                     key="dt_compare", horizontal=True)
            if compare_type == "チームを選択":
                compare_team = st.selectbox("チーム", sorted(df_teams["team_name"].tolist()),
                                             key="dt_team")
            else:
                compare_team = None

            # 比較指標
            dt_metrics_labels = ["xG","xA","xGI","Goals","Assists",
                                  "Clean Sheets","Saves","CBI","Tackles","Recoveries",
                                  "Creativity","Threat","Influence"]
            dt_metrics_map = {
                "xG": "expected_goals", "xA": "expected_assists",
                "xGI": "expected_goal_involvements",
                "Goals": "goals_scored", "Assists": "assists",
                "Clean Sheets": "clean_sheets", "Saves": "saves",
                "CBI": "clearances_blocks_interceptions",
                "Tackles": "tackles", "Recoveries": "recoveries",
                "Creativity": "creativity", "Threat": "threat", "Influence": "influence",
            }
            sel_dt_metrics = st.multiselect("比較指標", dt_metrics_labels,
                                             default=["xG","xA","Clean Sheets","CBI","Saves","Creativity"],
                                             key="dt_met")

        with col_d2:
            selected_players = [v for v in assignments.values() if v != "(未選択)"]

            # ── フィールド配置図 ──────────────────────────────────
            pos_colors = {"GK":"#f59e0b","LB":"#22c55e","LCB":"#22c55e",
                          "CB":"#22c55e","RCB":"#22c55e","RB":"#22c55e",
                          "LDM":"#3b82f6","RDM":"#3b82f6",
                          "LM":"#3b82f6","LCM":"#3b82f6","CM":"#3b82f6",
                          "CAM":"#8b5cf6","LAM":"#8b5cf6","RAM":"#8b5cf6",
                          "RCM":"#3b82f6","RM":"#3b82f6",
                          "LW":"#ef4444","CF":"#ef4444","RW":"#ef4444",
                          "LST":"#ef4444","RST":"#ef4444","ST":"#ef4444",}

            n_rows = max(r for _, r, _ in pos_list) + 1
            import matplotlib.patches as mpatches
            fig_field, ax_field = plt.subplots(figsize=(7, 4.8))
            fig_field.patch.set_facecolor("#1a6b2a")
            ax_field.set_facecolor("#1a6b2a")
            ax_field.set_xlim(0, 1); ax_field.set_ylim(0, 1)
            ax_field.axis("off")

            _lw, _a = 1.2, 0.55  # 線の太さ・透明度

            # ── ピッチ描画 ──
            # 外枠
            ax_field.add_patch(mpatches.FancyBboxPatch(
                (0.03,0.03),0.94,0.94, boxstyle="square,pad=0",
                ec="#ffffffbb", fc="none", lw=_lw))
            # センターライン
            ax_field.plot([0.03,0.97],[0.5,0.5], color="#ffffffbb", lw=_lw)
            # センターサークル＋スポット
            ax_field.add_patch(plt.Circle((0.5,0.5),0.10,fill=False,
                color="#ffffffbb",lw=_lw))
            ax_field.plot(0.5,0.5,"o",color="white",ms=2.5,alpha=0.7)
            # ペナルティエリア（上下）
            for ya, yb in [(0.74,0.97),(0.03,0.26)]:
                ax_field.add_patch(mpatches.FancyBboxPatch(
                    (0.2,ya),0.6,yb-ya, boxstyle="square,pad=0",
                    ec="#ffffffbb",fc="none",lw=_lw))
            # ゴールエリア（上下）
            for ya, yb in [(0.86,0.97),(0.03,0.14)]:
                ax_field.add_patch(mpatches.FancyBboxPatch(
                    (0.34,ya),0.32,yb-ya, boxstyle="square,pad=0",
                    ec="#ffffffbb",fc="none",lw=_lw*0.7))
            # ゴール枠（上下）
            for ya in [0.97, 0.0]:
                ax_field.add_patch(mpatches.FancyBboxPatch(
                    (0.42,ya),0.16,0.03, boxstyle="square,pad=0",
                    ec="#ffffffbb",fc="#ffffff18",lw=_lw*0.7))
            # ペナルティスポット（上下）
            ax_field.plot(0.5,0.89,"o",color="white",ms=2,alpha=0.5)
            ax_field.plot(0.5,0.11,"o",color="white",ms=2,alpha=0.5)

            # 選手プロット
            row_positions = {}
            for pos_label, row, col_pct in pos_list:
                y = 1 - (row + 0.5) / n_rows
                row_positions[pos_label] = (col_pct, y)
                player_name = assignments.get(pos_label, "(未選択)")
                short_name = player_name.split(" (")[0][:10] if player_name != "(未選択)" else "?"
                color = pos_colors.get(pos_label, "#64748b")
                alpha = 0.95 if player_name != "(未選択)" else 0.4

                # ポジションラベル（ごく小さく）
                ax_field.text(col_pct, y+0.055, pos_label,
                              ha="center", va="bottom",
                              fontsize=5.5, color="white", alpha=0.65, zorder=4)
                # 選手名（大きく・丸角背景）
                _fc = color if alpha > 0.5 else "#555555"
                ax_field.text(col_pct, y, short_name,
                              ha="center", va="center",
                              fontsize=9, color="white", fontweight="bold", zorder=5,
                              bbox=dict(boxstyle="round,pad=0.3",
                                        fc=_fc, ec="#ffffffaa",
                                        lw=1.0, alpha=min(alpha+0.1,1.0)))

            plt.tight_layout(pad=0.3)
            st.pyplot(fig_field, use_container_width=True)

            # ── スタッツカード ───────────────────────────────────
            if len(selected_players) >= 1:
                df_sp = df_filt[df_filt["display_name"].isin(selected_players)].copy()
                # per matchに換算: 出場試合数の近似としてstarts列を使用
                # minutesから試合換算（90分=1試合）
                _total_min = df_sp["minutes"].sum()
                _matches   = max(_total_min / 90, 1)  # 11人合計の90分換算試合数

                _xg  = df_sp["expected_goals"].sum() / _matches
                _xa  = df_sp["expected_assists"].sum() / _matches
                _cbi = df_sp["clearances_blocks_interceptions"].sum() / _matches
                _cre = df_sp["creativity"].sum() / _matches
                _total_price = df_sp["price_m"].sum()

                st.caption(f"📊 per 90分換算（合計出場{_total_min:.0f}分 / {_matches:.1f}試合分）　💰 Price = FPL移籍市場の現在価格合計")
                _kcols = st.columns(5)
                _kcols[0].metric("⚽ xG/90",     f"{_xg:.3f}")
                _kcols[1].metric("🎯 xA/90",     f"{_xa:.3f}")
                _kcols[2].metric("🛡️ CBI/90",    f"{_cbi:.2f}")
                _kcols[3].metric("💡 Cre/90",    f"{_cre:.1f}")
                _kcols[4].metric("💰 総額",       f"£{_total_price:.1f}M")

            # ── レーダーチャート ──────────────────────────────────
            if len(selected_players) >= 3 and sel_dt_metrics:
                # 選択した11人の合算値を計算
                df_selected = df_filt[df_filt["display_name"].isin(selected_players)]

                # Dream Team: per 90min換算
                _dt_min     = df_selected["minutes"].sum()
                _dt_matches = max(_dt_min / 90, 1)

                dt_vals = {}
                for m_label in sel_dt_metrics:
                    col = dt_metrics_map.get(m_label)
                    if col and col in df_selected.columns:
                        _raw = float(df_selected[col].sum())
                        dt_vals[m_label] = _raw / _dt_matches
                    else:
                        dt_vals[m_label] = 0.0

                # 比較対象もper 90min換算
                if compare_type == "チームを選択" and compare_team:
                    df_cmp    = df_filt[df_filt["team_name"] == compare_team]
                    cmp_label = compare_team
                else:
                    df_cmp    = df_filt
                    cmp_label = "League Avg"

                cmp_vals = {}
                for m_label in sel_dt_metrics:
                    col = dt_metrics_map.get(m_label)
                    if col and col in df_cmp.columns:
                        if compare_type == "リーグ平均":
                            # チームごとにper90換算してから平均
                            def _per90(grp):
                                m = max(grp["minutes"].sum() / 90, 1)
                                return grp[col].sum() / m
                            cmp_vals[m_label] = float(
                                df_cmp.groupby("team_name").apply(_per90).mean()
                            )
                        else:
                            _cmp_min = max(df_cmp["minutes"].sum() / 90, 1)
                            cmp_vals[m_label] = float(df_cmp[col].sum()) / _cmp_min
                    else:
                        cmp_vals[m_label] = 0.0

                # レーダー描画
                labels = sel_dt_metrics
                n = len(labels)
                angles = [i * 2 * np.pi / n for i in range(n)] + [0]

                def scale_vals(d_vals, c_vals):
                    """両者の最大値でスケーリング（0〜1）"""
                    scaled_d, scaled_c = [], []
                    for lbl in labels:
                        mx = max(d_vals.get(lbl,0), c_vals.get(lbl,0), 1e-9)
                        scaled_d.append(d_vals.get(lbl,0) / mx)
                        scaled_c.append(c_vals.get(lbl,0) / mx)
                    return scaled_d + [scaled_d[0]], scaled_c + [scaled_c[0]]

                s_dt, s_cmp = scale_vals(dt_vals, cmp_vals)

                fig_radar, ax_r = plt.subplots(figsize=(5, 5),
                                                subplot_kw=dict(polar=True))
                fig_radar.patch.set_facecolor("#ffffff")
                ax_r.set_facecolor("#f8f9fa")
                ax_r.set_theta_offset(np.pi / 2)
                ax_r.set_theta_direction(-1)
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(labels, size=8, color="#1a1a2e")
                ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax_r.set_yticklabels(["25%","50%","75%","100%"], size=6, color="#94a3b8")
                ax_r.set_ylim(0, 1)
                ax_r.grid(color="#e0e0e0", lw=0.5)

                ax_r.plot(angles, s_dt, color="#3b82f6", lw=2, zorder=3, label="Dream Team")
                ax_r.fill(angles, s_dt, color="#3b82f6", alpha=0.15)
                ax_r.plot(angles, s_cmp, color="#ef4444", lw=2, ls="--", zorder=3, label=cmp_label)
                ax_r.fill(angles, s_cmp, color="#ef4444", alpha=0.08)
                ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                             fontsize=8, facecolor="#ffffff", edgecolor="#cccccc",
                             labelcolor="#1a1a2e")
                ax_r.set_title("Dream Team vs " + cmp_label + "  (per 90 min)",
                                color="#1a1a2e", fontweight="bold", pad=15)
                plt.tight_layout()
                st.pyplot(fig_radar, use_container_width=True)

                # 数値テーブル
                df_compare_table = pd.DataFrame({
                    "指標": labels,
                    "Dream Team (p90)": [round(dt_vals.get(m, 0), 3) for m in labels],
                    cmp_label+" (p90)": [round(cmp_vals.get(m, 0), 3) for m in labels],
                })
                df_compare_table["差分"] = (
                    df_compare_table["Dream Team (p90)"] - df_compare_table[cmp_label+" (p90)"]
                ).round(3)
                st.dataframe(df_compare_table, use_container_width=True, hide_index=True)
                st.caption(f"選手数: {len(selected_players)}人  |  青=Dream Team  赤破線={cmp_label}")
            else:
                st.info("左のパネルで3人以上の選手を割り当て、指標を選択するとレーダーチャートが表示されます。")

    with tab_custp:
        st.markdown("## 🔧 Build Your Own Player Metric")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        pmetric_name = st.text_input("Metric Name", value="My Player Score")
        use_p90_cust = st.toggle("per 90分に変換", value=False, key="cust_p90",
                                  help="全指標を90分あたりに変換してから計算します")
        pn_rows = st.number_input("Number of metrics", 1, 8, 3, key="pn")
        pcomps = []
        st.columns([3,2,2,3])[0].markdown("**Metric**")

        for i in range(int(pn_rows)):
            c1,c2,c3,_ = st.columns([3,2,2,3])
            with c1:
                m = st.selectbox(f"Metric {i+1}", all_player_labels,
                                  key=f"pm_{i}",
                                  index=min(i, len(all_player_labels)-1))
            with c2:
                w = st.number_input("Weight", value=1.0, step=0.1, key=f"pw_{i}")
            with c3:
                d = st.selectbox("", ["+","−"], key=f"pd_{i}")
            pcomps.append((m, PLAYER_METRICS[m][0], w, 1 if d=="+" else -1))

        show_pn = st.radio("Show top N", [10,20,30], horizontal=True, key="pshow")

        if st.button("▶  Calculate & Rank", type="primary", key="pcalc"):
            _p90_skip_c = {"price_m","minutes","starts"}
            _pcomps_resolved = []
            for r in pcomps:
                raw_c = r[1]
                if use_p90_cust and raw_c not in _p90_skip_c and not raw_c.endswith("_p90"):
                    p90c, _ = to_p90(raw_c, df_filt)
                    _pcomps_resolved.append((r[0], p90c, r[2], r[3]))
                else:
                    _pcomps_resolved.append(r)
            df_pc = df_filt[["player_name","team_name","position","minutes"]].copy()
            for r in _pcomps_resolved:
                if r[1] not in df_pc.columns and r[1] in df_filt.columns:
                    df_pc[r[1]] = df_filt[r[1]].values
            df_pc[pmetric_name] = sum(
                r[2]*r[3]*df_pc[r[1]].fillna(0) for r in _pcomps_resolved
            )
            df_pc = df_pc.sort_values(pmetric_name, ascending=False).head(int(show_pn)).reset_index(drop=True)
            df_pc.index += 1
            # パーセンタイル
            full_scores = sum(r[2]*r[3]*df_filt[r[1]].fillna(0) for r in pcomps)
            df_pc["Percentile"] = df_pc[pmetric_name].apply(
                lambda v: f"Top {100-int((full_scores<=v).mean()*100)}%"
            )
            st.dataframe(df_pc.style
                         .background_gradient(subset=[pmetric_name], cmap="RdYlGn")
                         .format({pmetric_name:"{:.3f}"}),
                         use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#0f172a !important;border-radius:8px;"
    "padding:.8rem 1.2rem;margin-top:2rem;text-align:center;"
    "border-top:2px solid #1a5c36'>"
    "<span style='color:#e2e8f0 !important;font-size:.72rem'>"
    "<b>Data: vaastav/Fantasy-Premier-League</b>"
    " (github.com/vaastav/Fantasy-Premier-League)"
    " · FPL data © Premier League · Non-commercial personal use only"
    "</span><br>"
    "<span style='color:#94a3b8 !important;font-size:.70rem'>"
    "Match stats (shots, possession, etc.): "
    "<b style='color:#e2e8f0'>API-Football</b>"
    " (api-football.com)"
    " · Data rights belong to respective rights holders"
    "</span><br>"
    "<span style='color:#64748b !important;font-size:.66rem'>"
    "Built with Streamlit · Python · NumPy · Pandas"
    "</span></div>",
    unsafe_allow_html=True
)
