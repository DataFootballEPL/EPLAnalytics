"""
EPL Predictor
=============
「N節時点の各指標が最終勝ち点とどれだけ相関するか」を可視化する検証アプリ。
EPL Analytics の別アプリとして単独デプロイ。

データソース:
  - vaastav/Fantasy-Premier-League (xG, xGC, Goals, Points 等)
  - api_stats_XXXX-XX.json (枠内シュート数等 API-Football)
    → GitHub リポジトリの Secrets に GITHUB_USER / GITHUB_REPO を設定
"""

import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import pearsonr, spearmanr

# ── ページ設定 ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="EPL Predictor", layout="wide", page_icon="📈")
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#f5f7fa !important;}
[data-testid="stSidebar"]{background:#1e2d3d !important;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
h1,h2,h3{color:#1a1a2e !important;}
p,span,li{color:#1a1a2e !important;}
.stTabs [data-baseweb="tab"]{background:#e8f0eb;color:#1a1a2e !important;font-weight:600;}
.stTabs [aria-selected="true"]{background:#1e3a5f !important;}
.stTabs [aria-selected="true"] *{color:#fff !important;}
</style>
""", unsafe_allow_html=True)

# ── 定数 ───────────────────────────────────────────────────────────────────
VAASTAV  = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS  = {"2025-26": 2025, "2024-25": 2024, "2023-24": 2023, "2022-23": 2022}

APF_NAME_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Nottingham Forest": "Nott'm Forest", "Newcastle United": "Newcastle",
    "Brighton & Hove Albion": "Brighton", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Spurs",
    "Tottenham": "Spurs", "Leicester City": "Leicester", "Ipswich Town": "Ipswich",
    "Sheffield United": "Sheffield Utd",
}

# ── データ取得 ──────────────────────────────────────────────────────────────
def _get(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_vaastav(season_str: str) -> pd.DataFrame | None:
    """GW×チームの時系列データを構築"""
    r = _get(f"{VAASTAV}/{season_str}/gws/merged_gw.csv")
    if not r:
        return None
    dg = pd.read_csv(io.StringIO(r.text))
    for c in ["expected_goals", "expected_goals_conceded",
               "creativity", "threat", "influence"]:
        if c in dg.columns:
            dg[c] = pd.to_numeric(dg[c], errors="coerce").fillna(0)
    dg["was_home"] = dg["was_home"].fillna(False).astype(bool)
    dg["gf"] = np.where(dg["was_home"],
                         pd.to_numeric(dg["team_h_score"], errors="coerce"),
                         pd.to_numeric(dg["team_a_score"], errors="coerce"))
    dg["ga"] = np.where(dg["was_home"],
                         pd.to_numeric(dg["team_a_score"], errors="coerce"),
                         pd.to_numeric(dg["team_h_score"], errors="coerce"))
    # fixture単位で勝ち点を計算（選手行の重複を避ける）
    dg["match_pts"] = np.where(dg["gf"] > dg["ga"], 3,
                      np.where(dg["gf"] == dg["ga"], 1, 0))
    # pts列 = fixture単位で一意化した勝ち点（後でgroupby合計するため）
    # fixture×teamで先頭行のmatch_ptsをptsとして割り当て
    _fix_pts = dg.groupby(["team","GW","fixture"])["match_pts"].first().reset_index()
    _fix_pts = _fix_pts.rename(columns={"match_pts":"pts"})
    dg = dg.merge(_fix_pts[["team","GW","fixture","pts"]], on=["team","GW","fixture"], how="left")
    return dg


@st.cache_data(ttl=3600, show_spinner=False)
def load_apf(season_str: str, repo_user: str, repo_name: str) -> pd.DataFrame:
    """API-Football JSONを読み込み GW×チームに変換"""
    if not repo_user or not repo_name:
        return pd.DataFrame()
    url = (f"https://raw.githubusercontent.com/{repo_user}/{repo_name}"
           f"/main/api_stats_{season_str}.json")
    r = _get(url)
    if not r:
        return pd.DataFrame()
    try:
        apf = r.json()
    except Exception:
        return pd.DataFrame()
    rows = []
    for fid, sides in apf.items():
        meta = sides.get("_meta", {})
        gw   = meta.get("gw", 0)
        if not gw:
            continue
        for side, opp in [("home", "away"), ("away", "home")]:
            s = sides.get(side, {})
            o = sides.get(opp, {})
            if not s or not s.get("team_name"):
                continue
            tname = APF_NAME_MAP.get(s["team_name"], s["team_name"])
            rows.append({
                "team":            tname,
                "GW":              gw,
                "shots_on_tgt":    s.get("Shots on Goal"),
                "total_shots":     s.get("Total Shots"),
                "shots_inbox":     s.get("Shots insidebox"),
                "possession":      s.get("Ball Possession"),
                "shots_on_tgt_ag": o.get("Shots on Goal"),
                "fouls":           s.get("Fouls"),
                "corners":         s.get("Corner Kicks"),
                "pass_accuracy":   s.get("Passes %"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ["team", "GW"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_correlation_curve(dg: pd.DataFrame, df_apf: pd.DataFrame,
                              metrics: list[str],
                              rank_filter: tuple | None = None) -> pd.DataFrame:
    """
    各GW N において「N節までの累積指標 vs 最終勝ち点」のPearson rを計算。
    Returns: DataFrame(GW, metric1, metric2, ...)
    """
    # final_pts: fixture単位で勝ち点を集計（選手行の重複を除く）
    _fix_fp = dg.groupby(["team","GW","fixture"])["pts"].first().reset_index()
    final_pts = _fix_fp.groupby("team")["pts"].sum().reset_index()
    final_pts.columns = ["team", "final_pts"]

    # 順位フィルタ: 最終順位が指定範囲のチームのみ対象
    if rank_filter is not None:
        final_pts = final_pts.sort_values("final_pts", ascending=False).reset_index(drop=True)
        final_pts["rank"] = final_pts.index + 1
        lo, hi = rank_filter
        final_pts = final_pts[final_pts["rank"].between(lo, hi)]

    max_gw = int(dg["GW"].max())
    results = []
    for n in range(1, max_gw + 1):
        row = {"GW": n}
        _dg_n = dg[dg["GW"] <= n]
        sub_v = _dg_n.groupby("team").agg(
            xG_cum   = ("expected_goals",          "sum"),
            xGC_cum  = ("expected_goals_conceded",  "sum"),
            cre_cum  = ("creativity",               "sum"),
            thr_cum  = ("threat",                   "sum"),
        ).reset_index()
        # 勝ち点・得失点はfixture単位で集計（選手行の重複を除く）
        _fix_agg = _dg_n.groupby(["team","GW","fixture"]).agg(
            pts=("pts","first"), gf=("gf","first")
        ).reset_index()
        _pts_agg = _fix_agg.groupby("team").agg(
            pts_cum=("pts","sum"), gf_cum=("gf","sum")
        ).reset_index()
        sub_v = sub_v.merge(_pts_agg, on="team", how="left")

        sub_v["net_xG_cum"] = sub_v["xG_cum"] - sub_v["xGC_cum"]
        sub = sub_v.merge(final_pts, on="team")
        if not df_apf.empty:
            sub_a = df_apf[df_apf["GW"] <= n].groupby("team").agg(
                sot_cum  = ("shots_on_tgt",    "sum"),
                shot_cum = ("total_shots",      "sum"),
                sib_cum  = ("shots_inbox",      "sum"),
                sot_ag   = ("shots_on_tgt_ag",  "sum"),
                poss_avg = ("possession",       "mean"),
            ).reset_index()
            sub = sub.merge(sub_a, on="team", how="left")

        if len(sub) < 3:
            continue  # pearsonr には最低3サンプル必要

        col_map = {
            "xG":               "xG_cum",
            "xGC":              "xGC_cum",
            "Net xG (xG-xGC)":  "net_xG_cum",
            "Goals":            "gf_cum",
            "Points (running)": "pts_cum",
            "Creativity":       "cre_cum",
            "Threat":           "thr_cum",
            "Shots on Target ⚡": "sot_cum",
            "Total Shots ⚡":    "shot_cum",
            "Shots in Box ⚡":   "sib_cum",
            "Shots on Tgt Ag ⚡":"sot_ag",
            "Possession % ⚡":   "poss_avg",
        }

        # 低い方が良い指標は符号を反転して表示
        INVERT = {"xGC", "Shots on Tgt Ag ⚡"}

        for label in metrics:
            col = col_map.get(label)
            if not col or col not in sub.columns:
                row[label] = np.nan
                continue
            vals = sub[col].fillna(0)
            if vals.std() < 1e-9:
                row[label] = np.nan
                continue
            try:
                r_val, _ = pearsonr(vals, sub["final_pts"])
                if label in INVERT:
                    r_val = -r_val  # 低いほど良い指標は符号反転
                row[label] = round(r_val, 4)
            except Exception:
                row[label] = np.nan

        results.append(row)
    return pd.DataFrame(results)


# ── UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;border-radius:10px;padding:1rem 1.5rem;
            margin-bottom:.8rem;border-left:6px solid #1e3a5f">
  <h1 style="color:#fff;margin:0;font-size:2rem;font-weight:900">EPL Predictor</h1>
  <div style="color:#94a3b8;font-size:.82rem;margin-top:.3rem">
    N節時点の指標 vs 最終勝ち点の相関係数推移
  </div>
</div>
<div style="height:4px;background:linear-gradient(90deg,#1e3a5f,#c45c00,#0077aa,transparent);
     border-radius:2px;margin-bottom:1rem"></div>
""", unsafe_allow_html=True)

# サイドバー
st.sidebar.markdown("### 設定")
# デバッグ情報
_debug_user = ""
_debug_repo = ""
try:
    _debug_user = st.secrets.get("GITHUB_USER", "未設定")
    _debug_repo = st.secrets.get("GITHUB_REPO", "未設定")
except Exception:
    _debug_user = "Secrets読込エラー"
    _debug_repo = "Secrets読込エラー"
with st.sidebar.expander("🔧 接続情報", expanded=False):
    st.caption(f"GITHUB_USER: `{_debug_user}`")
    st.caption(f"GITHUB_REPO: `{_debug_repo}`")
selected_seasons = st.sidebar.multiselect(
    "シーズン（複数選択で合算）",
    list(SEASONS.keys()), default=["2024-25"]
)

ALL_METRICS = [
    "xG", "xGC", "Net xG (xG-xGC)", "Goals", "Points (running)", "Creativity", "Threat",
    "Shots on Target ⚡", "Total Shots ⚡", "Shots in Box ⚡",
    "Shots on Tgt Ag ⚡", "Possession % ⚡",
]
selected_metrics = st.sidebar.multiselect(
    "比較する指標",
    ALL_METRICS,
    default=["xG", "Points (running)", "Shots on Target ⚡"]
)

gw_range = st.sidebar.slider("表示するGW範囲", 1, 38, (1, 38))

st.sidebar.markdown("---")
st.sidebar.markdown("**チームグループ絞り込み**")
group_mode = st.sidebar.radio(
    "対象チーム",
    ["全20チーム", "最終順位で絞り込み"],
    key="group_mode"
)
_rank_filter = None
rank_range = (1, 20)  # デフォルト（全チーム）
if group_mode == "最終順位で絞り込み":
    rank_range = st.sidebar.slider("最終順位の範囲", 1, 20, (1, 5))
    _rank_filter = tuple(rank_range)
    st.sidebar.caption(
        f"※ 対象: 最終順位 {rank_range[0]}〜{rank_range[1]}位のチーム"
        f"（{len(selected_seasons)}シーズン × {rank_range[1]-rank_range[0]+1}チーム"
        f" = 最大{len(selected_seasons)*(rank_range[1]-rank_range[0]+1)}サンプル）"
    )

show_highlight = st.sidebar.toggle("GW5・GW10・GW19 に縦線を表示", value=True)
show_r2        = st.sidebar.toggle("R²（決定係数）も表示", value=False)

if not selected_seasons:
    st.info("シーズンを1つ以上選択してください")
    st.stop()
if not selected_metrics:
    st.info("指標を1つ以上選択してください")
    st.stop()

# Secrets読み込み（キャッシュ関数の外で実行）
_repo_user = ""
_repo_name = ""
try:
    _repo_user = st.secrets.get("GITHUB_USER", "")
    _repo_name = st.secrets.get("GITHUB_REPO", "")
except Exception:
    pass

# データロード
all_dg  = []
all_apf = []
with st.spinner("データを読み込み中..."):
    for s in selected_seasons:
        dg = load_vaastav(s)
        if dg is not None:
            dg["season"] = s
            all_dg.append(dg)
        apf = load_apf(s, _repo_user, _repo_name)
        if not apf.empty:
            apf["season"] = s
            all_apf.append(apf)

if not all_dg:
    st.error("データを読み込めませんでした")
    st.stop()

dg_all  = pd.concat(all_dg,  ignore_index=True)
apf_all = pd.concat(all_apf, ignore_index=True) if all_apf else pd.DataFrame()

has_apf = not apf_all.empty
if not has_apf:
    apf_metrics = [m for m in selected_metrics if "⚡" in m]
    if apf_metrics:
        st.warning(f"⚡指標（{', '.join(apf_metrics)}）はAPI-Football JSONが必要です。"
                    "Streamlit Secrets に GITHUB_USER / GITHUB_REPO を設定してください。")
        selected_metrics = [m for m in selected_metrics if "⚡" not in m]

if not selected_metrics:
    st.info("有効な指標がありません")
    st.stop()

# 相関係数計算
with st.spinner("相関係数を計算中..."):
    season_dfs = {}
    for s in selected_seasons:
        _dg  = dg_all[dg_all["season"] == s]
        _apf = apf_all[apf_all["season"] == s] if has_apf else pd.DataFrame()
        _df  = build_correlation_curve(_dg, _apf, selected_metrics, rank_filter=_rank_filter)
        season_dfs[s] = _df

    if len(selected_seasons) == 1:
        df_corr = list(season_dfs.values())[0]
        df_std  = None
    else:
        all_df = pd.concat(season_dfs.values())
        df_corr = all_df.groupby("GW")[selected_metrics].mean().reset_index()
        df_std  = all_df.groupby("GW")[selected_metrics].std().reset_index()

# GW範囲フィルタ
if df_corr.empty or "GW" not in df_corr.columns:
    st.warning(
        "相関係数を計算できませんでした。"
        "チームの絞り込みが厳しすぎる（サンプル数不足）か、"
        "選択したシーズンにデータがない可能性があります。"
        "シーズンを追加するか、順位範囲を広げてください。"
    )
    st.stop()
df_plot = df_corr[df_corr["GW"].between(gw_range[0], gw_range[1])]
if df_plot.empty:
    st.warning("選択したGW範囲にデータがありません。")
    st.stop()

# メインタブ
tab_line, tab_scatter_tab, tab_burnout, tab_rank, tab_table, tab_note = st.tabs(["📈 相関係数推移", "⊕ 2-Axis Plot", "📉 息切れ分析", "🏆 息切れランキング", "📊 数値テーブル", "📖 読み方"])

with tab_line:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="both", color="#e0e0e0", lw=0.5, zorder=0)

    COLORS = ["#ef4444","#3b82f6","#22c55e","#f59e0b",
               "#8b5cf6","#ec4899","#06b6d4","#84cc16","#f97316","#64748b","#a855f7"]
    STYLES = ["-","--","-.",":",(0,(3,1,1,1))]*3

    show_errbar = st.sidebar.toggle("シーズン間のばらつき（エラーバー）を表示",
                                       value=len(selected_seasons) > 1,
                                       key="show_err",
                                       help="複数シーズン選択時にシーズン間の標準偏差を帯で表示します")
    df_std_plot = df_std[df_std["GW"].between(gw_range[0], gw_range[1])] if df_std is not None else None

    for i, metric in enumerate(selected_metrics):
        if metric not in df_plot.columns:
            continue
        vals = df_plot[metric]
        if show_r2:
            vals = vals ** 2
        color = COLORS[i % len(COLORS)]
        ax.plot(df_plot["GW"], vals,
                color=color,
                ls=STYLES[i % len(STYLES)],
                lw=2.2, marker="o", markersize=3.5,
                label=metric, alpha=0.9, zorder=3)
        # エラーバー（シーズン間標準偏差）
        if show_errbar and df_std_plot is not None and metric in df_std_plot.columns:
            std_vals = df_std_plot[metric]
            if show_r2:
                std_vals = std_vals * 2 * vals.abs()  # 誤差伝播
            ax.fill_between(df_plot["GW"],
                            vals - std_vals, vals + std_vals,
                            color=color, alpha=0.12, zorder=2)

    # 縦線
    if show_highlight:
        for gw_mark, label in [(5,"GW5"),(10,"GW10"),(19,"Half Season")]:
            if gw_range[0] <= gw_mark <= gw_range[1]:
                ax.axvline(gw_mark, color="#64748b", lw=1, ls=":", alpha=0.7)
                ax.text(gw_mark+0.2, ax.get_ylim()[0]+0.01, label,
                        fontsize=7.5, color="#64748b")

    ax.axhline(0.7, color="#cccccc", lw=0.7, ls="--")
    ax.text(gw_range[0], 0.71, "r = 0.70", fontsize=7.5, color="#999999")

    y_label = "R2 (Coefficient of determination)" if show_r2 else "Pearson r (correlation with final points)"
    ax.set_xlabel("Gameweek (GW)", color="#333333", fontsize=11)
    ax.set_ylabel(y_label, color="#333333", fontsize=11)
    seasons_label = " + ".join(selected_seasons)
    ax.set_title(f"Cumulative metric vs Final Points  [{seasons_label}]",
                 color="#1a1a2e", fontweight="bold", fontsize=12)
    ax.tick_params(colors="#333333")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor="#cccccc",
              labelcolor="#1a1a2e", loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    note = "縦軸: N節時点の累積値と最終勝ち点のPearson r (1 = perfect positive correlation)."
    if df_std_plot is not None:
        note += " Shaded area = ±1 SD across seasons."
    note += " xGC & Shots on Tgt Ag are sign-inverted (lower = better → shown as positive correlation)."
    st.caption(note)

with tab_scatter_tab:
    st.markdown("#### 2-Axis Plot — N節時点の指標 vs 最終勝ち点")
    st.caption("横軸: 選んだ指標のN節時点累積値  縦軸: 最終勝ち点  回帰直線・相関係数を表示")

    col_sc1, col_sc2 = st.columns([1, 3])
    with col_sc1:
        sc_gw    = st.slider("N節時点", 1, 38, 10, key="sc_gw")
        sc_metric = st.selectbox("指標", ALL_METRICS, key="sc_metric",
                                  index=ALL_METRICS.index("Shots on Target ⚡") if "Shots on Target ⚡" in ALL_METRICS else 0)
        sc_season = st.selectbox("シーズン", selected_seasons, key="sc_season")
        sc_show_reg = st.toggle("回帰直線を表示", value=True, key="sc_reg")
        # 順位絞り込み（サイドバーの _rank_filter を流用）
        use_rank_sc = st.toggle("最終順位で絞り込む", value=(_rank_filter is not None), key="sc_rank")

    with col_sc2:
        from scipy.stats import linregress

        # 対象シーズンのデータ準備
        _dg_sc  = dg_all[dg_all["season"] == sc_season]
        _apf_sc = apf_all[apf_all["season"] == sc_season] if has_apf else pd.DataFrame()

        # N節時点の累積値を計算
        _dg_sc_n = _dg_sc[_dg_sc["GW"] <= sc_gw]
        _sub_v = _dg_sc_n.groupby("team").agg(
            xG_cum    = ("expected_goals",         "sum"),
            xGC_cum   = ("expected_goals_conceded", "sum"),
            cre_cum   = ("creativity",             "sum"),
            thr_cum   = ("threat",                 "sum"),
        ).reset_index()
        _fix_agg_sc = _dg_sc_n.groupby(["team","GW","fixture"]).agg(
            pts=("pts","first"), gf=("gf","first")
        ).reset_index()
        _pts_agg_sc = _fix_agg_sc.groupby("team").agg(
            pts_cum=("pts","sum"), gf_cum=("gf","sum")
        ).reset_index()
        _sub_v = _sub_v.merge(_pts_agg_sc, on="team", how="left")
        _sub_v["net_xG_cum"] = _sub_v["xG_cum"] - _sub_v["xGC_cum"]

        _fix_fp_sc = _dg_sc.groupby(["team","GW","fixture"])["pts"].first().reset_index()
        _fp_sc = _fix_fp_sc.groupby("team")["pts"].sum().reset_index()
        _fp_sc.columns = ["team", "final_pts"]
        _sub_v = _sub_v.merge(_fp_sc, on="team")

        if not _apf_sc.empty:
            _sub_a = _apf_sc[_apf_sc["GW"] <= sc_gw].groupby("team").agg(
                sot_cum   = ("shots_on_tgt",   "sum"),
                shot_cum  = ("total_shots",    "sum"),
                sib_cum   = ("shots_inbox",    "sum"),
                sot_ag    = ("shots_on_tgt_ag","sum"),
                poss_avg  = ("possession",     "mean"),
            ).reset_index()
            _sub_v = _sub_v.merge(_sub_a, on="team", how="left")

        # 順位フィルタ
        if use_rank_sc:
            _fp_ranked = _fp_sc.sort_values("final_pts", ascending=False).reset_index(drop=True)
            _fp_ranked["rank"] = _fp_ranked.index + 1
            _lo = rank_range[0] if _rank_filter else 1
            _hi = rank_range[1] if _rank_filter else 20
            _valid_teams = _fp_ranked[_fp_ranked["rank"].between(_lo, _hi)]["team"]
            _sub_v = _sub_v[_sub_v["team"].isin(_valid_teams)]

        col_map_sc = {
            "xG": "xG_cum", "xGC": "xGC_cum", "Net xG (xG-xGC)": "net_xG_cum",
            "Goals": "gf_cum", "Points (running)": "pts_cum",
            "Creativity": "cre_cum", "Threat": "thr_cum",
            "Shots on Target ⚡": "sot_cum", "Total Shots ⚡": "shot_cum",
            "Shots in Box ⚡": "sib_cum", "Shots on Tgt Ag ⚡": "sot_ag",
            "Possession % ⚡": "poss_avg",
        }
        sc_col = col_map_sc.get(sc_metric)

        if not sc_col or sc_col not in _sub_v.columns or len(_sub_v) < 3:
            st.warning("データが不足しています。シーズン・指標・絞り込み条件を確認してください。")
        else:
            _x = _sub_v[sc_col].fillna(0).values.astype(float)
            _y = _sub_v["final_pts"].values.astype(float)
            _teams = _sub_v["team"].values

            # xGC / Shots on Tgt Ag は符号反転
            INVERT_SC = {"xGC", "Shots on Tgt Ag ⚡"}
            _x_plot = -_x if sc_metric in INVERT_SC else _x

            # 図を描画
            fig_sc, ax_sc = plt.subplots(figsize=(8, 5.5))
            fig_sc.patch.set_facecolor("#ffffff")
            ax_sc.set_facecolor("#f8f9fa")
            ax_sc.grid(color="#e0e0e0", lw=0.5, zorder=0)

            ax_sc.scatter(_x_plot, _y, s=80, zorder=3,
                           c=[f"#{hash(t)%0xFFFFFF:06x}" for t in _teams],
                           edgecolors="#555555", lw=0.5, alpha=0.9)

            # チーム名アノテーション
            for xi, yi, tn in zip(_x_plot, _y, _teams):
                ax_sc.annotate(tn[:10], (xi, yi),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=7.5, color="#1a1a2e", alpha=0.85)

            # 回帰直線と統計
            if sc_show_reg and len(_x_plot) >= 3:
                from scipy.stats import pearsonr as _pr
                slope, intercept, r_val, p_val, _ = linregress(_x_plot, _y)
                r_sq = r_val ** 2
                _xl = np.linspace(_x_plot.min(), _x_plot.max(), 100)
                ax_sc.plot(_xl, slope * _xl + intercept,
                            color="#f4a261", lw=2, ls="-", zorder=4,
                            label=f"y={slope:.2f}x+{intercept:.1f}")
                ax_sc.text(0.03, 0.97,
                            f"r = {r_val:+.3f}   R² = {r_sq:.3f}   p = {p_val:.3f}",
                            transform=ax_sc.transAxes,
                            va="top", ha="left", fontsize=10, color="#1a1a2e",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="#ffffffcc", ec="#cccccc"))

                # 外れ値ハイライト（残差 > 1.5σ）
                _y_pred = slope * _x_plot + intercept
                _resid  = _y - _y_pred
                _thr    = 1.5 * np.std(_resid)
                for xi, yi, tn, ri in zip(_x_plot, _y, _teams, _resid):
                    if abs(ri) >= _thr:
                        ax_sc.annotate(f"← {tn[:10]}",
                                        (xi, yi), xytext=(6, 0),
                                        textcoords="offset points",
                                        fontsize=8, color="#ef4444",
                                        fontweight="bold")

            _x_label = f"{sc_metric} (inverted)" if sc_metric in INVERT_SC else sc_metric
            ax_sc.set_xlabel(f"GW1-{sc_gw} cumulative {_x_label}", color="#333333", fontsize=10)
            ax_sc.set_ylabel("Final Points", color="#333333", fontsize=10)
            _grp = f" (rank {_lo}-{_hi})" if use_rank_sc else ""
            ax_sc.set_title(f"{sc_metric} vs Final Points — {sc_season} GW{sc_gw}{_grp}",
                             color="#1a1a2e", fontweight="bold", fontsize=11)
            for spine in ax_sc.spines.values():
                spine.set_color("#cccccc")
            if sc_show_reg:
                ax_sc.legend(fontsize=9, facecolor="#ffffff",
                              edgecolor="#cccccc", labelcolor="#1a1a2e")
            plt.tight_layout()
            st.pyplot(fig_sc, use_container_width=True)

            if sc_show_reg and len(_x_plot) >= 3:
                c1, c2, c3 = st.columns(3)
                c1.metric("Pearson r", f"{r_val:+.3f}")
                c2.metric("R²", f"{r_sq:.3f}")
                c3.metric("p-value", f"{p_val:.3f}",
                           delta="significant" if p_val < 0.05 else "not significant",
                           delta_color="normal" if p_val < 0.05 else "off")

with tab_burnout:
    st.markdown("## 📉 息切れ分析（前半戦 vs 後半戦）")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        "Analyze the correlation between first-half metrics (proxy for pressing style) "
        "and burnout score (2nd half - 1st half points per match)."
    )

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        b_season = st.selectbox("シーズン", list(SEASONS.keys()), key="b_season")
        split_gw = st.slider("前後半の区切りGW", 10, 25, 19, key="b_split",
                              help="この節以前を前半戦、以降を後半戦として計算します")

        st.markdown("**前半戦の指標（横軸）**")
        BURNOUT_METRICS = {
            "xG/M":                  ("xG",          "vaastav"),
            "xGC/M":                 ("xGC",         "vaastav"),
            "Net xG/M":              ("net_xG",      "vaastav"),
            "Goals/M":               ("gf",          "vaastav"),
            "Creativity/M":          ("creativity",  "vaastav"),
            "Threat/M":              ("threat",      "vaastav"),
            "Yellow+Red Cards/M":    ("cards",       "vaastav"),
            "Goal Luck/M":           ("goal_luck",   "vaastav"),
            "Def Luck/M":            ("def_luck",    "vaastav"),
            "Total Luck/M":          ("total_luck",  "vaastav"),
            "Possession % ⚡":        ("possession",  "apf"),
            "Pass Accuracy % ⚡":     ("pass_acc",    "apf"),
            "Fouls/M ⚡":             ("fouls",       "apf"),
            "Shot Conversion % ⚡":   ("shot_conv",   "apf"),
            "Shots on Tgt Rate % ⚡": ("sot_rate",    "apf"),
            "Total Shots/M ⚡":       ("total_shots", "apf"),
            "Shots on Tgt/M ⚡":      ("shots_on_tgt","apf"),
            "Shots Against/M ⚡":     ("shots_ag",    "apf"),
            "Corners/M ⚡":           ("corners",     "apf"),
        }

        x_metric = st.selectbox("X軸指標（前半戦）", list(BURNOUT_METRICS.keys()), key="b_xmetric")

        use_turnover = st.toggle("主力選手数を指標に使用", value=False, key="b_turnover",
                                  help="前半戦に一定分数以上出場した選手数（ターンオーバーの少なさ）")
        if use_turnover:
            min_minutes = st.slider("最低出場分数", 300, 1500, 855, 90, key="b_minmin",
                                     help="855分 ≈ 前半戦19試合の半分以上出場")
            x_metric = f"主力選手数（{min_minutes}分以上）"
            BURNOUT_METRICS[x_metric] = ("turnover", "vaastav")

        st.markdown("**除外チーム（監督交代など）**")
        _dg_b = load_vaastav(b_season)
        _all_teams_b = sorted(_dg_b["team"].dropna().unique().tolist()) if _dg_b is not None else []
        exclude_teams = st.multiselect("除外するチーム", _all_teams_b, key="b_exclude")

    with col_b2:
        if _dg_b is None:
            st.warning("データを読み込めませんでした")
        else:
            _dg_b = _dg_b.copy()
            for c in ["expected_goals","expected_goals_conceded","creativity","threat","yellow_cards","red_cards"]:
                if c in _dg_b.columns:
                    _dg_b[c] = pd.to_numeric(_dg_b[c], errors="coerce").fillna(0)
            _dg_b["was_home"] = _dg_b["was_home"].fillna(False).astype(bool)
            _dg_b["gf"] = np.where(_dg_b["was_home"],
                                    pd.to_numeric(_dg_b["team_h_score"], errors="coerce"),
                                    pd.to_numeric(_dg_b["team_a_score"], errors="coerce"))
            _dg_b["ga"] = np.where(_dg_b["was_home"],
                                    pd.to_numeric(_dg_b["team_a_score"], errors="coerce"),
                                    pd.to_numeric(_dg_b["team_h_score"], errors="coerce"))

            # fixture単位で勝ち点
            _fix_b = _dg_b.groupby(["team","GW","fixture"]).agg(
                gf=("gf","first"), ga=("ga","first")
            ).reset_index()
            _fix_b["pts"] = np.where(_fix_b["gf"]>_fix_b["ga"],3,
                            np.where(_fix_b["gf"]==_fix_b["ga"],1,0))

            _gw_max   = int(_fix_b["GW"].max())
            _n_first  = split_gw
            _n_second = max(_gw_max - split_gw, 1)

            _df_pts = _fix_b.groupby("team").apply(
                lambda g: pd.Series({
                    "first_pts":  g[g["GW"]<=split_gw]["pts"].sum(),
                    "second_pts": g[g["GW"]> split_gw]["pts"].sum(),
                    "total_pts":  g["pts"].sum(),
                })
            ).reset_index()
            _df_pts["first_ppm"]  = _df_pts["first_pts"]  / _n_first
            _df_pts["second_ppm"] = _df_pts["second_pts"] / _n_second
            _df_pts["burnout"]    = _df_pts["second_ppm"] - _df_pts["first_ppm"]

            if exclude_teams:
                _df_pts = _df_pts[~_df_pts["team"].isin(exclude_teams)]

            # X軸値の計算
            _metric_col, _metric_src = BURNOUT_METRICS.get(x_metric, ("xG","vaastav"))

            if _metric_col == "turnover":
                _fw = _dg_b[_dg_b["GW"]<=split_gw].groupby(["team","element"])["minutes"].sum().reset_index()
                _x_vals = _fw[_fw["minutes"]>=min_minutes].groupby("team").size().reset_index(name="x_val")
            elif _metric_src == "vaastav":
                if _metric_col == "cards":
                    _dg_b["_cards"] = _dg_b["yellow_cards"] + _dg_b["red_cards"]*2
                    # fixture単位で選手ごとに合算（ファウルカードは選手ごとに記録）
                    _fw2 = _dg_b[_dg_b["GW"]<=split_gw].groupby(["team","GW"])["_cards"].sum().reset_index()
                    _x_vals = _fw2.groupby("team")["_cards"].sum().div(_n_first).reset_index(name="x_val")
                elif _metric_col == "net_xG":
                    _fw2 = _dg_b[_dg_b["GW"]<=split_gw].groupby("team").agg(
                        xg=("expected_goals","sum"),xgc=("expected_goals_conceded","sum")).reset_index()
                    _fw2["x_val"] = (_fw2["xg"]-_fw2["xgc"])/_n_first
                    _x_vals = _fw2[["team","x_val"]]
                elif _metric_col == "gf":
                    _fw2 = _fix_b[_fix_b["GW"]<=split_gw].groupby("team")["gf"].sum().div(_n_first).reset_index(name="x_val")
                    _x_vals = _fw2
                elif _metric_col == "xGC":
                    # xGC: 選手が出場した間のチーム被xG → fixture単位でfirstではなく選手ごとにsumしてから試合ごとに平均
                    _fw2 = _dg_b[_dg_b["GW"]<=split_gw].groupby(["team","GW","fixture"])["expected_goals_conceded"].mean().reset_index()
                    _x_vals = _fw2.groupby("team")["expected_goals_conceded"].mean().reset_index(name="x_val")
                else:
                    if _metric_col in ("goal_luck", "def_luck", "total_luck"):
                        # Luck系: (goals - xG) / (xGC - goals_conceded) をチーム×GWで計算
                        _gl = _dg_b[_dg_b["GW"]<=split_gw].groupby(["team","GW"]).agg(
                            goals=("goals_scored","sum"),
                            xg=("expected_goals","sum"),
                            ga_sum=("goals_conceded","sum"),
                            xgc=("expected_goals_conceded","mean"),
                        ).reset_index()
                        _gl["goal_luck"]  = _gl["goals"] - _gl["xg"]
                        _gl["def_luck"]   = _gl["xgc"]  - _gl["ga_sum"]
                        _gl["total_luck"] = _gl["goal_luck"] + _gl["def_luck"]
                        _x_vals = _gl.groupby("team")[_metric_col].mean().reset_index(name="x_val")
                    else:
                        _raw_map = {"xG":"expected_goals","creativity":"creativity","threat":"threat"}
                        _raw_c = _raw_map.get(_metric_col, _metric_col)
                        _fw2 = _dg_b[_dg_b["GW"]<=split_gw].groupby(["team","GW"])[_raw_c].sum().reset_index()
                        _x_vals = _fw2.groupby("team")[_raw_c].mean().reset_index(name="x_val")
            else:
                _apf_b = load_apf(b_season, _repo_user, _repo_name)
                if _apf_b.empty:
                    st.warning("⚡指標はAPI-Football JSONが必要です")
                    _x_vals = pd.DataFrame(columns=["team","x_val"])
                else:
                    _gw_col_apf = "GW" if "GW" in _apf_b.columns else "gw" if "gw" in _apf_b.columns else None
                    _apf_fw = _apf_b[_apf_b[_gw_col_apf]<=split_gw] if _gw_col_apf else _apf_b
                    _apf_col = {"possession":"possession","pass_acc":"pass_accuracy",
                                "fouls":"fouls","shots_on_tgt":"shots_on_tgt",
                                "total_shots":"total_shots","shots_ag":"shots_on_tgt_ag",
                                "corners":"corners"}
                    if _metric_col in ("shot_conv","sot_rate"):
                        _gv = _apf_fw.groupby("team").agg(
                            sot=("shots_on_tgt","sum"),tot=("total_shots","sum")).reset_index()
                        _gv["x_val"] = _gv["sot"]/(_gv["tot"].clip(lower=1))*100
                        _x_vals = _gv[["team","x_val"]]
                    else:
                        _rc = _apf_col.get(_metric_col,"")
                        if _rc and _rc in _apf_fw.columns:
                            _x_vals = _apf_fw.groupby("team")[_rc].mean().reset_index()
                            _x_vals.columns = ["team","x_val"]
                        else:
                            _x_vals = pd.DataFrame(columns=["team","x_val"])

            # デバッグ: チーム名の一致確認
            _pts_teams = set(_df_pts["team"].tolist())
            _x_teams   = set(_x_vals["team"].tolist()) if not _x_vals.empty else set()
            _unmatched  = _pts_teams - _x_teams
            if _unmatched and not _x_vals.empty:
                st.caption(f"DEBUG: unmatched teams = {sorted(_unmatched)[:5]}")
            _df_plot = _df_pts.merge(_x_vals, on="team", how="inner")
            if exclude_teams:
                _df_plot = _df_plot[~_df_plot["team"].isin(exclude_teams)]

            if len(_df_plot) < 4:
                st.warning("データが不足しています")
            else:
                from scipy.stats import linregress as _linreg
                _x = _df_plot["x_val"].fillna(0).values.astype(float)
                _y = _df_plot["burnout"].values.astype(float)

                fig_b, ax_b = plt.subplots(figsize=(8, 5.5))
                fig_b.patch.set_facecolor("#ffffff")
                ax_b.set_facecolor("#f8f9fa")
                ax_b.grid(color="#e0e0e0", lw=0.5, zorder=0)
                ax_b.axhline(0, color="#94a3b8", lw=1.2, ls="--", alpha=0.7)

                _cols_b = ["#ef4444" if v < 0 else "#3b82f6" for v in _y]
                ax_b.scatter(_x, _y, c=_cols_b, s=90, zorder=3,
                              edgecolors="#555555", lw=0.5, alpha=0.9)
                for xi, yi, tn in zip(_x, _y, _df_plot["team"]):
                    ax_b.annotate(tn[:12], (xi, yi), xytext=(4,4),
                                   textcoords="offset points",
                                   fontsize=7.5, color="#1a1a2e", alpha=0.9)

                slope, intercept, r_val, p_val, _ = _linreg(_x, _y)
                _xl = np.linspace(_x.min(), _x.max(), 100)
                ax_b.plot(_xl, slope*_xl+intercept, color="#f4a261", lw=2, zorder=4)
                ax_b.text(0.03, 0.97,
                           f"r = {r_val:+.3f}   R² = {r_val**2:.3f}   p = {p_val:.3f}",
                           transform=ax_b.transAxes, va="top", fontsize=10,
                           color="#1a1a2e",
                           bbox=dict(boxstyle="round,pad=0.3",fc="#ffffffcc",ec="#cccccc"))

                ax_b.set_xlabel(f"First half (GW1-{split_gw}): {x_metric}", color="#333333", fontsize=10)
                ax_b.set_ylabel("Burnout score (2nd half pts/match - 1st half pts/match)", color="#333333", fontsize=10)
                ax_b.set_title(
                    f"{b_season}  {x_metric} vs Burnout score"
                    + (f"  ({len(exclude_teams)} teams excluded)" if exclude_teams else ""),
                    color="#1a1a2e", fontweight="bold", fontsize=11)
                for spine in ax_b.spines.values():
                    spine.set_color("#cccccc")
                plt.tight_layout()
                st.pyplot(fig_b, use_container_width=True)

                _sign = "negative correlation (strong 1st half -> burnout)" if r_val < 0 else "positive correlation (strong 1st half -> sustained)"
                st.caption(
                    f"Blue = 2nd half improvement, Red = burnout. "
                    f"r={r_val:+.3f}: {_sign}. "
                    + ("p<0.05 significant." if p_val<0.05 else "p>=0.05 not significant.")
                )

                with st.expander("📊 数値テーブル"):
                    _df_show = _df_plot[["team","x_val","first_ppm","second_ppm","burnout","total_pts"]].copy()
                    _df_show.columns = ["Team", x_metric,
                                         f"1st half pts/M (GW1-{split_gw})",
                                         f"2nd half pts/M (GW{split_gw+1}-)",
                                         "Burnout score", "Final pts"]
                    _df_show = _df_show.sort_values("Burnout score")
                    st.dataframe(
                        _df_show.round(3).style.background_gradient(subset=["Burnout score"], cmap="RdYlGn"),
                        use_container_width=True, hide_index=True
                    )

with tab_rank:
    st.markdown("## 🏆 息切れランキング")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("複数シーズン・複数指標の息切れスコアをまとめて表示。監督交代チームを除外して比較できます。")

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        rank_seasons = st.multiselect("シーズン（複数選択可）", list(SEASONS.keys()),
                                       default=list(SEASONS.keys())[:2], key="rank_seasons")
        rank_split   = st.slider("前後半の区切りGW", 10, 25, 19, key="rank_split")
        st.markdown("**ターンオーバー指標（オプション）**")
        show_turnover_rank = st.toggle("主力選手数もランキング表示", value=False, key="rank_turnover")
        if show_turnover_rank:
            rank_tv_gw  = st.slider("集計対象GW（最大）", 1, 38, 19, key="rank_tv_gw",
                                     help="この節までの出場分数で集計します")
            rank_tv_min = st.slider("最低出場分数", 90, 2000, 855, 90, key="rank_tv_min",
                                     help="855分 ≈ 19試合の半分以上出場")

        rank_exclude = {}
        with st.expander("除外チーム設定（監督交代等）", expanded=False):
            for _rs in rank_seasons:
                _dg_rs = load_vaastav(_rs)
                if _dg_rs is not None:
                    _teams_rs = sorted(_dg_rs["team"].dropna().unique().tolist())
                    _ex = st.multiselect(f"{_rs}", _teams_rs, key=f"rank_ex_{_rs}")
                    rank_exclude[_rs] = _ex

    with col_r2:
        if not rank_seasons:
            st.info("シーズンを選択してください")
        else:
            all_burnout_rows = []

            for _rs in rank_seasons:
                _dg_rs = load_vaastav(_rs)
                if _dg_rs is None:
                    continue

                _dg_rs = _dg_rs.copy()
                for c in ["expected_goals","expected_goals_conceded"]:
                    if c in _dg_rs.columns:
                        _dg_rs[c] = pd.to_numeric(_dg_rs[c], errors="coerce").fillna(0)
                _dg_rs["was_home"] = _dg_rs["was_home"].fillna(False).astype(bool)
                _dg_rs["gf"] = np.where(_dg_rs["was_home"],
                                         pd.to_numeric(_dg_rs["team_h_score"], errors="coerce"),
                                         pd.to_numeric(_dg_rs["team_a_score"], errors="coerce"))
                _dg_rs["ga"] = np.where(_dg_rs["was_home"],
                                         pd.to_numeric(_dg_rs["team_a_score"], errors="coerce"),
                                         pd.to_numeric(_dg_rs["team_h_score"], errors="coerce"))

                _fix_rs = _dg_rs.groupby(["team","GW","fixture"]).agg(
                    gf=("gf","first"), ga=("ga","first")
                ).reset_index()
                _fix_rs["pts"] = np.where(_fix_rs["gf"]>_fix_rs["ga"],3,
                                  np.where(_fix_rs["gf"]==_fix_rs["ga"],1,0))

                _gw_max_rs = int(_fix_rs["GW"].max())
                _n1 = rank_split
                _n2 = max(_gw_max_rs - rank_split, 1)

                _df_rs = _fix_rs.groupby("team").apply(
                    lambda g: pd.Series({
                        "first_pts":  g[g["GW"]<=rank_split]["pts"].sum(),
                        "second_pts": g[g["GW"]> rank_split]["pts"].sum(),
                        "total_pts":  g["pts"].sum(),
                    })
                ).reset_index()
                _df_rs["first_ppm"]  = _df_rs["first_pts"]  / _n1
                _df_rs["second_ppm"] = _df_rs["second_pts"] / _n2
                _df_rs["burnout"]    = _df_rs["second_ppm"] - _df_rs["first_ppm"]
                _df_rs["season"]     = _rs

                # 除外チームを除く
                _ex_rs = rank_exclude.get(_rs, [])
                if _ex_rs:
                    _df_rs = _df_rs[~_df_rs["team"].isin(_ex_rs)]

                all_burnout_rows.append(_df_rs)

            if not all_burnout_rows:
                st.warning("データがありません")
            else:
                df_all = pd.concat(all_burnout_rows, ignore_index=True)

                # ── 表示オプション ──
                view_mode = st.radio("表示形式", ["シーズン別平均", "全エントリ（チーム×シーズン）"],
                                      horizontal=True, key="rank_view")

                if view_mode == "シーズン別平均":
                    df_disp = (df_all.groupby("team")
                               .agg(burnout_avg=("burnout","mean"),
                                    total_pts_avg=("total_pts","mean"),
                                    n_seasons=("season","count"))
                               .reset_index()
                               .sort_values("burnout_avg"))
                    df_disp.columns = ["チーム","息切れ度(平均)","最終勝ち点(平均)","対象シーズン数"]
                else:
                    df_disp = df_all[["team","season","burnout","first_ppm","second_ppm","total_pts"]].copy()
                    df_disp.columns = ["チーム","シーズン","息切れ度","前半pts/M","後半pts/M","最終勝ち点"]
                    df_disp = df_disp.sort_values(["チーム","シーズン"])

                # ── 棒グラフ ──
                fig_r, ax_r = plt.subplots(figsize=(8, max(4, len(df_disp)*0.42)))
                fig_r.patch.set_facecolor("#ffffff")
                ax_r.set_facecolor("#f8f9fa")
                ax_r.grid(axis="x", color="#e0e0e0", lw=0.5, zorder=0)
                ax_r.axvline(0, color="#94a3b8", lw=1.2, ls="--", alpha=0.7)

                _burnout_col = "息切れ度(平均)" if view_mode == "シーズン別平均" else "息切れ度"
                _label_col   = "チーム" if view_mode == "シーズン別平均" else df_disp.apply(
                    lambda row: f"{row['チーム']} ({row['シーズン']})", axis=1)

                _vals  = df_disp[_burnout_col].values
                if view_mode == "全エントリ（チーム×シーズン）":
                    _labels = df_disp.apply(lambda r: f"{r['チーム']} ({r['シーズン']})", axis=1).values
                else:
                    _labels = df_disp["チーム"].values

                _colors = ["#ef4444" if v < 0 else "#3b82f6" for v in _vals]
                ax_r.barh(range(len(_vals)), _vals, color=_colors, alpha=0.85,
                           edgecolor="#555555", lw=0.3)
                ax_r.set_yticks(range(len(_labels)))
                ax_r.set_yticklabels(_labels, fontsize=8.5, color="#1a1a2e")
                ax_r.set_xlabel("Burnout score (2nd half - 1st half pts/match)",
                                 color="#333333", fontsize=9)
                _title = f"Burnout Ranking  GW split={rank_split}"
                if any(rank_exclude.values()):
                    _n_ex = sum(len(v) for v in rank_exclude.values())
                    _title += f"  ({_n_ex} team-seasons excluded)"
                ax_r.set_title(_title, color="#1a1a2e", fontweight="bold", fontsize=10)
                for spine in ax_r.spines.values():
                    spine.set_color("#cccccc")
                plt.tight_layout()
                st.pyplot(fig_r, use_container_width=True)

                st.caption("Blue = 2nd half improvement, Red = burnout. "
                           "Excludes seasons with mid-season manager changes if specified above.")

                st.dataframe(
                    df_disp.round(3).style.background_gradient(
                        subset=[_burnout_col], cmap="RdYlGn"),
                    use_container_width=True, hide_index=True
                )

                # ── ターンオーバーランキング ──
                if show_turnover_rank:
                    st.markdown(f"---")
                    st.markdown(f"#### 主力選手数ランキング（GW1-{rank_tv_gw}で{rank_tv_min}分以上出場）")
                    tv_rows = []
                    for _rs in rank_seasons:
                        _dg_tv = load_vaastav(_rs)
                        if _dg_tv is None:
                            continue
                        _dg_tv["minutes"] = pd.to_numeric(_dg_tv.get("minutes", 0), errors="coerce").fillna(0)
                        _fw_tv = _dg_tv[_dg_tv["GW"] <= rank_tv_gw].groupby(
                            ["team","element"])["minutes"].sum().reset_index()
                        _cnt = (_fw_tv[_fw_tv["minutes"] >= rank_tv_min]
                                .groupby("team").size().reset_index(name="n_players"))
                        _cnt["season"] = _rs
                        _ex_tv = rank_exclude.get(_rs, [])
                        if _ex_tv:
                            _cnt = _cnt[~_cnt["team"].isin(_ex_tv)]
                        tv_rows.append(_cnt)

                    if tv_rows:
                        df_tv = pd.concat(tv_rows, ignore_index=True)
                        if view_mode == "シーズン別平均":
                            df_tv_disp = (df_tv.groupby("team")["n_players"]
                                          .mean().reset_index()
                                          .sort_values("n_players", ascending=False))
                            df_tv_disp.columns = ["チーム", f"主力選手数(avg, {rank_tv_min}分以上)"]
                            _tv_col = df_tv_disp.columns[1]
                            _tv_labels = df_tv_disp["チーム"].values
                        else:
                            df_tv_disp = df_tv.sort_values(["n_players"], ascending=False)
                            df_tv_disp.columns = ["チーム","主力選手数","シーズン"]
                            _tv_col = "主力選手数"
                            _tv_labels = df_tv_disp.apply(
                                lambda r: f"{r['チーム']} ({r['シーズン']})", axis=1).values

                        fig_tv, ax_tv = plt.subplots(figsize=(7, max(3, len(df_tv_disp)*0.4)))
                        fig_tv.patch.set_facecolor("#ffffff")
                        ax_tv.set_facecolor("#f8f9fa")
                        ax_tv.grid(axis="x", color="#e0e0e0", lw=0.5, zorder=0)
                        _tv_vals = df_tv_disp[_tv_col].values
                        ax_tv.barh(range(len(_tv_vals)), _tv_vals,
                                    color="#6366f1", alpha=0.85, edgecolor="#555555", lw=0.3)
                        ax_tv.set_yticks(range(len(_tv_labels)))
                        ax_tv.set_yticklabels(_tv_labels, fontsize=8.5, color="#1a1a2e")
                        ax_tv.set_xlabel(f"Players with >= {rank_tv_min} min in GW1-{rank_tv_gw}",
                                          color="#333333", fontsize=9)
                        ax_tv.set_title("Squad Depth Ranking", color="#1a1a2e",
                                         fontweight="bold", fontsize=10)
                        ax_tv.invert_yaxis()
                        for spine in ax_tv.spines.values():
                            spine.set_color("#cccccc")
                        plt.tight_layout()
                        st.pyplot(fig_tv, use_container_width=True)
                        st.caption(f"選手数が少ない = 固定メンバー中心（ターンオーバー少）、多い = ローテーション多用")
                        st.dataframe(df_tv_disp.round(1), use_container_width=True, hide_index=True)

with tab_table:
    # Spearman rも追加
    st.markdown("#### GWごとの相関係数（Pearson r）")
    # 特定GWを選んで詳細
    gw_sel = st.select_slider("GWを選択", options=df_plot["GW"].tolist(),
                               value=df_plot["GW"].iloc[len(df_plot)//2])
    row = df_plot[df_plot["GW"] == gw_sel].iloc[0]
    cols = st.columns(len(selected_metrics))
    for i, m in enumerate(selected_metrics):
        val = row.get(m, np.nan)
        r2  = val**2 if not np.isnan(val) else np.nan
        cols[i].metric(m[:20], f"r = {val:.3f}" if not np.isnan(val) else "N/A",
                        delta=f"R² = {r2:.3f}" if not np.isnan(r2) else None)

    st.markdown("---")
    st.markdown("#### 全GWのテーブル")
    df_disp = df_plot.set_index("GW")[selected_metrics].round(3)
    st.dataframe(
        df_disp.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True
    )

with tab_note:
    st.markdown("""
    #### 読み方

    **縦軸（Pearson r）**
    - `1.0` に近い → その指標が最終勝ち点と完全に連動
    - `0.7` 以上 → 強い相関（実用的な予測力あり）
    - `0.5` 未満 → 弱い相関（単独では予測力が低い）

    **横軸（GW）**
    - GW1時点: 1試合だけのデータで相関を計算（ノイズが大きい）
    - GW38時点: `Points (running)` の r = 1.0 はトートロジー（自分自身との相関）

    **Points (running) について**
    - 「その時点の勝ち点累積」と「最終勝ち点」の相関
    - 序盤は低く徐々に上がる → 序盤の勝ち点は「本来の実力」を反映しにくい

    **仮説の検証方法**
    - 「序盤の枠内シュートが xG より早く高い r を示すか」を GW5〜10 あたりで確認
    - 複数シーズンを選択して平均値で見ると安定した傾向がわかる

    #### データソース
    - FPL指標（xG, xGC 等）: vaastav/Fantasy-Premier-League © Premier League
    - ⚡指標（枠内シュート等）: API-Football (api-football.com)
    """)

# フッター
st.markdown(
    "<div style='background:#0f172a;padding:.7rem 1rem;border-radius:8px;"
    "margin-top:2rem;text-align:center;font-size:.68rem'>"
    "<span style='color:#94a3b8'>Non-commercial personal use only · "
    "Data © Premier League / API-Football</span></div>",
    unsafe_allow_html=True
)
