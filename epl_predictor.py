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
SEASONS  = {"2024-25": 2024, "2023-24": 2023, "2022-23": 2022}

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
                "team": tname, "GW": gw,
                "shots_on_tgt":    s.get("Shots on Goal"),
                "total_shots":     s.get("Total Shots"),
                "shots_inbox":     s.get("Shots insidebox"),
                "possession":      s.get("Ball Possession"),
                "shots_on_tgt_ag": o.get("Shots on Goal"),
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
tab_line, tab_scatter_tab, tab_table, tab_note = st.tabs(["📈 相関係数推移", "⊕ 2-Axis Plot", "📊 数値テーブル", "📖 読み方"])

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
