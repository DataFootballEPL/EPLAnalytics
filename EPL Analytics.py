"""
fetch_fpl_data.py
=================
FPL公式APIから直接データを取得してvaastav形式のCSVを生成するスクリプト。
vaastavの更新を待たずに最新GWデータをアプリで使えるようにします。

生成ファイル:
  players_raw.csv  → 全選手の累積スタッツ
  merged_gw.csv    → GW別スタッツ（vaastavと同形式）
  teams.csv        → チーム情報

使い方（vaastavが更新されるまでの代替として）:
  python fetch_fpl_data.py

  生成されたCSVをそのままGitHubのリポジトリの
  data/2026-27/ フォルダにアップロードすれば
  アプリが自動的に読み込みます。

注意:
  ・FPL APIはPLの個人利用規約に基づきます（非商業利用）
  ・レート制限に配慮するため選手間に0.2秒のスリープを入れています
  ・全選手取得には約10〜15分かかります（EPL登録選手約700人）
  ・--gw オプションで特定のGWのみ取得可能（高速）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# ── 設定 ─────────────────────────────────────────────────────────────────────
FPL_BASE = "https://fantasy.premierleague.com/api"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer":         "https://fantasy.premierleague.com/",
}

# vaastavのmerged_gw.csv に対応する element-summary > history の列マッピング
GW_COL_MAP = {
    "element":                         "element",
    "fixture":                         "fixture",
    "opponent_team":                   "opponent_team",
    "total_points":                    "total_points",
    "was_home":                        "was_home",
    "kickoff_time":                    "kickoff_time",
    "team_h_score":                    "team_h_score",
    "team_a_score":                    "team_a_score",
    "round":                           "round",
    "minutes":                         "minutes",
    "goals_scored":                    "goals_scored",
    "assists":                         "assists",
    "clean_sheets":                    "clean_sheets",
    "goals_conceded":                  "goals_conceded",
    "own_goals":                       "own_goals",
    "penalties_saved":                 "penalties_saved",
    "penalties_missed":                "penalties_missed",
    "yellow_cards":                    "yellow_cards",
    "red_cards":                       "red_cards",
    "saves":                           "saves",
    "bonus":                           "bonus",
    "bps":                             "bps",
    "influence":                       "influence",
    "creativity":                      "creativity",
    "threat":                          "threat",
    "ict_index":                       "ict_index",
    "starts":                          "starts",
    "expected_goals":                  "expected_goals",
    "expected_assists":                "expected_assists",
    "expected_goal_involvements":      "expected_goal_involvements",
    "expected_goals_conceded":         "expected_goals_conceded",
    "value":                           "value",
    "transfers_balance":               "transfers_balance",
    "selected":                        "selected",
    "transfers_in":                    "transfers_in",
    "transfers_out":                   "transfers_out",
    # 2025-26以降の新列（存在しない場合は0）
    "tackles":                         "tackles",
    "recoveries":                      "recoveries",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "defensive_contribution":          "defensive_contribution",
}


def fpl_get(endpoint: str, retries: int = 3) -> dict | None:
    url = f"{FPL_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limit → {wait}秒待機...")
                time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code}: {url}")
                return None
        except Exception as e:
            print(f"  Error ({attempt+1}/{retries}): {e}")
            time.sleep(5)
    return None


def fetch_bootstrap() -> dict | None:
    print("📋 bootstrap-static を取得中...")
    return fpl_get("bootstrap-static/")


def build_teams_csv(teams: list) -> pd.DataFrame:
    """チーム情報をteams.csv形式に変換"""
    rows = []
    for t in teams:
        rows.append({
            "code":       t.get("code"),
            "id":         t.get("id"),
            "name":       t.get("name"),
            "short_name": t.get("short_name"),
            "strength":   t.get("strength"),
            "played":     t.get("played"),
            "win":        t.get("win"),
            "draw":       t.get("draw"),
            "loss":       t.get("loss"),
            "points":     t.get("points"),
            "position":   t.get("position"),
        })
    return pd.DataFrame(rows)


def build_players_raw(elements: list, teams: list) -> pd.DataFrame:
    """選手の累積スタッツをplayers_raw.csv形式に変換"""
    team_map = {t["id"]: t["name"] for t in teams}
    pos_map  = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    rows = []
    for e in elements:
        rows.append({
            "id":                e.get("id"),
            "web_name":          e.get("web_name"),
            "first_name":        e.get("first_name"),
            "second_name":       e.get("second_name"),
            "team":              team_map.get(e.get("team"), e.get("team")),
            "team_code":         e.get("team_code"),
            "element_type":      e.get("element_type"),
            "position":          pos_map.get(e.get("element_type"), "UNK"),
            "now_cost":          e.get("now_cost"),
            "total_points":      e.get("total_points"),
            "minutes":           e.get("minutes"),
            "goals_scored":      e.get("goals_scored"),
            "assists":           e.get("assists"),
            "clean_sheets":      e.get("clean_sheets"),
            "goals_conceded":    e.get("goals_conceded"),
            "own_goals":         e.get("own_goals"),
            "penalties_saved":   e.get("penalties_saved"),
            "penalties_missed":  e.get("penalties_missed"),
            "yellow_cards":      e.get("yellow_cards"),
            "red_cards":         e.get("red_cards"),
            "saves":             e.get("saves"),
            "bonus":             e.get("bonus"),
            "bps":               e.get("bps"),
            "influence":         e.get("influence"),
            "creativity":        e.get("creativity"),
            "threat":            e.get("threat"),
            "ict_index":         e.get("ict_index"),
            "starts":            e.get("starts"),
            "expected_goals":    e.get("expected_goals"),
            "expected_assists":  e.get("expected_assists"),
            "expected_goal_involvements": e.get("expected_goal_involvements"),
            "expected_goals_conceded":    e.get("expected_goals_conceded"),
            "form":              e.get("form"),
            "points_per_game":   e.get("points_per_game"),
            "selected_by_percent": e.get("selected_by_percent"),
            "status":            e.get("status"),
            # 2025-26以降
            "tackles":                         e.get("tackles", 0),
            "recoveries":                      e.get("recoveries", 0),
            "clearances_blocks_interceptions": e.get("clearances_blocks_interceptions", 0),
            "defensive_contribution":          e.get("defensive_contribution", 0),
        })
    return pd.DataFrame(rows)


def fetch_gw_data(elements: list, teams: list,
                  target_gws: list | None = None,
                  sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    全選手のelement-summaryを取得してGW別データを構築。
    target_gws が指定された場合はそのGWのみ抽出。
    """
    team_map     = {t["id"]: t["name"] for t in teams}
    opp_team_map = team_map.copy()

    all_rows  = []
    n_players = len(elements)
    errors    = 0

    print(f"\n📊 GW別データを取得中（全{n_players}選手）...")
    print(f"   対象GW: {target_gws if target_gws else '全GW'}")
    print(f"   推定時間: 約{n_players * sleep_sec / 60:.1f}分\n")

    start = time.time()

    for i, e in enumerate(elements):
        pid   = e["id"]
        pname = e.get("web_name", str(pid))
        team  = team_map.get(e.get("team"), "")

        if i % 50 == 0:
            elapsed = time.time() - start
            remaining = (n_players - i) * sleep_sec
            print(f"  [{i:4d}/{n_players}] 経過: {elapsed:.0f}s  残り: 約{remaining:.0f}s  "
                  f"エラー: {errors}", flush=True)

        summary = fpl_get(f"element-summary/{pid}/")
        if not summary or "history" not in summary:
            errors += 1
            time.sleep(sleep_sec)
            continue

        for h in summary["history"]:
            gw = h.get("round")
            if target_gws and gw not in target_gws:
                continue

            row = {
                "name":        pname,
                "position":    {1:"GK",2:"DEF",3:"MID",4:"FWD"}.get(e.get("element_type"),"UNK"),
                "team":        team,
                "opponent":    opp_team_map.get(h.get("opponent_team"), ""),
                "xP":          h.get("expected_points", 0),
                "GW":          gw,
            }
            # GW_COL_MAP に従って全列を追加
            for api_key, csv_key in GW_COL_MAP.items():
                row[csv_key] = h.get(api_key, 0)

            # was_home をboolに変換
            row["was_home"] = bool(h.get("was_home", False))

            # team_h_score / team_a_score
            row["team_h_score"] = h.get("team_h_score")
            row["team_a_score"] = h.get("team_a_score")

            all_rows.append(row)

        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    print(f"\n  取得完了: {len(df)}行  エラー: {errors}選手")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="FPL公式APIからvaastav形式のCSVを生成"
    )
    parser.add_argument(
        "--gw", type=int, nargs="*",
        help="取得するGW番号（省略時は全GW）。例: --gw 1 2"
    )
    parser.add_argument(
        "--out", type=str, default=".",
        help="出力先ディレクトリ（デフォルト: カレント）"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.2,
        help="選手間のスリープ秒数（デフォルト: 0.2）"
    )
    parser.add_argument(
        "--players-only", action="store_true",
        help="players_raw.csv と teams.csv のみ生成（高速）"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 58)
    print("  FPL公式APIデータ取得スクリプト")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 58)

    # ── bootstrap取得 ──
    bootstrap = fetch_bootstrap()
    if not bootstrap:
        print("❌ bootstrap-static の取得に失敗しました")
        print("   FPL公式サイトにアクセスできない環境かもしれません")
        sys.exit(1)

    elements = bootstrap["elements"]
    teams    = bootstrap["teams"]
    events   = bootstrap["events"]

    # 終了済みGWを確認
    finished_gws = [e["id"] for e in events if e.get("finished")]
    current_gw   = next((e["id"] for e in events if e.get("is_current")), None)
    print(f"\n✅ 接続OK")
    print(f"   登録選手数: {len(elements)}")
    print(f"   チーム数:   {len(teams)}")
    print(f"   終了済みGW: {finished_gws}")
    print(f"   現在のGW:   {current_gw}")

    if not finished_gws:
        print("\n⚠️  まだ終了したGWがありません")
        sys.exit(0)

    # target GWs
    if args.gw:
        target_gws = args.gw
    else:
        target_gws = finished_gws
    print(f"   取得対象GW: {target_gws}")

    # ── teams.csv ──
    df_teams = build_teams_csv(teams)
    teams_path = os.path.join(args.out, "teams.csv")
    df_teams.to_csv(teams_path, index=False)
    print(f"\n💾 teams.csv → {teams_path}  ({len(df_teams)}チーム)")

    # ── players_raw.csv ──
    df_players = build_players_raw(elements, teams)
    players_path = os.path.join(args.out, "players_raw.csv")
    df_players.to_csv(players_path, index=False)
    print(f"💾 players_raw.csv → {players_path}  ({len(df_players)}選手)")

    if args.players_only:
        print("\n✅ --players-only モード完了")
        return

    # ── merged_gw.csv ──
    df_gw = fetch_gw_data(elements, teams, target_gws, args.sleep)
    if df_gw.empty:
        print("❌ GWデータが空です")
        sys.exit(1)

    gw_path = os.path.join(args.out, "merged_gw.csv")

    # 既存ファイルがある場合はマージ（古いGWを保持）
    if os.path.exists(gw_path) and args.gw:
        try:
            df_existing = pd.read_csv(gw_path)
            existing_gws = df_existing["GW"].dropna().unique() if "GW" in df_existing.columns else []
            df_existing = df_existing[~df_existing["GW"].isin(target_gws)]
            df_gw = pd.concat([df_existing, df_gw], ignore_index=True)
            print(f"  既存データ({list(existing_gws)})に新GWをマージ")
        except Exception as e:
            print(f"  ⚠️  マージ失敗（上書き）: {e}")

    df_gw = df_gw.sort_values(["GW", "element"]).reset_index(drop=True)
    df_gw.to_csv(gw_path, index=False)
    print(f"💾 merged_gw.csv → {gw_path}  ({len(df_gw)}行, GW: {sorted(df_gw['GW'].dropna().unique().astype(int))})")

    print(f"""
{"=" * 58}
  ✅ 完了！
{"=" * 58}
  次のステップ:
  1. 生成されたCSVファイルをGitHubにアップロード
     （既存ファイルを上書き）
  2. アプリをリロード → 最新GWデータが反映されます

  毎週の更新:
    python fetch_fpl_data.py --gw {max(target_gws) + 1 if target_gws else 'N'}
    （新しいGWのみ差分取得・既存データにマージ）
{"=" * 58}
""")


if __name__ == "__main__":
    main()
