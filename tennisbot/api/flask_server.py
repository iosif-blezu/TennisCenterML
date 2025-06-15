import os
from dotenv import load_dotenv
from flask_cors import CORS
import json
import numpy as np
import requests

from utils.predict_between_two_players import  predict_between_two_players
import urllib.parse
from bs4 import BeautifulSoup, Tag



RPM_CAP = 80 # requests per minute, to avoid hitting the API too hard
MAX_WORKERS = 6 # parallel cleaner threads

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    project_root_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(project_root_env_path):
        load_dotenv(project_root_env_path)
    else:
        load_dotenv()

from flask import Flask, request, jsonify
from tennisbot.agent.router import get_router_agent
from tennisbot.config import get_settings
from utils.head_to_head import head_to_head, _dataset


app = Flask(__name__)
CORS(app, origins=["http://localhost:3001", "http://localhost:3000"])

print("FLASK_SERVER: Loading settings…")
cfg = get_settings()

print("FLASK_SERVER: Instantiating router agent…")
agent = get_router_agent()
print("FLASK_SERVER: Agent ready.")

_prev_stats_data = None
PREV_STATS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "prev_stats.json")


def parse_str_tuple_key(key_str):
    """Converts a string like '(1, 2)' back to a tuple (1, 2)."""
    try:
        # Remove parentheses and split by comma
        parts = key_str.strip("()").split(",")
        return tuple(int(p.strip()) for p in parts)
    except:
        return key_str  # Return original if parsing fails (should not happen for valid keys)


def load_previous_stats():
    global _prev_stats_data
    if _prev_stats_data is None:
        try:
            print(f"FLASK_SERVER: Attempting to load player stats from {PREV_STATS_FILE_PATH}...")
            if os.path.exists(PREV_STATS_FILE_PATH):
                with open(PREV_STATS_FILE_PATH, "r") as f:
                    # JSON data is loaded as standard dicts and lists
                    _prev_stats_data = json.load(f)
                print("FLASK_SERVER: Successfully loaded player stats data (prev_stats.json).")
            else:
                print(f"FLASK_SERVER: ERROR - Player stats file not found at {PREV_STATS_FILE_PATH}.")
                _prev_stats_data = {}
        except Exception as e:
            print(f"FLASK_SERVER: ERROR loading player stats data (prev_stats.json): {e}")
            import traceback
            traceback.print_exc()
            _prev_stats_data = {}
    return _prev_stats_data


load_previous_stats()



@app.route("/chat", methods=["POST"])
def chat_handler():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        result = agent.invoke({"input": user_input})
        mem = agent.memory.load_memory_variables({})
        print("===== CHAT MEMORY =====")
        for msg in mem["chat_history"]:
            print(f"{type(msg).__name__}: {msg.content}")
        print("=======================")
        out = result.get("output") if isinstance(result, dict) else str(result)
        return jsonify({"output": out})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats/elo_history", methods=["GET"])
def elo_history_handler():
    stats_data = load_previous_stats()

    if not stats_data or not _prev_stats_data:
        return jsonify({"error": "Player statistics data is not available or failed to load."}), 503

    player_id_str = request.args.get("player_id")
    if not player_id_str:
        return jsonify({"error": "player_id query parameter is required."}), 400

    try:
        player_id_key = player_id_str
        int(player_id_str)
    except ValueError:
        return jsonify({"error": "player_id must be an integer string."}), 400

    elo_history_all_players = stats_data.get("elo_history_players", {})
    surface_elo_history_all_players = stats_data.get("elo_surface_history_players", {})

    player_general_elo_history = elo_history_all_players.get(player_id_key, [])

    player_surface_elo_data = {}
    for surface, players_on_surface in surface_elo_history_all_players.items():
        history = players_on_surface.get(player_id_key, [])
        if history:
            player_surface_elo_data[surface] = history

    if not player_general_elo_history and not player_surface_elo_data:
        return jsonify({"error": f"No ELO history found for player_id {player_id_key}."}), 404

    return jsonify({
        "player_id": player_id_key,
        "general_elo": player_general_elo_history,
        "surface_elo": player_surface_elo_data
    })

@app.route("/stats/head_to_head", methods=["GET"])
def head_to_head_handler():
    try:
        p1 = int(request.args["player1_id"])
        p2 = int(request.args["player2_id"])
    except (KeyError, ValueError):
        return jsonify({"error": "player1_id and player2_id query params (ints) are required"}), 400

    data = head_to_head(p1, p2)
    status = 200 if data["total_matches"] else 404
    return jsonify(data), status

@app.route("/stats/player", methods=["GET"])
def player_stats_handler():
    stats = load_previous_stats()
    if not stats:
        return jsonify({"error": "Player statistics data is not available."}), 503

    pid = request.args.get("player_id")
    if not pid or not pid.isdigit():
        return jsonify({"error": "player_id query parameter (integer) is required."}), 400

    last_k            = list(stats.get("last_k_matches",       {}).get(pid, []))
    lk_stats          = stats.get   ("last_k_matches_stats", {}).get(pid, {})
    elo_players       = stats.get   ("elo_players",           {})
    elo_surfaces      = stats.get   ("elo_surface_players",  {})
    elo_grad_players  = stats.get   ("elo_grad_players",     {})

    def rate(seq, k=None):
        if not seq: return None
        s = seq[-k:] if (k and len(seq)>=k) else seq
        return sum(s)/len(s)
    winrates_last = {f"{k}": rate(last_k, k) for k in (5, 10, 20, 50, 100)}
    career_wr     = rate(last_k)

    def longest_streak(seq):
        best = cur = 0
        for x in seq:
            if x==1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best
    current_streak = 0
    for x in reversed(last_k):
        if x==1: current_streak += 1
        else:    break
    longest_wr_streak = longest_streak(last_k)

    current_elo = elo_players.get(pid, 1500)
    elo_by_surface = {
        surface: d.get(pid, 1500)
        for surface, d in elo_surfaces.items()
    }
    def elo_slope(history, k):
        h = history[-k:] if len(history)>=k else history
        if len(h) < 2: return None
        xs = np.arange(len(h))
        return float(np.polyfit(xs, h, 1)[0])

    elo_hist = list(elo_grad_players.get(pid, []))
    slope_20 = elo_slope(elo_hist, 20)
    slope_50 = elo_slope(elo_hist, 50)

    career_avg = {
        metric: (sum(vals)/len(vals)) if vals else None
        for metric, vals in lk_stats.items()
    }

    resp = {
        "player_id":            pid,
        "total_matches":        len(last_k),
        "career_winrate":       career_wr,
        "winrates_last":        winrates_last,
        "current_win_streak":   current_streak,
        "longest_win_streak":   longest_wr_streak,
        "elo": {
            "current":           current_elo,
            "by_surface":        elo_by_surface,
            "trend_slope_last_20": slope_20,
            "trend_slope_last_50": slope_50
        },
        "career_avg_stats":     career_avg
    }

    return jsonify(resp), 200

def _compute_surface_stats(df_surf, pid):
    """
    df_surf : DataFrame of all matches for player pid on ONE surface
    pid     : int  (player_id)
    Returns : dict  with win-rates, streaks, career averages, etc.
    """
    if df_surf.empty:
        empty_block = {
            "total_matches": 0,
            "career_winrate": None,
            "winrates_last": {str(k): None for k in (5, 10, 20, 50, 100)},
            "current_win_streak": 0,
            "longest_win_streak": 0,
            "career_avg_stats": {m: None for m in
                                 ("ace", "df", "1stIn", "1stWon", "2ndWon", "bpSaved")}
        }
        return empty_block

    # Chronological order
    df = df_surf.sort_values("tourney_date")

    wins      = []
    stat_bkt  = {m: [] for m in ("ace", "df", "1stIn", "1stWon", "2ndWon", "bpSaved")}

    for _, row in df.iterrows():
        res, p1, p2 = row["RESULT"], row["p1_id"], row["p2_id"]

        if isinstance(res, (int, float)):
            if res in (0, 1) and ((res == 1 and p1 == pid) or (res == 0 and p2 == pid)):
                wins.append(1)
            elif res == pid:
                wins.append(1)
            else:
                wins.append(0)
        else:
            if (res == "Player 1" and p1 == pid) or (res == "Player 2" and p2 == pid):
                wins.append(1)
            else:
                wins.append(0)

        pref = "p1_" if p1 == pid else "p2_"

        svpt      = row[f"{pref}svpt"]     or 0
        ace       = row[f"{pref}ace"]      or 0
        dfault    = row[f"{pref}df"]       or 0
        first_in  = row[f"{pref}1stIn"]    or 0
        first_won = row[f"{pref}1stWon"]   or 0
        second_won= row[f"{pref}2ndWon"]   or 0
        bp_saved  = row[f"{pref}bpSaved"]  or 0
        bp_faced  = row[f"{pref}bpFaced"]  or 0

        if svpt:
            stat_bkt["ace"].append(     100 * ace      / svpt)
            stat_bkt["df"].append(      100 * dfault   / svpt)
            stat_bkt["1stIn"].append(   100 * first_in / svpt)
        else:
            stat_bkt["ace"].append(0);  stat_bkt["df"].append(0); stat_bkt["1stIn"].append(0)

        if first_in:
            stat_bkt["1stWon"].append( 100 * first_won / first_in)
        else:
            stat_bkt["1stWon"].append(0)

        if svpt - first_in:
            stat_bkt["2ndWon"].append(100 * second_won / (svpt - first_in))
        else:
            stat_bkt["2ndWon"].append(0)

        if bp_faced:
            stat_bkt["bpSaved"].append(100 * bp_saved / bp_faced)
        else:
            stat_bkt["bpSaved"].append(0)

    def rate(seq, k=None):
        if not seq:
            return None
        s = seq[-k:] if (k and len(seq) >= k) else seq
        return sum(s) / len(s)

    def longest_streak(seq):
        best = cur = 0
        for w in seq:
            if w: cur += 1; best = max(best, cur)
            else: cur = 0
        return best

    cur_streak = 0
    for w in reversed(wins):
        if w: cur_streak += 1
        else: break

    block = {
        "total_matches": len(wins),
        "career_winrate": rate(wins),
        "winrates_last": {str(k): rate(wins, k) for k in (5, 10, 20, 50, 100)},
        "current_win_streak": cur_streak,
        "longest_win_streak": longest_streak(wins),
        "career_avg_stats": {
            m: (sum(v) / len(v) if v else None)
            for m, v in stat_bkt.items()
        }
    }
    return block

@app.route("/stats/player_surfaces", methods=["GET"])
def player_surfaces_handler():
    pid = request.args.get("player_id")
    if not pid or not pid.isdigit():
        return jsonify({"error": "player_id (integer) is required"}), 400
    pid = int(pid)

    # Load full dataset (cached)
    df = _dataset()

    # Only these three surfaces
    surfaces = ["Clay", "Grass", "Hard"]
    out = {"player_id": pid, "surfaces": {}}

    # Filter once per surface
    for surf in surfaces:
        df_surf = df[
            ((df["p1_id"] == pid) | (df["p2_id"] == pid)) &
            (df["surface"] == surf)
        ]
        out["surfaces"][surf] = _compute_surface_stats(df_surf, pid)

    return jsonify(out), 200


@app.route("/news/tennisexplorer", methods=["GET"])
def tennisexplorer_news_handler():
    # how many pages to grab, up to 5
    pages = min(max(int(request.args.get("pages", 1)), 1), 5)
    base_url = "https://www.tennisexplorer.com/tennis-news/?page={}"
    buckets = {}

    for page in range(1, pages + 1):
        resp = requests.get(base_url.format(page))
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")  # ← use html.parser
        center = soup.find("div", id="center")
        if not center:
            continue

        for header in center.find_all("b"):
            category = header.get_text(strip=True)
            items = []

            for sib in header.next_siblings:
                if isinstance(sib, Tag) and sib.name == "b":
                    break
                if isinstance(sib, Tag) and sib.name == "a":
                    href = sib["href"]
                    if href.startswith("/redirect/"):
                        qs = urllib.parse.urlparse(href).query
                        actual = urllib.parse.parse_qs(qs).get("url", [""])[0]
                    else:
                        actual = urllib.parse.urljoin("https://www.tennisexplorer.com", href)
                    title = sib.get_text(strip=True)
                    items.append({"title": title, "url": actual})

            if items:
                buckets.setdefault(category, []).extend(items)

    return jsonify(buckets), 200

@app.route("/predict", methods=["GET"])
def predict_handler():
    try:
        p1 = int(request.args.get("player1_id"))
        p2 = int(request.args.get("player2_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "player1_id and player2_id are required integers"}), 400

    surface = request.args.get("surface", "Hard")
    try:
        best_of   = int(request.args.get("best_of", 3))
        draw_size = int(request.args.get("draw_size", 128))
    except ValueError:
        return jsonify({"error": "best_of and draw_size must be integers"}), 400

    result = predict_between_two_players(
        player1_id=p1,
        player2_id=p2,
        surface=surface,
        best_of=best_of,
        draw_size=draw_size
    )

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)