import os
from dotenv import load_dotenv
from flask_cors import CORS
import json


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

app = Flask(__name__)
CORS(app, origins=["http://localhost:3001", "http://localhost:3000"])

print("FLASK_SERVER: Loading settings…")
cfg = get_settings()
# ... (rest of your settings print statements) ...

print("FLASK_SERVER: Instantiating router agent…")
agent = get_router_agent()
print("FLASK_SERVER: Agent ready.")

# --- Additions for ELO Stats ---
_prev_stats_data = None
PREV_STATS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                    "prev_stats.json")  # <-- Changed to .json


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
        # Player IDs in the JSON are strings because all keys are strings if they were ints
        # However, if the player_id itself is a key in a sub-dictionary,
        # like elo_history_players[player_id], it might have been an int in Python
        # and then converted to string by the outer dict key conversion.
        # For consistency, we'll expect the player_id query param as a string,
        # and use it as a string to look up in our loaded JSON (which is now dicts).
        player_id_key = player_id_str  # Use as string for dictionary lookup
        int(player_id_str)  # Validate it can be an int, but use string key
    except ValueError:
        return jsonify({"error": "player_id must be an integer string."}), 400

    # Data is already in list/dict format from json.load()
    elo_history_all_players = stats_data.get("elo_history_players", {})
    surface_elo_history_all_players = stats_data.get("elo_surface_history_players", {})

    # Player IDs will be strings as keys in these dictionaries after JSON load
    player_general_elo_history = elo_history_all_players.get(player_id_key, [])

    player_surface_elo_data = {}
    # Surface names (e.g., "Hard", "Clay") are keys at the first level of surface_elo_history_all_players
    for surface, players_on_surface in surface_elo_history_all_players.items():
        # player_id_key is used here as well
        history = players_on_surface.get(player_id_key, [])
        if history:
            player_surface_elo_data[surface] = history  # Already a list

    if not player_general_elo_history and not player_surface_elo_data:
        return jsonify({"error": f"No ELO history found for player_id {player_id_key}."}), 404

    return jsonify({
        "player_id": player_id_key,  # Return the ID as received (string)
        "general_elo": player_general_elo_history,
        "surface_elo": player_surface_elo_data
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)