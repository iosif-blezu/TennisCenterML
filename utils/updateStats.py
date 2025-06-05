# updateStats.py
import numpy as np
from collections import defaultdict, deque


# --- Define factory functions at the top level ---
def int_factory():
    return 0  # or just use int directly: defaultdict(int)


def nested_int_defaultdict_factory():
    return defaultdict(int)  # or defaultdict(int_factory)


def deque_factory():
    return deque()


def nested_deque_defaultdict_factory():
    return defaultdict(deque_factory)  # or defaultdict(lambda: deque()) if lambda is top-level


def createStats():
    # import numpy as np # Already imported at top
    # from collections import defaultdict, deque # Already imported at top

    prev_stats = {}

    # Use the top-level factory functions or direct types
    prev_stats["elo_players"] = defaultdict(int)  # int is fine as it's a built-in type
    prev_stats["elo_surface_players"] = defaultdict(nested_int_defaultdict_factory)
    prev_stats["elo_grad_players"] = defaultdict(lambda: deque(maxlen=1000))  # deque(maxlen=...) is fine
    prev_stats["last_k_matches"] = defaultdict(lambda: deque(maxlen=1000))  # deque(maxlen=...) is fine
    prev_stats["last_k_matches_stats"] = defaultdict(
        lambda: defaultdict(lambda: deque(maxlen=1000)))  # complex, see below

    prev_stats["matches_played"] = defaultdict(int)
    prev_stats["h2h"] = defaultdict(int)  # For simple (w_id, l_id) keys
    # If h2h_surface keys are simple (e.g. (w_id,l_id) after surface key), int is fine.
    prev_stats["h2h_surface"] = defaultdict(nested_int_defaultdict_factory)

    # --- For full ELO history tracking ---
    prev_stats["elo_history_players"] = defaultdict(deque_factory)  # Uses top-level deque_factory
    prev_stats["elo_surface_history_players"] = defaultdict(nested_deque_defaultdict_factory)  # Uses top-level factory

    # For last_k_matches_stats which is defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
    # This is nested. Let's define a factory for the inner part.
    def inner_last_k_stats_factory():
        return defaultdict(lambda: deque(maxlen=1000))

    prev_stats["last_k_matches_stats"] = defaultdict(inner_last_k_stats_factory)

    return prev_stats


# ... (rest of your updateStats.py: updateStats function, getStats function) ...

# The updateStats and getStats functions remain the same as in the previous correct version.
# Just ensure the imports of `mean` and `getWinnerLoserIDS` from `utils.common` are correct.

# (The rest of updateStats and getStats functions as provided in the previous correct answer)
# ... (Make sure to copy the full correct versions of updateStats and getStats here) ...
def updateStats(match, prev_stats):
    from utils.common import mean, getWinnerLoserIDS  # Assuming these are in utils.common
    import numpy as np
    from collections import deque  # Ensure deque is available if not already

    # Get Winner and Loser ID'S
    p1_id, p2_id, surface, result = match.p1_id, match.p2_id, match.surface, match.RESULT
    w_id, l_id = getWinnerLoserIDS(p1_id, p2_id, result)

    # update
    # elo
    # Get current ELO ratings (before the match)
    elo_w_before = prev_stats["elo_players"].get(w_id, 1500)
    elo_l_before = prev_stats["elo_players"].get(l_id, 1500)
    elo_surface_w_before = prev_stats["elo_surface_players"][surface].get(w_id, 1500)
    elo_surface_l_before = prev_stats["elo_surface_players"][surface].get(l_id, 1500)

    # Calculate expected probabilities
    k_elo_factor = 24  # Standard ELO k-factor
    exp_w = 1 / (1 + (10 ** ((elo_l_before - elo_w_before) / 400)))
    # exp_l = 1 / (1 + (10 ** ((elo_w_before - elo_l_before) / 400))) # exp_l is 1 - exp_w
    exp_surface_w = 1 / (1 + (10 ** ((elo_surface_l_before - elo_surface_w_before) / 400)))
    # exp_surface_l = 1 / (1 + (10 ** ((elo_surface_w_before - elo_surface_l_before) / 400))) # exp_surface_l is 1 - exp_surface_w

    # Update ELO ratings for next match
    elo_w_after = elo_w_before + k_elo_factor * (1 - exp_w)
    elo_l_after = elo_l_before + k_elo_factor * (0 - (1 - exp_w))  # or k_elo_factor * (0 - exp_l)
    elo_surface_w_after = elo_surface_w_before + k_elo_factor * (1 - exp_surface_w)
    elo_surface_l_after = elo_surface_l_before + k_elo_factor * (
                0 - (1 - exp_surface_w))  # or k_elo_factor * (0 - exp_surface_l)

    # Store updated current ratings
    prev_stats["elo_players"][w_id] = elo_w_after
    prev_stats["elo_players"][l_id] = elo_l_after
    prev_stats["elo_surface_players"][surface][w_id] = elo_surface_w_after
    prev_stats["elo_surface_players"][surface][l_id] = elo_surface_l_after

    # --- Store ELO history (after update) ---
    prev_stats["elo_history_players"][w_id].append(elo_w_after)
    prev_stats["elo_history_players"][l_id].append(elo_l_after)

    # Store history for the current match's surface
    prev_stats["elo_surface_history_players"][surface][w_id].append(elo_surface_w_after)
    prev_stats["elo_surface_history_players"][surface][l_id].append(elo_surface_l_after)

    _defined_surfaces_for_padding = ["Clay", "Grass", "Hard", "Carpet"]

    for s_other in _defined_surfaces_for_padding:
        if surface != s_other:
            current_elo_w_other_surface = prev_stats["elo_surface_players"][s_other].get(w_id, 1500)
            prev_stats["elo_surface_history_players"][s_other][w_id].append(current_elo_w_other_surface)
            current_elo_l_other_surface = prev_stats["elo_surface_players"][s_other].get(l_id, 1500)
            prev_stats["elo_surface_history_players"][s_other][l_id].append(current_elo_l_other_surface)
    # --- End of ELO history update ---

    prev_stats["elo_grad_players"][w_id].append(elo_w_after)
    prev_stats["elo_grad_players"][l_id].append(elo_l_after)

    prev_stats["matches_played"][w_id] += 1
    prev_stats["matches_played"][l_id] += 1

    prev_stats["last_k_matches"][w_id].append(1)
    prev_stats["last_k_matches"][l_id].append(0)

    prev_stats["h2h"][(w_id, l_id)] += 1
    prev_stats["h2h_surface"][surface][(w_id, l_id)] += 1

    if p1_id == w_id:
        w_ace, l_ace = match.p1_ace, match.p2_ace
        w_df, l_df = match.p1_df, match.p2_df
        w_svpt, l_svpt = match.p1_svpt, match.p2_svpt
        w_1stIn, l_1stIn = match.p1_1stIn, match.p2_1stIn
        w_1stWon, l_1stWon = match.p1_1stWon, match.p2_1stWon
        w_2ndWon, l_2ndWon = match.p1_2ndWon, match.p2_2ndWon
        w_bpSaved, l_bpSaved = match.p1_bpSaved, match.p2_bpSaved
        w_bpFaced, l_bpFaced = match.p1_bpFaced, match.p2_bpFaced
    else:
        w_ace, l_ace = match.p2_ace, match.p1_ace
        w_df, l_df = match.p2_df, match.p1_df
        w_svpt, l_svpt = match.p2_svpt, match.p1_svpt
        w_1stIn, l_1stIn = match.p2_1stIn, match.p1_1stIn
        w_1stWon, l_1stWon = match.p2_1stWon, match.p1_1stWon
        w_2ndWon, l_2ndWon = match.p2_2ndWon, match.p1_2ndWon
        w_bpSaved, l_bpSaved = match.p2_bpSaved, match.p1_bpSaved
        w_bpFaced, l_bpFaced = match.p2_bpFaced, match.p1_bpFaced

    if w_svpt > 0:
        prev_stats["last_k_matches_stats"][w_id]["p_ace"].append(100 * (w_ace / w_svpt) if w_svpt > 0 else 0)
        prev_stats["last_k_matches_stats"][w_id]["p_df"].append(100 * (w_df / w_svpt) if w_svpt > 0 else 0)
        prev_stats["last_k_matches_stats"][w_id]["p_1stIn"].append(100 * (w_1stIn / w_svpt) if w_svpt > 0 else 0)
        if w_1stIn > 0:
            prev_stats["last_k_matches_stats"][w_id]["p_1stWon"].append(100 * (w_1stWon / w_1stIn))
        if (w_svpt - w_1stIn) > 0:
            prev_stats["last_k_matches_stats"][w_id]["p_2ndWon"].append(100 * (w_2ndWon / (w_svpt - w_1stIn)))
    if w_bpFaced > 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpSaved"].append(100 * (w_bpSaved / w_bpFaced))

    if l_svpt > 0:
        prev_stats["last_k_matches_stats"][l_id]["p_ace"].append(100 * (l_ace / l_svpt) if l_svpt > 0 else 0)
        prev_stats["last_k_matches_stats"][l_id]["p_df"].append(100 * (l_df / l_svpt) if l_svpt > 0 else 0)
        prev_stats["last_k_matches_stats"][l_id]["p_1stIn"].append(100 * (l_1stIn / l_svpt) if l_svpt > 0 else 0)
        if l_1stIn > 0:
            prev_stats["last_k_matches_stats"][l_id]["p_1stWon"].append(100 * (l_1stWon / l_1stIn))
        if (l_svpt - l_1stIn) > 0:
            prev_stats["last_k_matches_stats"][l_id]["p_2ndWon"].append(100 * (l_2ndWon / (l_svpt - l_1stIn)))
    if l_bpFaced > 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpSaved"].append(100 * (l_bpSaved / l_bpFaced))

    return prev_stats


def getStats(player1, player2, match, prev_stats):
    from utils.common import mean, getWinnerLoserIDS
    import numpy as np

    output = {}
    PLAYER1_ID = player1["ID"]
    PLAYER2_ID = player2["ID"]
    SURFACE = match["SURFACE"]

    output["BEST_OF"] = match["BEST_OF"]
    output["DRAW_SIZE"] = match["DRAW_SIZE"]
    output["AGE_DIFF"] = player1["AGE"] - player2["AGE"]
    output["HEIGHT_DIFF"] = player1["HEIGHT"] - player2["HEIGHT"]
    output["ATP_RANK_DIFF"] = player1["ATP_RANK"] - player2["ATP_RANK"]
    output["ATP_POINTS_DIFF"] = player1["ATP_POINTS"] - player2["ATP_POINTS"]

    elo_p1 = prev_stats["elo_players"].get(PLAYER1_ID, 1500)
    elo_p2 = prev_stats["elo_players"].get(PLAYER2_ID, 1500)
    output["ELO_DIFF"] = elo_p1 - elo_p2

    elo_surface_p1 = prev_stats["elo_surface_players"][SURFACE].get(PLAYER1_ID, 1500)
    elo_surface_p2 = prev_stats["elo_surface_players"][SURFACE].get(PLAYER2_ID, 1500)
    output["ELO_SURFACE_DIFF"] = elo_surface_p1 - elo_surface_p2

    matches_played = prev_stats["matches_played"]
    h2h = prev_stats["h2h"]
    h2h_surface = prev_stats["h2h_surface"]
    last_k_matches = prev_stats["last_k_matches"]
    last_k_matches_stats = prev_stats["last_k_matches_stats"]
    elo_grad_players = prev_stats["elo_grad_players"]

    output["N_GAMES_DIFF"] = matches_played.get(PLAYER1_ID, 0) - matches_played.get(PLAYER2_ID, 0)
    output["H2H_DIFF"] = h2h.get((PLAYER1_ID, PLAYER2_ID), 0) - h2h.get((PLAYER2_ID, PLAYER1_ID), 0)
    output["H2H_SURFACE_DIFF"] = h2h_surface[SURFACE].get((PLAYER1_ID, PLAYER2_ID), 0) - \
                                 h2h_surface[SURFACE].get((PLAYER2_ID, PLAYER1_ID), 0)

    for k_val in [3, 5, 10, 25, 50, 100, 200]:
        p1_last_k = list(last_k_matches[PLAYER1_ID])
        p2_last_k = list(last_k_matches[PLAYER2_ID])
        if len(p1_last_k) >= k_val and len(p2_last_k) >= k_val:
            output[f"WIN_LAST_{k_val}_DIFF"] = sum(p1_last_k[-k_val:]) - sum(p2_last_k[-k_val:])
        else:
            output[f"WIN_LAST_{k_val}_DIFF"] = sum(p1_last_k) - sum(p2_last_k) if (
                        len(p1_last_k) > 0 or len(p2_last_k) > 0) else 0

        elo_hist_p1 = list(elo_grad_players[PLAYER1_ID])
        elo_hist_p2 = list(elo_grad_players[PLAYER2_ID])

        if len(elo_hist_p1) >= k_val and len(elo_hist_p2) >= k_val and k_val > 1:
            slope_1 = np.polyfit(np.arange(k_val), np.array(elo_hist_p1[-k_val:]), 1)[0]
            slope_2 = np.polyfit(np.arange(k_val), np.array(elo_hist_p2[-k_val:]), 1)[0]
            output[f"ELO_GRAD_LAST_{k_val}_DIFF"] = slope_1 - slope_2
        elif len(elo_hist_p1) >= 2 and len(elo_hist_p2) >= 2 and k_val > 1:
            slope_1 = np.polyfit(np.arange(len(elo_hist_p1)), np.array(elo_hist_p1), 1)[0]
            slope_2 = np.polyfit(np.arange(len(elo_hist_p2)), np.array(elo_hist_p2), 1)[0]
            output[f"ELO_GRAD_LAST_{k_val}_DIFF"] = slope_1 - slope_2
        else:
            output[f"ELO_GRAD_LAST_{k_val}_DIFF"] = 0

        stats_to_calc = ["p_ace", "p_df", "p_1stIn", "p_1stWon", "p_2ndWon", "p_bpSaved"]
        for stat_name in stats_to_calc:
            p1_stat_hist = list(last_k_matches_stats[PLAYER1_ID][stat_name])
            p2_stat_hist = list(last_k_matches_stats[PLAYER2_ID][stat_name])

            mean_p1_stat = mean(p1_stat_hist[-k_val:]) if len(p1_stat_hist) >= k_val else mean(p1_stat_hist)
            mean_p2_stat = mean(p2_stat_hist[-k_val:]) if len(p2_stat_hist) >= k_val else mean(p2_stat_hist)

            output[f"{stat_name.upper()}_LAST_{k_val}_DIFF"] = mean_p1_stat - mean_p2_stat

    return output


if __name__ == '__main__':
    pass