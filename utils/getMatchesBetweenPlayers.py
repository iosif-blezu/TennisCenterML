import pandas as pd
from utils.predict_between_two_players import predict_match
pd.set_option('display.max_columns', None)

def getMatchesBetweenPlayers(player1_id, player2_id):
    """
    Get all matches between two players from the cleaned dataset.
    :param player1_id: ID of player 1
    :param player2_id: ID of player 2
    :param df: DataFrame containing the cleaned dataset
    :return: DataFrame containing all matches between the two players
    """
    df = pd.read_csv("../data/cleanedDataset.csv")
    # Filter the DataFrame for matches between the two players
    matches = df[((df['p1_id'] == player1_id) & (df['p2_id'] == player2_id)) |
                 ((df['p1_id'] == player2_id) & (df['p2_id'] == player1_id))]

    return matches


def main():
    clean_data = pd.read_csv("../data/cleanedDataset.csv")
    player1_id = 134770
    player2_id = 126203
    matches = getMatchesBetweenPlayers(player1_id, player2_id, clean_data)
    print(f"Found {len(matches)} matches between player {player1_id} and player {player2_id}.")
    for match in matches.itertuples():
        print(f"Player 1: {match.p1_name}, Player 2: {match.p2_name}  Winner: {match.RESULT}, Score: {match.score}, Surface: {match.surface}, Tournament Date: {match.tourney_date}, Tournament: {match.tourney_name}, Round: {match.round}, Best of: {match.best_of}")

    result = predict_match(
        player2_id=134770,  # Casper Ruud
        player1_id=126203,  # Taylor Fritz
        surface="Clay",
        best_of=3,
        draw_size=128
    )
    print(result)


if __name__ == '__main__':
    main()