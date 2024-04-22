import argparse
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def pre(player_df):
    """
    Preprocess the data. This function should be used to clean and
    transform the data as needed. The function should return a pandas
    DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be cleaned.
    
    Returns
    -------
    df : pandas.DataFrame
        The cleaned dataset.
    """
    team_df = pd.read_csv("team_traditional.csv")

     # Load team data
    
    print("Team DF columns:", team_df.columns)  # Initial check

    # Self-merge to find opposing team points in the same game
    merged_team_df = pd.merge(team_df, team_df, on='gameid', suffixes=('', '_opponent'))
    print("Merged Team DF columns:", merged_team_df.columns)  # Check after merge

    merged_team_df = merged_team_df[merged_team_df['team'] != merged_team_df['team_opponent']]

    # Assign points scored by the opponent as points against
    merged_team_df['PTS_Against'] = merged_team_df['PTS_opponent']
    merged_team_df['opponent_team'] = merged_team_df['team_opponent']

    # Prepare the final DataFrame with defensive stats
    final_df = merged_team_df[['gameid', 'team', 'opponent_team', 'PTS_Against']].copy()
    final_df.sort_values(by=['team', 'gameid'], inplace=True)

    # Calculate rolling averages for points given up
    final_df['Avg_PTS_Given_Up_Last_5'] = final_df.groupby('team')['PTS_Against'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    final_df['Avg_PTS_Given_Up_Last_20'] = final_df.groupby('team')['PTS_Against'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)
    print("Final DF columns after rolling calc:", final_df.columns)  # Check columns

    # Load player data
  
    player_df['opponent_team'] = player_df.apply(lambda row: row['away'] if row['team'] == row['home'] else row['home'], axis=1)

    # Merge player data with team defensive stats
    merged_df = pd.merge(player_df, final_df, how='left', on=['gameid', 'opponent_team'])
    print("Merged DF columns:", merged_df.columns)  # Final check before select

    # Selecting necessary columns
    try:
        columns_to_keep = ['gameid', 'date', 'playerid', 'player', 'team', 'PTS', 'MIN', 'Avg_PTS_Given_Up_Last_5', 'Avg_PTS_Given_Up_Last_20']
        columns_to_keep.extend(player_df.columns.difference(['opponent_team', 'home', 'away', 'Avg_PTS_Given_Up_Last_5', 'Avg_PTS_Given_Up_Last_20']).tolist())
        merged_df = merged_df[columns_to_keep]
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Available columns:", merged_df.columns)  # Help diagnose what's missing

    

# Assuming file paths are correctly specified

    
    combined_df = merged_df
    # Clean and prepare final dataset
    combined_df.dropna(inplace=True)
    combined_df.drop_duplicates(inplace=True)

    # Ensure date column is correct and calculate player-specific rolling averages
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df.sort_values(by=['player', 'date'], inplace=True)
    combined_df['Avg_Last_5_Games'] = combined_df.groupby('player')['PTS'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    combined_df['Avg_Last_20_Games'] = combined_df.groupby('player')['PTS'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)

   



# Step 2: Calculate rolling average for last 5 games and 20 games

    df = combined_df.copy()





    
    
    # Drop rows with negative values
    df.drop('gameid', axis=1, inplace=True)
    df.drop('win', axis=1, inplace=True)
    df.drop('MIN', axis=1, inplace=True)
    df = df[df['season'] >= 2012]

    

    df.drop('FGM', axis=1, inplace=True)
    df.drop('FGA', axis=1, inplace=True)
    df.drop('FG%', axis=1, inplace=True)
    df.drop('3PM', axis=1, inplace=True)
    df.drop('3PA', axis=1, inplace=True)
    df.drop('FTM', axis=1, inplace=True)
    df.drop('FTA', axis=1, inplace=True)
    df.drop('FT%', axis=1, inplace=True)
    df.drop('OREB', axis=1, inplace=True)
    df.drop('DREB', axis=1, inplace=True)
    df.drop('PTS_Against', axis=1, inplace=True)
    df.drop('3P%', axis=1, inplace=True)
    df.drop('REB', axis=1, inplace=True)
    df.drop('AST', axis=1, inplace=True)
    df.drop('STL', axis=1, inplace=True)
    df.drop('BLK', axis=1, inplace=True)
    df.drop('TOV', axis=1, inplace=True)
    df.drop('PF', axis=1, inplace=True)
    df.drop('player', axis=1, inplace=True)
    df.drop('+/-', axis=1, inplace=True)
    df.drop('opponent_team', axis=1, inplace=True)
    df.drop('team_y', axis=1, inplace=True)
    
    df.drop('away', axis=1, inplace=True)


    df = df.rename(columns={'team_x': 'team'})

    df['athome'] = np.where(df['team'] == df['home'], 1, 0)

    df.drop('home', axis=1, inplace=True)
    df.drop('team', axis=1, inplace=True)
    df.drop('playerid', axis=1, inplace=True)

    if 'type' not in df.columns:
        raise KeyError("Column 'type' does not exist in the DataFrame")

# Initialize the LabelEncoder
    encoder = LabelEncoder()

# Fit and transform the 'type' column to label encode it
    df['type_encoded'] = encoder.fit_transform(df['type'])

# Optionally, you can drop the original 'type' column if it's no longer needed
    df.drop(columns=['type'], inplace=True)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    
    df['Month'] = df['date'].dt.month
    

    
    df = df.drop(['date', 'date'], axis=1)
    df = _reorder_columns(df, target='PTS')
    

    return df
    
def _reorder_columns(df, target='PTS'):
    """
    Re-order the columns so the target is the last one
    """
    df = df[[col for col in df.columns if col != target] + [target]]
    return df





    


def create_split(df):
    """
    Create the train-test split. The method should be 
    randomized so each call will likely yield different 
    results.
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    train_df : pandas.DataFrame
        return the training dataset as a pandas dataframe
    test_df : pandas.DataFrame
        return the test dataset as a pandas dataframe.
    """

    train_df, test_df = train_test_split(df, test_size=0.2)
    return train_df, test_df


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="filename of training data")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = pre(df)
    train_df, test_df = create_split(df)
    # solution here

    
    nonext = os.path.splitext(args.input)[0]
    print("Training DF Shape:", train_df.shape)
    train_df.to_csv(nonext+"_train.csv", index=False)
    print("Test DF Shape:", test_df.shape)
    test_df.to_csv(nonext+"_test.csv", index=False)


if __name__ == "__main__":
    main()
