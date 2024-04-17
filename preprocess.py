import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def pre(df):
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
    # Drop rows with missing values
    df = df.dropna()
    # Drop duplicate rows
    df = df.drop_duplicates()
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



    if 'type' not in df.columns:
        raise KeyError("Column 'STOCK' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['type']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['type'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    if 'away' not in df.columns:
        raise KeyError("Column 'STOCK' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['away']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['away'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    if 'team' not in df.columns:
        raise KeyError("Column 'STOCK' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['team']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['team'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)


    if 'home' not in df.columns:
        raise KeyError("Column 'STOCK' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['home']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['home'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    
    df['Month'] = df['date'].dt.month
    df['DayOfWeek'] = df['date'].dt.dayofweek

    
    df = df.drop(['date', 'date'], axis=1)

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
