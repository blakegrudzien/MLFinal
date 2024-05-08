import argparse
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, Ridge
import numpy as np
import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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

 
    
    

  
    merged_team_df = pd.merge(team_df, team_df, on='gameid', suffixes=('', '_opponent'))
    

    merged_team_df = merged_team_df[merged_team_df['team'] != merged_team_df['team_opponent']]

    merged_team_df['PTS_Against'] = merged_team_df['PTS_opponent']
    merged_team_df['opponent_team'] = merged_team_df['team_opponent']

    final_df = merged_team_df[['gameid', 'team', 'opponent_team', 'PTS_Against']].copy()
    final_df.sort_values(by=['team', 'gameid'], inplace=True)

    final_df['Avg_PTS_Given_Up_Last_5'] = final_df.groupby('team')['PTS_Against'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    final_df['Avg_PTS_Given_Up_Last_20'] = final_df.groupby('team')['PTS_Against'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)
    


  
    player_df['opponent_team'] = player_df.apply(lambda row: row['away'] if row['team'] == row['home'] else row['home'], axis=1)

    merged_df = pd.merge(player_df, final_df, how='left', on=['gameid', 'opponent_team'])

    try:
        columns_to_keep = ['gameid', 'date', 'playerid', 'player', 'team', 'PTS', 'MIN', 'Avg_PTS_Given_Up_Last_5', 'Avg_PTS_Given_Up_Last_20']
        columns_to_keep.extend(player_df.columns.difference(['opponent_team', 'home', 'away', 'Avg_PTS_Given_Up_Last_5', 'Avg_PTS_Given_Up_Last_20']).tolist())
        merged_df = merged_df[columns_to_keep]
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Available columns:", merged_df.columns) 


    combined_df = merged_df

    combined_df.dropna(inplace=True)
    combined_df.drop_duplicates(inplace=True)

    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df.sort_values(by=['player', 'date'], inplace=True)
    combined_df['Avg_Last_5_Games'] = combined_df.groupby('player')['PTS'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    combined_df['Avg_Last_20_Games'] = combined_df.groupby('player')['PTS'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)


    df = combined_df.copy()

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
    encoder = LabelEncoder()
    df['type_encoded'] = encoder.fit_transform(df['type'])
    df.drop(columns=['type'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  
    df['Month'] = df['date'].dt.month 
    df = df.drop(['date', 'date'], axis=1)
    df = _reorder_columns(df, target='PTS')
    return df
    
def _reorder_columns(df, target='PTS'):
    df = df[[col for col in df.columns if col != target] + [target]]
    return df

#Random Forest
def train_stacked_model(X_train, y_train):

    base_estimators = [
        ('ridge', Ridge(random_state=42)),
        ('random_forest', RandomForestRegressor(n_estimators=50, min_samples_leaf=4, random_state=42, n_jobs=-1))
    ]


    stack_reg = StackingRegressor(estimators=base_estimators)


    param_grid = {
        'ridge__alpha': [0.001, 0.01, 0.1, 1.0],  
        'random_forest__n_estimators': [10, 50, 100, 200], 
        'random_forest__max_features': ['sqrt', 'log2', None],  
        'random_forest__max_depth': [None, 10, 20, 30]  
    }

    grid_search = RandomizedSearchCV(stack_reg, param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_stack_reg = grid_search.best_estimator_

    print("Best Parameters Found:")
    for param_name in grid_search.best_params_:
        print(f"{param_name}: {grid_search.best_params_[param_name]}")

    return best_stack_reg







def train_MLR(X_train, y_train):

    mlr = LinearRegression()


    param_grid = {
        'fit_intercept': [True, False],
        'n_jobs': [None, -1]
    }


    grid_search = GridSearchCV(mlr, param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)


    for i, _ in enumerate(grid_search.cv_results_['mean_test_score'], 1):
        print(f"Configuration {i} Score: {_}")


    best_mlr = grid_search.best_estimator_


    print("Best Parameters Found:")
    print(grid_search.best_params_)

    return best_mlr




def train_KNN(X_train, y_train):
   
    knn = KNeighborsRegressor()


    param_grid = {
        'n_neighbors': [3, 5, 10],  
        'weights': ['uniform', 'distance'],  
        'p': [1, 2]  
    }

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)


    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print(f"An error occurred during model fitting: {e}")
        return None

    best_knn = grid_search.best_estimator_

    print("Best Parameters Found:")
    for param_name in grid_search.best_params_:
        print(f"{param_name}: {grid_search.best_params_[param_name]}")

    return best_knn




def train_ridge(X_train, y_train):

    ridge = Ridge()


    param_grid = {
        'alpha': np.logspace(-6, 6, 13),  
        'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg'],  
        'fit_intercept': [True, False]  
    }


    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)


    grid_search.fit(X_train, y_train)


    print("Best parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    return best_model




def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2




def create_split(X,y):
 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    
    
    parser = argparse.ArgumentParser(description="Run a stacked ensemble with Ridge regression and RandomForestRegressor.")
    parser.add_argument("input", help="filename of the dataset for training")
    args = parser.parse_args()

    # Load and preprocess the data
    df = pd.read_csv(args.input)
    cleaned_df = pre(df)

    # Prepare the data
    X = cleaned_df.drop('PTS', axis=1)
    y = cleaned_df['PTS']

    # Create train-test split
    X_train, X_test, y_train, y_test = create_split(X, y)

    # Train stacked model
    #trained_model = train_stacked_model(X_train, y_train)
    trained_model = train_KNN(X_train, y_train)
    #trained_model = train_MLR(X_train, y_train)
    #trained_model = train_ridge(X_train, y_train)
  



    # Evaluate the model
    mse, r2 = evaluate_model(trained_model, X_test, y_test)

    print("MSE:", mse)
    print("R-squared:", r2)

if __name__ == "__main__":
    main()
