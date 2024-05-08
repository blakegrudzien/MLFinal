import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("intrain", help="filename of training data")
parser.add_argument("intest", help="filename of testing data")
args = parser.parse_args()
xTrain = pd.read_csv(args.intrain)
xTest = pd.read_csv(args.intest)

yTrain = xTrain.iloc[:,-1:]
yTest = xTest.iloc[:,-1:]
xTrain = xTrain.iloc[:,:-1]
xTest = xTest.iloc[:,:-1]

# Initialize and train the MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(71,1), max_iter=1000, learning_rate='adaptive')
mlp.fit(xTrain, yTrain.values.ravel())

yPred = mlp.predict(xTest)
mse = mean_squared_error(yTest, yPred)
r2 = r2_score(yTest,yPred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')