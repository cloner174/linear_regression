#                             #      #                  In the name of God #  #
#
#https://github.com/cloner174
#cloner174.org@gmail.com
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Improved version to pass file path directly
def read_data(path):
    return pd.read_csv(path)

def preprocess_data(data, 
                    name_of_columns_to_be_dropped=None, 
                    target_column_name='mpg',
                    categorical_columns='origin',
                    missing_values='horsepower'):
    if name_of_columns_to_be_dropped is None:
        name_of_columns_to_be_dropped = ['mpg', 'name']
    
    X = data.drop(name_of_columns_to_be_dropped, axis=1)
    y = data[target_column_name]
    
    if categorical_columns is not None:
        X = pd.get_dummies(X, columns=[categorical_columns] if isinstance(categorical_columns, str) else categorical_columns)
    
    if missing_values is not None:
        X[missing_values].fillna(X[missing_values].mean(), inplace=True)
    
    return X, y

def train_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_test, y_pred

def plot_results(y_test, y_pred, title, target_column_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel(f'Actual {target_column_name}')
    plt.ylabel(f'Predicted {target_column_name}')
    plt.title(title)
    plt.legend()
    plt.show()

def main(path):
    data = read_data(path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    linear_model = LinearRegression()
    mse, r2, y_test_lr, y_pred_lr = train_evaluate_model(X_train, X_test, y_train, y_test, linear_model)
    print('Linear Regression MSE:', mse)
    print('Linear Regression R-squared:', r2)
    plot_results(y_test_lr, y_pred_lr, 'Actual vs. Predicted MPG (Linear Regression)', 'mpg')
    
    degrees = [1, 2, 3, 4, 5, 6]
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.transform(X_test)
        
        poly_model = LinearRegression()
        mse_poly, r2_poly, y_test_poly, y_pred_poly = train_evaluate_model(X_poly_train, X_poly_test, y_train, y_test, poly_model)
        
        print(f'Polynomial Regression (Degree {degree}) MSE:', mse_poly)
        print(f'Polynomial Regression (Degree {degree}) R-squared:', r2_poly)
        
        plot_results(y_test_poly, y_pred_poly, f'Actual vs. Predicted MPG (Polynomial Regression - Degree {degree})', 'mpg')

#end#
