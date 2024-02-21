#                                                     In the name of God.


#cloner174
#https://github.com/cloner174
#cloner174.org@gmail.com


import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



# Load data

try:
    
    data = pd.read_csv("data/Automobile_data.csv")
    print("The data file called -Automobile_data.csv- was successfully detected and load!")
    time.sleep(2)

except:
    
    print("Please insert the full path to Automobile_data.csv dataset WITHOUT quotes:")
    path = input().strip()
    data = pd.read_csv(path)
    print(" Done! ")
    time.sleep(2)



# Separate features and target variable

X = data.drop(['mpg', 'name'], axis=1)                            #Independet Variables
y = data['mpg']                                                   #Target




# Convert categorical 'origin' column to dummy variables

X = pd.get_dummies(X, columns=['origin'])




# Handle missing values in the 'horsepower' column

X['horsepower'] = X['horsepower'].fillna(X['horsepower'].mean())




# Initialize the scaler

#scaler = StandardScaler()




# Fit and transform the numerical features

#X[['horsepower', 'weight', 'acceleration']] = scaler.fit_transform(X[['horsepower', 'weight', 'acceleration']])




# Splitting into train and test sets
#with randomly shufle them with rate of 42 ! and a test size 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#create model function base on our needs:
#linear_regression_model

def linear_regression_model(X_train, y_train, X_test, y_test):
    
    linear_regression = LinearRegression()
    
    linear_regression.fit(X_train, y_train)
    
    y_pred_linear = linear_regression.predict(X_test)
    
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    
    r2_linear = r2_score(y_test, y_pred_linear)
    
    print('Linear Regression - MSE:', mse_linear, '- R2 Score:', r2_linear)
    
    return y_pred_linear, linear_regression.coef_





# polynomial_regression_model

def polynomial_regression_model(X_train, y_train, X_test, y_test):
    
    polynomial_features = PolynomialFeatures()
    
    X_poly_train = polynomial_features.fit_transform(X_train)
    
    X_poly_test = polynomial_features.transform(X_test)
    
    non_linear_regression = LinearRegression()
    
    non_linear_regression.fit(X_poly_train, y_train)
    
    y_pred_non_linear = non_linear_regression.predict(X_poly_test)
    
    mse_non_linear = mean_squared_error(y_test, y_pred_non_linear)
    
    r2_non_linear = r2_score(y_test, y_pred_non_linear)
    
    print('Polynomial Regression - MSE:', mse_non_linear, '- R2 Score:', r2_non_linear)
    
    return y_pred_non_linear





#main.Run -->> All and Plots

def run_models(X_train, y_train, X_test, y_test, title):
    
    y_pred_linear, coef_linear = linear_regression_model(X_train, y_train, X_test, y_test)
    
    plt.scatter(X_test['weight'], y_test, color='red', label='Actual')
    
    plt.plot(X_test['weight'], y_pred_linear, color='blue', label='Linear Regression')
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title(title + ' - Linear Regression')
    plt.legend()
    plt.show()
    
    residuals_linear = y_test - y_pred_linear
    
    plt.scatter(y_pred_linear, residuals_linear)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted MPG')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for Linear Regression')
    plt.show()
    
    feature_names = X_train.columns
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coef_linear)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Coefficients in Linear Regression')
    plt.show()
    
    y_pred_non_linear = polynomial_regression_model(X_train, y_train, X_test, y_test)
    plt.scatter(X_test['weight'], y_test, color='red', label='Actual')
    plt.scatter(X_test['weight'], y_pred_non_linear, color='green', label='Polynomial Regression')
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title(title + ' - Polynomial Regression')
    plt.legend()
    plt.show()





# Run model
run_models(X_train, y_train, X_test, y_test, 'AutomobileMPG')







#End__________________

#cloner174
#https://github.com/cloner174
#cloner174.org@gmail.com