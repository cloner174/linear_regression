#                                                     In the name of God.


#cloner174
#https://github.com/cloner174
#cloner174.org@gmail.com


import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#from sklearn.feature_selection import SelectKBest, f_regression



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
X = data.drop(['mpg', 'name'], axis=1)
y = data['mpg']


# Convert categorical 'origin' column to dummy variables
X = pd.get_dummies(X, columns=['origin'])



# Handle missing values in the 'horsepower' column
X['horsepower'] = X['horsepower'].fillna(X['horsepower'].mean())


# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical features
X[['horsepower', 'weight', 'acceleration']] = scaler.fit_transform(X[['horsepower', 'weight', 'acceleration']])



# Select top k features based on F-test
#selector = SelectKBest(score_func=f_regression, k=5)
#X_selected = selector.fit_transform(X, y)




def run_models(X_train, y_train, X_test, y_test, title):
    
    # Linear Regression
    linear_regression = LinearRegression()
    
    linear_regression.fit(X_train, y_train)
    
    y_pred_linear = linear_regression.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    print('Linear Regression - MSE:', mse_linear, '- R2 Score:', r2_linear)
    
    # Polynomial Regression
    
    polynomial_features = PolynomialFeatures()
    
    X_poly_train = polynomial_features.fit_transform(X_train)
    
    X_poly_test = polynomial_features.transform(X_test)
    
    non_linear_regression = LinearRegression()
    
    non_linear_regression.fit(X_poly_train, y_train)
    
    y_pred_non_linear = non_linear_regression.predict(X_poly_test)
    mse_non_linear = mean_squared_error(y_test, y_pred_non_linear)
    r2_non_linear = r2_score(y_test, y_pred_non_linear)
    print('Polynomial Regression - MSE:', mse_non_linear, '- R2 Score:', r2_non_linear)
    
    degrees = [1, 2, 3, 4, 5]                        # Adjustble!
    global train_scores
    train_scores, test_scores = validation_curve(
        LinearRegression(),                          # Use the trained linear model
        X_poly_train,                                # Train on polynomial features
        y_train,
        param_name='fit_intercept',                  # Adjustble!
        param_range=degrees,
        scoring='neg_mean_squared_error',            # Adjustble!
        cv=5                                         # Adjustble!
    )
    
    train_scores_mean = list(train_scores[0])
    test_scores_mean = list(test_scores[0])
    
    plt.plot(train_scores_mean, label='Training error')
    plt.plot(test_scores_mean, label='Validation error')
    plt.ylabel('Mean Squared Error')
    plt.title('Validation Curve for Polynomial Regression')
    plt.legend()
    plt.show()
    
    # Logistic Regression
    y_median = y.median()
    y_train_binary = (y_train > y_median).astype(int)
    y_test_binary = (y_test > y_median).astype(int)
    
    logistic_regression = LogisticRegression(max_iter=10000)
    
    logistic_regression.fit(X_train, y_train_binary)
    
    y_pred_logistic = logistic_regression.predict(X_test)
    accuracy_logistic = accuracy_score(y_test_binary, y_pred_logistic)
    print('Logistic Regression - Accuracy:', accuracy_logistic)
    
    # Ridge Regression
    ridge = Ridge(alpha=0.5)                        # Adjustble!
    ridge.fit(X_train, y_train)
    
    # Lasso Regression
    lasso = Lasso(alpha=0.1)                        # Adjustble!
    lasso.fit(X_train, y_train)
    
    # Plot the results
    plt.scatter(X_test['weight'], y_test, color='red', label='Actual')
    plt.plot(X_test['weight'], y_pred_linear, color='blue', label='Linear Regression')
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title(title + ' - Linear Regression')
    plt.legend()
    plt.show()
    
    plt.scatter(X_test['weight'], y_test, color='red', label='Actual')
    plt.plot(X_test['weight'], y_pred_non_linear, color='green', label='Polynomial Regression')
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title(title + ' - Polynomial Regression')
    plt.legend()
    plt.show()




# Splitting data for different evaluations

#  10% dataset
X_ten_percent, X_ten_test, y_ten_percent, y_ten_test = train_test_split(X, y, test_size=0.9, random_state=42)
print("Ten percent dataset shape:", X_ten_percent.shape, y_ten_percent.shape)

run_models(X_ten_percent, y_ten_percent, X_ten_test, y_ten_test, "10% of the data")




#  90% dataset
X_ninety_percent, X_ninety_test,  y_ninety_percent, y_ninety_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("Ninety percent dataset shape:", X_ninety_percent.shape, y_ninety_percent.shape)

run_models(X_ninety_percent, y_ninety_percent, X_ninety_test, y_ninety_percent, "90% of the data")




# the entire dataset
X_train, X_test,  y_train, y_test = train_test_split(X, y, X, y, random_state=42)
print('All of the dataset shape:', X.shape, y.shape)

run_models(X_train, y_train, X_test, y_test, 'All of the data');










#End__________________

#cloner174
#https://github.com/cloner174
#cloner174.org@gmail.com