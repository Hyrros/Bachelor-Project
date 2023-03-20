import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

class LinearRegressionTools:

    def __init__(self, X, y, confidence_level=0.95):
        self.X = X  # Feature matrix
        self.y = y  # Target variable
        self.confidence_level = confidence_level  # Confidence level for the confidence ellipsoid
        self.regression = LinearRegression()  # Initialize the linear regression model
        self.fit()

    def fit(self):
        # Fit the linear regression model using the input data
        self.regression.fit(self.X, self.y)
        # Compute the confidence ellipsoid
        self.compute_confidence_ellipsoid()

    def compute_confidence_ellipsoid(self):
        # Compute the predicted y values
        y_pred = self.regression.predict(self.X)
        # Number of data points
        n = len(self.y)
        # Number of parameters in the model (features + intercept term)
        p = self.X.shape[1] + 1
        # Compute the residual sum of squares
        residual_sum_of_squares = np.sum((self.y - y_pred) ** 2)
        # Compute the estimate of the variance of the error term
        sigma_hat_squared = residual_sum_of_squares / (n - p)

        # Add an intercept term to the feature matrix
        X_with_intercept = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        # Compute the inverse of the (X^T * X) matrix
        XTX_inv = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
        # Compute the covariance matrix of the parameter estimates
        covariance_matrix = sigma_hat_squared * XTX_inv

        # Compute the scaling factor for the confidence ellipsoid
        scaling_factor = chi2.ppf(self.confidence_level, p)
        # Compute the confidence ellipsoid
        self.confidence_ellipsoid = scaling_factor * covariance_matrix

    def plot_data_and_regression_line(self):
        # Check if the data is 2D (only one feature)
        if self.X.shape[1] != 1:
            print("Plotting is supported only for the 2D case.")
            return

        # Plot the data points
        plt.scatter(self.X, self.y, label='Data')
        plt.xlabel('Feature')
        plt.ylabel('Target')

        # Create a set of x values for the regression line
        x_line = np.linspace(np.min(self.X), np.max(self.X), 100)
        # Compute the corresponding y values for the regression line
        y_line = self.regression.predict(x_line[:, np.newaxis])
        # Plot the regression line
        plt.plot(x_line, y_line, color='r', label='Regression Line')

        plt.legend()
        plt.show()

# Example dataset: y values depend on a single feature x
x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_values = np.array([60, 65, 70, 73, 76, 82, 85, 87, 89, 92])

# Initialize the LinearRegressionTools class and plot the data and regression line
lrt = LinearRegressionTools(x_values[:, np.newaxis], y_values)
lrt.plot_data_and_regression_line()
