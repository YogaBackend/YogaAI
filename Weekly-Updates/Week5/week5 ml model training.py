import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the Excel data
excel_data = pd.read_excel('location of the file')

# Replace the column numbers with the actual numeric indices you want to include as features
feature_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

# Prepare the data
X = excel_data.iloc[:, feature_indices]
y = excel_data['Motion']  # Replace 'target_column_name' with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (for regression, you can use metrics like mean squared error)
y_pred = model.predict(X_test)  # Predict on the test data
mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
print(f'Mean Squared Error: {mse}')

# Specify the directory where you want to save the trained model
save_directory = 'C:/Users/Pravin/Desktop/heatmap generation/'

# Save the trained model to a .pkl file in the specified directory
with open(save_directory + 'bone_data_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Regression model saved as 'bone_data_regression_model.pkl'")
