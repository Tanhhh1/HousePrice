from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso

app = Flask(__name__)

# Load data
data = pd.read_csv('house_data.csv')

# Define features and target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'yr_built', 'view', 'condition', 'grade', 'zipcode', 'lat', 'long']
target = 'price'

# Clean data
data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42, learning_rate_init=0.01)
mlp.fit(X_scaled, y)

linear_reg = LinearRegression()
linear_reg.fit(X_scaled, y)

lasso_reg = Lasso(alpha=0.1, max_iter=5000)
lasso_reg.fit(X_scaled, y)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-price', methods=['POST'])
def predict_price():
    try:
        # Collect all the form data
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        floors = float(request.form['floors'])
        waterfront = int(request.form['waterfront'])
        yr_built = int(request.form['yr_built'])
        view = int(request.form['view'])
        condition = int(request.form['condition'])
        grade = int(request.form['grade'])
        zipcode = int(request.form['zipcode'])
        lat = float(request.form['lat'])
        long = float(request.form['long'])
        method = request.form['method']

        # Prepare data using all features
        new_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'floors': [floors],
            'waterfront': [waterfront],
            'yr_built': [yr_built],
            'view': [view],
            'condition': [condition],
            'grade': [grade],
            'zipcode': [zipcode],
            'lat': [lat],
            'long': [long],
        })

        new_data_scaled = scaler.transform(new_data)

        # Predict based on chosen method
        if method == 'linear_regression':
            predicted_price = linear_reg.predict(new_data_scaled)[0]
        elif method == 'lasso_regression':
            predicted_price = lasso_reg.predict(new_data_scaled)[0]
        elif method == 'neural_network':
            predicted_price = mlp.predict(new_data_scaled)[0]
        else:
            return "Phương pháp không hợp lệ", 400

        return render_template('index.html', predicted_price=f"${predicted_price:,.2f}", selected_method=method)

    except Exception as e:
        return f"Đã có lỗi xảy ra: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
