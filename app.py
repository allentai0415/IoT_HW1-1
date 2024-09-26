from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    # 渲染主頁面，包含滑動條的表單
    return render_template('index.html')

@app.route('/generate-regression', methods=['POST'])
def generate_regression():
    # Step 1: 接收滑動條提交的值
    a = float(request.json.get('a'))
    b = 50  # 固定 b = 50
    c = float(request.json.get('c'))
    n = int(request.json.get('n'))
    variance = 1.0  # variance for the normal distribution

    # Step 2: Generate n random points for x and y
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-10, 10, n)

    # Generate y values based on the equation y = a*x + b + c*N(0, variance)
    noise = np.random.normal(0, variance, n)
    y = a * x + b + c * noise

    # Step 3: Perform Linear Regression
    x_reshaped = x.reshape(-1, 1)  # Reshape for sklearn
    model = LinearRegression()
    model.fit(x_reshaped, y)

    # Step 4: Get the line of best fit (predicted y values)
    y_pred = model.predict(x_reshaped)

    # Step 5: Plot the points and the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data points', color='blue')
    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.title(f"Linear Regression: y = {a:.2f}*x + {b} + {c:.2f}*N(0, {variance})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Save the plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Encode the image to base64 string
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the image as a base64 encoded string
    return jsonify({'image': image_base64})

if __name__ == '__main__':
    # 確保上一次的圖像被刪除
    if os.path.exists("regression_plot.png"):
        os.remove("regression_plot.png")
    
    app.run(debug=True)