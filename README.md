# Bike Sharing Demand Prediction

A custom-built predictive engine that forecasts hourly bike rental demand. This project implements **Linear Regression** and **Polynomial/Interaction Models** from scratch using the **Normal Equation**, avoiding standard "black-box" ML libraries to ensure mathematical transparency.

## Project Overview
The goal of this project is to predict the total count of bikes rented during a specific hour using the Kaggle Bike Sharing Demand dataset. We compared multiple regression strategies, proving that a **Quadratic Model with Interaction Terms** significantly outperforms high-degree polynomial models by capturing synergistic relationships (e.g., Temperature × Humidity).

**Key Achievement:**
- **Best Model:** Quadratic Interaction Model ($R^2 \approx 0.72$)
- **Baseline:** Linear Regression ($R^2 \approx 0.51$)
- **Optimization Method:** Normal Equation (Analytical Solution)

## Features & Methodology
- **Mathematical Optimization:** Implemented the Normal Equation $\theta = (X^T X)^{-1} X^T y$ using pure NumPy matrix operations.
- **Advanced Feature Engineering:**
  - **Cyclical Time Encoding:** Mapped hours (0-23) and months (1-12) to Sine/Cosine coordinates to preserve temporal continuity.
  - **Interaction Terms:** Created features like $Temp \times Humidity$ to model conditional dependencies.
- **Model Comparison:** Evaluated Linear, Polynomial ($d=2,3,4$), and Interaction models.

## Tech Stack
- **Language:** Python
- **Libraries:** NumPy (Matrix Math), Pandas (Data Handling)
- **Concepts:** Linear Algebra, Feature Engineering, Bias-Variance Tradeoff

## Results
| Model Name | MSE | $R^2$ Score |
| :--- | :--- | :--- |
| **Interaction (d=2)** | **8869.83** | **0.7229** |
| Poly (d=4) | 11229.39 | 0.6491 |
| Poly (d=3) | 11782.33 | 0.6318 |
| Poly (d=2) | 12938.30 | 0.5957 |
| Linear Regression | 15485.58 | 0.5161 |

*The Interaction model achieved the highest accuracy by balancing complexity (low bias) without overfitting (low variance).*

## How to Run

1. **Prerequisites**
   Ensure you have Python installed with the required libraries:
   ```bash
   pip install pandas numpy
2. Dataset Place the train.csv file in the same directory as the script.
3. Execute the Script
   ```bash
   python model.py
  The script will load the data, perform feature engineering, train all 5 models, and print the performance comparison table
