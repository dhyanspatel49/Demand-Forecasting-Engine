import pandas
import numpy

# --- Helper Functions ---

def get_weights(X, y):
    # solving for weights using Normal Equation
    rows = X.shape[0]
    
    # add bias (column of 1s)
    bias = numpy.ones((rows, 1))
    X_bias = numpy.hstack((bias, X))

    # using pinv for stability
    w = numpy.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    return w

def predict(X, w):
    rows = X.shape[0]
    
    # match shape by adding bias
    bias = numpy.ones((rows, 1))
    X_bias = numpy.hstack((bias, X))
    
    return X_bias @ w

def get_mse(y_true, y_pred):
    # mean squared error
    return numpy.mean((y_true - y_pred) ** 2)

def get_r2(y_true, y_pred):
    # R-squared score
    res_sum = numpy.sum((y_true - y_pred) ** 2)
    tot_sum = numpy.sum((y_true - numpy.mean(y_true)) ** 2)
    return 1 - (res_sum / tot_sum)

def poly_features(X, degree):
    # expands features: x -> x, x^2, x^3...
    out = X.copy()
    for d in range(2, degree + 1):
        out = numpy.hstack((out, numpy.power(X, d)))
    return out

def interact_features(X):
    # multiplies features: x1*x2, x1*x3...
    n_samples, n_feat = X.shape
    out = X.copy()
    
    for i in range(n_feat):
        for j in range(i, n_feat):
            new_col = (X[:, i] * X[:, j]).reshape(-1, 1)
            out = numpy.hstack((out, new_col))
    return out

# --- Main Script ---

print("Loading Data...")
try:
    df = pandas.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found.")
    exit()

# feature engineering
df['datetime'] = pandas.to_datetime(df['datetime'])

# 1. Basic Time Features
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# 2. Cyclical Features (The "Correct" way to handle time)
df['hour_sin'] = numpy.sin(2 * numpy.pi * df['hour'] / 24.0)
df['hour_cos'] = numpy.cos(2 * numpy.pi * df['hour'] / 24.0)
df['month_sin'] = numpy.sin(2 * numpy.pi * df['month'] / 12.0)
df['month_cos'] = numpy.cos(2 * numpy.pi * df['month'] / 12.0)

# 3. One Hot Encoding for Categories
df = pandas.get_dummies(df, columns=['season', 'weather'], drop_first=True)

# cleanup
drop_cols = ['count', 'datetime', 'casual', 'registered', 'month']
# We keep 'hour' numeric + cyclical to help separation
cols = [c for c in df.columns if c not in drop_cols]
X_data = df[cols]
y_data = df['count']

# convert to numpy
X = X_data.values.astype(float)
y = y_data.values.astype(float)

# manual split
numpy.random.seed(123)
shuffled = numpy.random.permutation(len(X))
split_idx = int(len(X) * 0.2) # 20% test

idx_test = shuffled[:split_idx]
idx_train = shuffled[split_idx:]

X_train_raw = X[idx_train]
y_train = y[idx_train]
X_test_raw = X[idx_test]
y_test = y[idx_test]

# --- Standardization ---
train_features = X_train_raw 
avg = numpy.mean(train_features, axis=0)
std = numpy.std(train_features, axis=0)
std[std == 0] = 1 

X_train = (train_features - avg) / std
X_test = (X_test_raw - avg) / std

results = []

# 1. Linear Regression
w_lin = get_weights(X_train, y_train)
preds = predict(X_test, w_lin)
results.append(("Linear Regression", get_mse(y_test, preds), get_r2(y_test, preds)))

# 2. Polynomials (Degrees 2, 3, 4)
for d in [2, 3, 4]:
    X_train_poly = poly_features(X_train, d)
    X_test_poly = poly_features(X_test, d)

    w_poly = get_weights(X_train_poly, y_train)
    p_poly = predict(X_test_poly, w_poly)

    results.append((f"Poly (d={d})", get_mse(y_test, p_poly), get_r2(y_test, p_poly)))

# 3. Quadratic with Interactions
print("Training Quadratic Model with Interactions (this may take a moment)...\n")
X_train_inter = interact_features(X_train)
X_test_inter = interact_features(X_test)

w_inter = get_weights(X_train_inter, y_train)
p_inter = predict(X_test_inter, w_inter)
results.append(("Interaction (d=2)", get_mse(y_test, p_inter), get_r2(y_test, p_inter)))

# --- Final Output ---
print(f"{'Model Name':<25} | {'MSE':<15} | {'R2':<10}")
print("-" * 50)

for name, mse, r2 in results:
    print(f"{name:<25} | {mse:<15.2f} | {r2:<10.4f}")
