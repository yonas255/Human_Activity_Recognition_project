import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    X_train = pd.read_csv("data/X_train.txt", sep= '\\s+', header=None)
    X_test = pd.read_csv("data/X_test.txt", sep= '\\s+', header=None)
    y_train = pd.read_csv("data/y_train.txt", sep= '\\s+', header=None)
    y_test = pd.read_csv("data/y_test.txt", sep='\\s+', header=None)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def preprocess():
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
