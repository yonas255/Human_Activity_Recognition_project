import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(): #function to load the data
    X_train = pd.read_csv("data/X_train.txt", sep= '\\s+', header=None) #reading the training data
    X_test = pd.read_csv("data/X_test.txt", sep= '\\s+', header=None)#reading the test data
    y_train = pd.read_csv("data/y_train.txt", sep= '\\s+', header=None)#reading the training labels
    y_test = pd.read_csv("data/y_test.txt", sep='\\s+', header=None)#reading the test labels
    return X_train, X_test, y_train, y_test #returning the data

#function to scale the data
def scale_data(X_train, X_test):
    scaler = StandardScaler()#creating an instance of the StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)#fitting and transforming the training data
    X_test_scaled = scaler.transform(X_test)#transforming the test data
    return X_train_scaled, X_test_scaled #returning the scaled data

#function to preprocess the data
def preprocess():
    X_train, X_test, y_train, y_test = load_data()#loading the data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)#scaling the data
    return X_train_scaled, X_test_scaled, y_train, y_test #returning the preprocessed data
