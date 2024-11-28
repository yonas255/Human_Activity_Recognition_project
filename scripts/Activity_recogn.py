import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Preprocessing import preprocess



print("preprocessing the data")
X_train_scaled, X_test_scaled, y_train, y_test = preprocess()

print("Logistic Regression")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train.values.ravel())

y_pred_log = log_reg.predict(X_test_scaled)

print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred_log))
print("classification_report :\n", classification_report(y_test, y_pred_log))

#calculate sensitivity and specificity for logistic regression
cm=confusion_matrix(y_test, y_pred_log) 

#sensitivity and specificity for each class
sensitivity_per_class = cm.diagonal()/cm.sum(axis=1) #tp/(tp+fn)
specificity_per_class = cm.sum()-(cm.sum(axis=0)+cm.sum(axis=1)-cm.diagonal()) /(cm.sum()-cm.sum(axis=1)) #tn/(tn+fp)

#printing the sensitivity and specificity for each class
for i, (sensitivity, specificity) in enumerate(zip(sensitivity_per_class, specificity_per_class), 1):
    print(f"Class {i} Sensitivity: {sensitivity}, Specificity: {specificity}")

#saving the confusion matrix plot to a png file
ConfusionMatrixDisplay.from_estimator(log_reg, X_test_scaled, y_test)
plt.title ("Logistic Regression Confusion Matrix")
plt.savefig("results/confusion_matrix_logreg.png")
plt.show()

print("KNN Model")
Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train_scaled, y_train.values.ravel())

y_pred_Knn = Knn.predict(X_test_scaled)

print("KNN Accuracy: ", accuracy_score(y_test, y_pred_Knn))
print("classification_report :\n", classification_report(y_test, y_pred_Knn))

#calculate sensitivity and specificity for KNN
cm=confusion_matrix(y_test, y_pred_Knn)

#sensitivity and specificity for each class
sensitivity_per_class = cm.diagonal()/cm.sum(axis=1) #tp/(tp+fn)
specificity_per_class = cm.sum()-(cm.sum(axis=0)+cm.sum(axis=1)-cm.diagonal()) /(cm.sum()-cm.sum(axis=1)) #tn/(tn+fp)

#printing the sensitivity and specificity for each class
for i, (sensitivity, specificity) in enumerate(zip(sensitivity_per_class, specificity_per_class), 1):
    print(f"Class {i} Sensitivity: {sensitivity}, Specificity: {specificity}")
    

#saving the confusion matrix plot to a png file
ConfusionMatrixDisplay.from_estimator(Knn, X_test_scaled, y_test)
plt.title ("KNN Confusion Matrix")
plt.savefig("results/confusion_matrix_knn.png")
plt.show()

for i in range(3, 10): #looping through different values of k for KNN
    Knn = KNeighborsClassifier(n_neighbors=i) #creating a
    Knn.fit(X_train_scaled, y_train.values.ravel()) #fitting the model
    y_pred_Knn = Knn.predict(X_test_scaled) #predicting the values
    print("KNN Accuracy for ", i, " neighbors: ", accuracy_score(y_test, y_pred_Knn)) #printing the accuracy
    print("classification_report for ", i, " neighbors:\n", classification_report(y_test, y_pred_Knn)) 
    #saving the output to a csv file
    output_logreg = pd.DataFrame({'True Label': y_test.values.ravel(), 'Predicted Label (LogisticRegression)': y_pred_log}) #creating a dataframe
    output_logreg =pd.DataFrame({'True Label': y_test.values.ravel(), 'Predicted Label (KNN)': y_pred_Knn})#creating a dataframe
    
    output_logreg.to_csv("results/output_logreg.csv", index=False) #saving the output to a csv file
    output_logreg.to_csv("results/output_knn.csv", index=False) #saving the output to a csv file
    
#Linear Regression
print("Linear Regression") 
lin_reg = LinearRegression() #creating a linear regression model
lin_reg.fit(X_train_scaled, y_train) #fitting the model
y_pred_lin = lin_reg.predict(X_test_scaled) #predicting the values
    
mse = mean_squared_error(y_test, y_pred_lin)
rmse=(mse**0.5) #taking square root of mean squared error to get the mean absolute error
rss= np.sum((y_test-y_pred_lin)**2) #residual sum of squares
r2 = r2_score(y_test, y_pred_lin) #r2 score

#outputting the metrics
print("Mean Squared Error: ", mse) #printing the mse
print("Root Mean Absolute Error: ", rmse) #printing the rmse
print("Residual Sum of Squares: ", rss)#printing the rss 
print("R2 Score: ", r2)   #printing the r2 score

##plotting the true values vs predicted values for linear regression
plt.scatter(y_test, y_pred_lin, alpha=0.5)#plotting the scatter plot
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue')#plotting the line y=x
plt.xlabel('True Values') #labeling the x-axis
plt.ylabel('Predicted Values') #labeling the y-axis
plt.title('true values vs predicted values for linear regression') #setting the title
plt.savefig("results/true_vs_predicted_linreg.png") #saving the plot to a png file
plt.show() #displaying the plot
    
#saving the linear regression output to a csv file
output_linreg = pd.DataFrame({'True Label': y_test.values.ravel(), 'Predicted Label': y_pred_lin.ravel()}) #creating a dataframe
output_linreg.to_csv("results/output_linreg.csv", index=False) #saving the output to a csv file
