import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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

ConfusionMatrixDisplay.from_estimator(Knn, X_test_scaled, y_test)
plt.title ("KNN Confusion Matrix")
plt.savefig("results/confusion_matrix_knn.png")
plt.show()

for i in range(3, 10):
    Knn = KNeighborsClassifier(n_neighbors=i)
    Knn.fit(X_train_scaled, y_train.values.ravel())
    y_pred_Knn = Knn.predict(X_test_scaled)
    print("KNN Accuracy for ", i, " neighbors: ", accuracy_score(y_test, y_pred_Knn))
    print("classification_report for ", i, " neighbors:\n", classification_report(y_test, y_pred_Knn))
    
    output_logreg = pd.DataFrame({'True Label': y_test.values.ravel(), 'Predicted Label (LogisticRegression)': y_pred_log})
    output_logreg =pd.DataFrame({'True Label': y_test.values.ravel(), 'Predicted Label (KNN)': y_pred_Knn})
    
    output_logreg.to_csv("results/output_logreg.csv", index=False)
    output_logreg.to_csv("results/output_knn.csv", index=False)