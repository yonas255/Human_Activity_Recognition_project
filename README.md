# humann Activity recognition using smartphones

## Objectivies

this project focus to classify human activities eg. walking and setting

## setpes

1. Data Loading and Preprocessing:
   - Data was loaded and normalized using StandardScaler.
   - Features were prepared for model training.
2. Model Training:
   - Logistic Regression and K-NN models were trained on the preprocessed data.
3. Evaluation:
   - Models were evaluated using accuracy, classification reports, and confusion matrices.

## Results

| Model                 | Accuracy | Weighted F1-Score |
|-----------------------|----------|--------------------|
| Logistic Regression   | 95.49%   | 0.95               |
| K-Nearest Neighbors   | 88.87%   | 0.89               |

Logistic Regression outperformed K-NN, demonstrating better performance across all metrics.

## Outputs

- **Confusion Matrices**:
  - Logistic Regression: `results/ConfusionMatrix_LogReg.png`.
  - K-NN: `results/ConfusionMatrix_KNN.png`.

- **Predictions**:
  - Logistic Regression predictions: `results/output_logreg.csv`.
  - K-NN predictions: `results/output_knn.csv`.
  
## instructions to run

  1. install all dependencies ``bash pip install pandas numpy scikit-learn matplotlib
  2. pip install matplotlib
  3. pip install scikit-learn
  4. pip install pandas
  5. venv\Scripts\activate  # For Windows source venv/bin/activate  # For macOS/Linux
  6. run Activity_recogn.py and see the results
   