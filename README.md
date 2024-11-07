# Breast Cancer Prediction with Multiple Machine Learning Models

This repository contains code for classifying breast cancer data using various machine learning models and evaluating their performance. The dataset used is the well-known **Breast Cancer Wisconsin dataset** available from `sklearn.datasets`. The models tested include Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, Neural Network (MLP), AdaBoost, and XGBoost.

## Requirements

The following Python libraries are required to run the code:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## Dataset

The dataset used is the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`. It includes features extracted from breast cancer biopsies, such as texture, radius, smoothness, and area, among others. The target variable indicates whether the sample is malignant (1) or benign (0).

## Code Explanation

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using `load_breast_cancer()`.
   - The data is split into features `X` and target `y`.
   - Feature scaling is applied using `StandardScaler` to normalize the data.

2. **Model Training:**
   - Several machine learning models are trained on the data, including:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Naive Bayes
     - Neural Network (MLP Classifier)
     - AdaBoost
     - XGBoost

3. **Evaluation:**
   - Each model is evaluated using metrics such as **Confusion Matrix**, **Classification Report**, **AUC Score**, and **ROC Curve**.
   - Additionally, Precision-Recall curves are plotted to compare the models.

4. **Visualization:**
   - The ROC Curves for each model are plotted to evaluate the trade-off between the true positive rate and false positive rate.
   - Precision-Recall curves are plotted to visualize the performance of the models, especially for imbalanced datasets.

## Functions

### `evaluate(y_true, y_pred, y_prob, model_name)`
This function evaluates the performance of a model using the following metrics:
- **Confusion Matrix**
- **Classification Report**
- **AUC Score** (if probabilities are provided)

### `plot_roc(model, X_test, y_test, label)`
This function plots the ROC curve for a given model.

### `plot_precision_recall_curve(model, X_test, y_test, model_name)`
This function plots the Precision-Recall curve for a given model.

## Usage

After setting up your environment and installing the required libraries, you can run the code by simply executing the Python script. The evaluation results and plots will be displayed.

```bash
python breast_cancer_model.py
```

## Example Output

The following output is displayed for each model:

- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Shows precision, recall, f1-score, and accuracy.
- **AUC Score**: The Area Under the ROC Curve score.
- **ROC Curve**: A plot comparing the true positive rate (recall) versus the false positive rate.
- **Precision-Recall Curve**: A plot of precision vs recall.

## Models and Performance

- **Logistic Regression**: Achieves a high accuracy of 98% with a very high AUC score of 0.998.
- **K-Nearest Neighbors (KNN)**: Achieves 96% accuracy and performs well in terms of precision and recall.
- **Support Vector Machine (SVM)**: AUC score of 0.996, with strong classification performance.
- **Random Forest**: Achieves a high accuracy of 96% and AUC score of 0.996.
- **Gradient Boosting**: Performs similarly to Random Forest with an AUC score of 0.996.
- **Naive Bayes**: Slightly lower performance compared to other models but still achieves 94% accuracy and strong recall.
- **Neural Network (MLP)**: Excellent performance with 98% accuracy.
- **AdaBoost**: Strong performance similar to other ensemble models, achieving 98% accuracy and a high AUC score of 0.996.
- **XGBoost**: One of the top performers with a high AUC score of 0.994 and excellent accuracy of 97%.

## Conclusion

This project demonstrates the application of multiple machine learning models for breast cancer classification, with strong performance across various models. The code provides a basis for comparing different models using various evaluation metrics, helping you select the best approach for cancer prediction tasks.
