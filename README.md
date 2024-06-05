# Capstone

Repository of my capstone project with Loyola University Maryland: 
# Differentiating LLM and Student Authorship in Long-Form Text

Classification models to distinguish LLM-Generated text from student-written essays. Feature importance analysis performed with SHAP explanations to provide interpretable insights on LLM detection. 

## Dataset
A subset of 25,059 essays were used for this project from two joined datasets:
* DAIGT-V4-Train-Dataset by Darek Kleczek (https://www.kaggle.com/datasets/thedrcat/daigt-v4-train-dataset/data)
* AI-Detection-Student-Writing by Scott Crosseye (https://github.com/scrosseye/AI-Detection-Student-Writing)

Data was filtered down by manual review, duplicate removal, outlier removal, and null value removal. 

## Feature extraction pipeline:
* Longformer Embeddings - CLS Pooling
* Bigram and Trigram Term Frequency - Inverse Document Frequency (TF-IDF)
* Linguistic Inquiry and Word Count (LIWC-22)* 

*LIWC-22 extracted via desktop application

Significant variables from unpaired t-tests with Holm's correction were used for modeling. 

## Classification Models

Support Vector Machine (SVM), Gaussian Naïve Bayes, Multi-Layer Perceptron, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), XGBoost, and Multiple Logistic Regression models were fit to predictor groups in three series:
* Embeddings
* Embeddings, TF-IDF, and LIWC-22
* TF-IDF and LIWC-22

Model performance was compared by accuracy, precision, recall, F1-score, and ROC-AUC. The final model chosen was the best model from the third series, which performed approximately as well as the other two predictor groups.

### Model Performance

|       |           |           |            |          |
|-------|-----------|-----------|------------|----------|
| Group | Accuracy  | Precision | Recall     | F1 Score |
| 1     | 0.929336  | 0.943304  | 0.885417   | 0.912827 |
| 2     | 0.897030  | 0.906775  |  0.838221  | 0.870157 |
| 3     | 0.878292  | 0.881288  | 0.819351   | 0.847633 |

## SHAP Explanations

SHAP explanations were generated for the final model to interpret feature importance of the model as a whole, and decisions for individual predictions. An interesting takeaway in model predictions was the difference in lengthy words between students and LLMs. More specifically, the percentage of words longer than six letters was higher in LLMs. 

![Alt text](Picture1.png)


Example of implementation can be found in demo.ipynb.

## Collaborators
A special thank you to Dr. Catherine Pollack and Dr. Catherine Schwartz from the Johns Hopkins University Applied Physics Laboratory for their mentorship and collaboration on this project!
