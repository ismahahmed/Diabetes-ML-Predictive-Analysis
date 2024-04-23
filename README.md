# Diabetes-ML-Predictive-Analysis

## Introduction

This repository presents an analysis of diabetes data with the aim of predicting diabetes in individuals using machine learning and data analytics principles. The analysis includes the exploration of various features, correlation analysis, and statistical summaries. 

The primary aim of this project is to assess the accuracy of predictive models trained using the dataset. By employing different machine learning algorithms, we seek to evaluate their efficacy in predicting the occurrence of diabetes based on the available features.

**Data Source**: The data I will be using for this analysis is found on [Kaggle](https://www.kaggle.com/datasets/whenamancodes/predict-diabities/data). This dataset was originally from the National Institute of Diabetes and Digestive and Kidney
Diseases. Things to note about the data: All individuals in the dataset are female, at least 21 years old and of Pima Indian Heritage.

## Project Files

- **data**
  - `diabetes.csv`: Dataset containing the project data

- **figures**
  - `confusion_matrix_modelname.png`: Confusion matrix plot, each model has a plot saved in this folder.
  - `matrix_correlation.png`: Correlation matrix plot of data features + outcome variable
  - `outcome_correlation.png`: Correlation plot showing the relationship between the Outcome variable and features
  - `decision_tree.png`: Visualization of the decision tree model

- **scripts**
  - `data_exploration.py`: Python script for exploring the dataset
  - `models.py`: Python script for training and testing machine learning models and saving correlation plots in the figures folder
  - `run.py`: Python script to execute data exploration and model training/testing scripts.

## Scope of Analysis

* Data Exploration: Examination of feature distributions, correlation analyses, and statistical summaries.
* Model Training: Training and optimization of machine learning models utilizing the dataset features.
* Model Evaluation: Evaluation of model performance metrics to determine the most accurate predictive algorithm for diabetes prediction.
* Conclusion: Summary of findings and implications for diabetes prediction and healthcare practices.

## The Dataset

The *diabetes.csv* dataset includes 768 patients information spanning the following features:

* Pregnancies - Number of times pregnant (Integer | Discrete)
* Glucose - Plasma Glucose Concentration (Integer | Continuous)
* Blood Pressure - Diastolic Blood Pressure (Integer | Continuous)
* Skin Thickness - Skin Thickness (Integer | Continuous)
* Insulin - 2-Hour Serum Insulin (Integer | Continuous)
* BMI - Body Mass Index (Float | Coninuous)
* Diabetes Pedigree Function - Diabetes Pedigree Function (Float | Continuous)
* Age - Age (years) (Integer | Continuous)
* Outcome - Class variable (0 or 1) (Integer | Discrete)

## Cleaning Data

Before proceeding with the modeling process, it was crucial to conduct preliminary data cleaning and preprocessing to ensure the dataset's quality and reliability. Initially with 769 observations, the original dataset revealed inconsistencies, specifically regarding missing data. Upon closer examination, instances were identified where essential variables such as Blood Pressure, Glucose, Insulin, or BMI were recorded as 0. To keep the data reliable and help build better models, these instances were removed from the dataset. Subsequently, after the removal of observations with missing data, the dataset was reduced to 392 observations. To evaluate model performance effectively, a 60/40 split was applied, resulting in 157 observations allocated to the testing dataset.

## Exploratory Data Analysis

This project does not delve into feature selection, however, I did examine the distributions, statistical summaries, and correlations of the features I worked with. This involved assessing the shape of different features and their correlations. I created a correlation matrix (matrix_correlation.png) to understand the relationship between features and the outcome variable. 

The plot saved as outcome_correlation.png shows a segment of the correlation plot (matrix_correlation.png) of the dataset. Here, we are looking at the correlation between the
Outcome variable and the different features in the dataset. We can see that Glucose has the highest linear correlation with the Outcome variable, which is not surprising. Individuals with diabetes often have higher levels of glucose in their blood compared to those without diabetes. Blood pressure has displayed the lowest correlation; however, this does not necessarily imply a lack of significant relationship, as there might be a stronger non-linear correlation between these variables. It's important to consider that the relationship between blood pressure and diabetes may not be as direct or strong compared to other variables in the dataset. Additionally, there could be a stronger non-linear correlation between these variables, and other factors such as age, lifestyle, and genetics may also play a role in influencing this relationship.

## Models

* KNN: I chose K-NN as my initial model because it is widely used for disease prediction. I used the grid search method to find best k. Accuracy: **77.07%**
* Logistic Regression uses linear seperation to classify data into two classes. Accuracy: **78.34%**
* Naive Bayesian: Derived from the Bayes' probability theory. Accuracy: **77.71%**
* Decision Tree: Utilizes information gain for feature selection. Prone to overfitting. Accuracy: **69.42%**
* Support Vector Machine can be used for binary classification. I used 3 different kernels. Linear kernal accuracy: **80.89**, Gaussian accuracy: **77.70%** and Polynomial accuracy: **78.34%**









