import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import models

from warnings import simplefilter #https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
simplefilter(action='ignore', category=FutureWarning)

# Exploratory Data Analysis
def data_description(df):
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    stats_data = {'Stat' : ['mean', 'sd', 'range']}
    stats_data.update({feature: [round(df[feature].mean(), 2), round(df[feature].std(), 2), (min(df[feature]), max(df[feature]))] for feature in col_names})
    stats_df = pd.DataFrame(stats_data)
    return(stats_df)

def outcome_corr_plots(df):
    plt.figure(figsize=(10,14))
    matrix = df.corr()
    outcome_corr = matrix[['Outcome']]
    corr_plot = sns.heatmap(outcome_corr, cmap="Blues", annot= True)
    corr_plot = corr_plot.get_figure()
    plt.savefig(f'figures/outcome_correlation.png')

def corr_plots(df):
    plt.figure(figsize=(16,14))
    matrix = df.corr()
    corr_plot = sns.heatmap(matrix, cmap="Blues", annot= True)
    corr_plot = corr_plot.get_figure()
    plt.savefig(f'figures/matrix_correlation.png')

def feature_histogram(df, cols):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
    axes = axes.flatten()

    for column, ax in zip(cols, axes):
        ax.hist(df[column], bins=10, alpha=0.5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(column)

    plt.tight_layout()
    plt.savefig(f'figures/feature-histogram.png')

def corr_pairplot(df):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    plt.figure(figsize=(16, 14))
    sns.pairplot(df[columns], hue='Outcome')
    plt.savefig('figures/corr_scatter_pairplot.png')


def main():
    df = pd.read_csv('data/diabetes.csv')
    df = models.clean_df(df)
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    outcome_corr_plots(df)
    feature_histogram(df, cols)
    corr_pairplot(df)

    print("Overall Data")
    print(data_description(df),'\n')

    df_0 = df[df['Outcome'] == 0]
    print("Outcome = 0")
    print(data_description(df_0),'\n')

    df_1 = df[df['Outcome'] == 1]
    print("Outcome = 1")
    print(data_description(df_1),'\n')



if __name__ == "__main__":
    main()