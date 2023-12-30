# library doc string
"""
Helper functions for predicting, customized from churn_notebook.ipynb
Author: Ahmad Hakami
Date:   Dec. 31th 2023
"""

# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(df_path: str):
    """
    Load the DataFrame from the specified path
    
    Args:
        df_path (str): The path directory of the input dataframe file
        
    Returns:
        pd.DataFrame: The readed dataframe
    """
    df = pd.read_csv(df_path)
    
    return df


def process_df(df: pd.DataFrame):
    """
    Process a pandas dataframe as follows:
    1. Create a new 'Churn' column based on the 'Attrition_Flag' column
    2. Convert column names to lowercase for consistency

    Args:
        df (pd.DataFrame): The input dataframe

    Returns:
        pd.DataFrame: The processed dataframe
    """

    # Create a new 'Churn' column based on the 'Attrition_Flag' column
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == 'Existing Customer' else 1)

    # Convert column names to lowercase for consistency
    df.columns = [column.lower() for column in df.columns]

    return df


def perform_eda(df: pd.DataFrame):
    """
    Perform exploratory data analysis (EDA) on the given dataframe and save figures to the 'images/eda' folder

    Args:
        df (pd.DataFrame): The pandas dataframe to analyze

    Returns:
        None
    """

    # plot distribution with categorical features 
    cat_columns = list(df.select_dtypes(include='object').columns)
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 5))
        df[cat_column].value_counts('normalize').plot(kind='bar', rot=45, title=f'{cat_column.capitalize()} Distribution')
        plt.ylabel('Percentage')
        plt.xlabel('Category')
        plt.savefig(f'images\eda\{cat_column}_distribution.png')
        # plt.show()

    # plot distribution of customer age
    plt.figure(figsize=(7, 5))
    df['customer_age'].plot(kind='hist', title='Customer Age Distribution')
    plt.xlabel('Age')
    plt.savefig(f'images\eda\customer_age_distribution.png')
    # plt.show()

    # plot correlation matrix
    plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images\eda\heatmap.png')
    # plt.show()


def encode_categorical_columns(df, category_columns, response):
    """
    Encode categorical columns in a dataframe using onehot encoding

    Args:
        df (pd.DataFrame): The input dataframe
        category_columns (list): List of column names containing categorical variables to be encoded
        response (str): Prefix to be added to the one-hot encoded columns

    Returns:
        df (pd.DataFrame): dataframe with onehot encoded categorical columns
    """
    # Apply onehot encoding to the specified categorical columns
    df_encoded = pd.get_dummies(df, columns=category_columns, drop_first=True, prefix=response)
    
    return df_encoded


def perform_feature_engineering(df, response='churn'):
    """
    Performs feature engineering, including categorical encoding and train-test split.

    Args:
        df (pd.DataFrame): The input dataframe
        response (str): Name of the response column. Defaults to 'churn'

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # Identify categorical columns efficiently
    cat_columns = list(df.select_dtypes(include='object').columns)

    # Encode categorical features
    df_encoded = encode_categorical_columns(df, cat_columns, response)

    # Split data into features (X) and target variable (y)
    y = df_encoded[response]
    X = df_encoded.drop(response, axis=1)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    df = import_data('data/bank_data.csv')
    processed_df = process_df(df)
    perform_eda(processed_df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(processed_df)
    print('splitted')

    
    
