# library doc string
"""
Helper functions for predicting, customized from churn_notebook.ipynb
Author: Ahmad Hakami
Date: Dec. 31th 2023
"""

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(df_path: str) -> pd.DataFrame:
    """
    Load the DataFrame from the specified path
    Args:
        df_path (str): The path directory of the input csv file
    Returns:
        pd.DataFrame: The read dataframe
    """
    # read dataframe using path directory
    bank_data = pd.read_csv(df_path)
    return bank_data


def process_df(df: pd.DataFrame) -> pd.DataFrame:
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
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)

    # Convert column names to lowercase for consistency
    df.columns = [column.lower() for column in df.columns]
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis (EDA) on the given dataframe
    and save figures to the 'images/eda' folder
    Args:
        df (pd.DataFrame): The pandas dataframe to analyze
    Returns:
        None
    """
    # Identify categorical columns efficiently
    category_columns = list(df.select_dtypes(include='object').columns)

    # plot distribution with categorical features
    for column in category_columns:
        plt.figure(figsize=(7, 5))
        df[column].value_counts(
            normalize=True).plot(
            kind='bar',
            rot=45,
            title=f'{column.capitalize()} Distribution')
        plt.ylabel('Percentage')
        plt.xlabel('Category')
        plt.savefig(
            f'images/eda/{column}_distribution.png',
            bbox_inches='tight')

    # plot distribution of customer age
    plt.figure(figsize=(7, 5))
    df['customer_age'].plot(kind='hist', title='Customer Age Distribution')
    plt.xlabel('Age')
    plt.savefig(
        f'images/eda/customer_age_distribution.png',
        bbox_inches='tight')

    # plot correlation matrix
    plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/heatmap.png', bbox_inches='tight')

    # plot distributions of 'total_trans_ct' with a smooth curve
    plt.figure(figsize=(7, 7))
    sns.histplot(df['total_trans_ct'], stat='density', kde=True)
    plt.title('Total Transaction Distribution')
    plt.savefig(
        'images/eda/total_transaction_distribution.png',
        bbox_inches='tight')


def encode_categorical_columns(dataframe, category_columns, response):
    """
    Encode categorical columns in a dataframe using onehot encoding
    Args:
        dataframe (pd.DataFrame): The input dataframe
        category_columns (list): List of column names containing categorical variables to be encoded
        response (str): Prefix to be added to the one-hot encoded columns
    Returns:
        df_encoded (pd.DataFrame): dataframe with onehot encoded categorical columns
    """
    # Apply onehot encoding to the specified categorical columns
    df_encoded = pd.get_dummies(dataframe,
                                columns=category_columns,
                                drop_first=True,
                                prefix=response)
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
    category_columns = list(df.select_dtypes(include='object').columns)

    # Encode categorical features
    df_encoded = encode_categorical_columns(df, category_columns, response)

    # Split data into features (X) and target variable (y)
    y = df_encoded[response]
    X = df_encoded.drop(response, axis=1)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results
    and and save figures to the 'images/results' folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    output:
        None
    '''
    # ________________________ Random Forest _________________________

    # Set the figure size
    plt.rc('figure', figsize=(5, 5))

    # Plotting text for Random Forest Train
    plt.text(0.01, 1.25, 'Random Forest Train', {'fontsize': 10},
             fontproperties='monospace')

    # Plotting text for classification report on the test set
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    # Plotting text for Random Forest Test
    plt.text(0.01, 0.6, 'Random Forest Test', {'fontsize': 10},
             fontproperties='monospace')

    # Plotting text for classification report on the train set
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    # Turn off axis and save the plot to the specified path
    plt.axis('off')
    plt.savefig('images/results/rf_results.png', bbox_inches='tight')

    # ________________________ Logistic Regression _________________________

    # Set the figure size
    plt.rc('figure', figsize=(5, 5))

    # Plotting text for Logistic Regression Train
    plt.text(0.01, 1.25, 'Logistic Regression Train',
             {'fontsize': 10}, fontproperties='monospace')

    # Plotting text for classification report on the test set
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')

    # Plotting text for Logistic Regression Test
    plt.text(0.01, 0.6, 'Logistic Regression Test',
             {'fontsize': 10}, fontproperties='monospace')

    # Plotting text for classification report on the train set
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')

    # Turn off axis and save the plot to the specified path
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png', bbox_inches='tight')


def feature_importance_plot(model, x_features):
    '''
    creates and stores the feature importances in images/results
    input:
        model: model object containing feature_importances_
        x_features: pandas dataframe of X values

    output:
        None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names
    names = [x_features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_features.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_features.shape[1]), names, rotation=90)

    # Save the plot to the specified path
    plt.savefig(f'images/results/feature_importances.png', bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train models, store results (images + scores), and store models.
    Parameters:
        X_train (array-like): X training data.
        X_test (array-like): X testing data.
        y_train (array-like): y training data.
        y_test (array-like): y testing data.
    Returns:
        None
    '''
    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')

    # Plot ROC curves
    plt.figure(figsize=(15, 8))
    plot_roc_curve(lrc, X_test, y_test, name='Logistic Regression')
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test,
                   name='Random Forest', linestyle='--')
    plt.title("ROC curves")
    plt.savefig('images/results/roc_curve_result.png', bbox_inches='tight')

    # _______________________ Classification Report ________________________________

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    # ______________________ Feature importance ____________________________________
    feature_importance_plot(cv_rfc, X_train)




if __name__ == "__main__":
    # Import data
    print("Importing data...")
    bank_data = import_data('data/bank_data.csv')

    # Process data
    print("Processing data...")
    processed_bank_data = process_df(bank_data)

    # Perform exploratory data analysis (EDA)
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(processed_bank_data)

    # Perform feature engineering and split data
    print("Performing feature engineering and splitting data...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        processed_bank_data)

    # Train models and plotting
    print("Training & Save models and plotting...")
    train_models(X_train, X_test, y_train, y_test)

    print("All steps completed successfully!")
