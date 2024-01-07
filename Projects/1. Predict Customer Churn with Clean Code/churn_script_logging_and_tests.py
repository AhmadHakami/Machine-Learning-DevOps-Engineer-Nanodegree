"""
Test functions for various components of the churn_library.py file
Author: Ahmad Hakami
Date: Dec. 31th 2023
"""

# import libraries
import logging
from churn_library import import_data, process_df, perform_eda, encode_categorical_columns, perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def test_import():
    """
    Test data import - this example is completed for you to assist with the other test functions
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_processer():
    """
    Test processer dataframe
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        process_df(dataframe)
        logging.info("Testing process_df: SUCCESS")
    except Exception as err:
        logging.error(f"Testing process_df: Failed with error - {err}")


def test_eda():
    """
    Test perform eda function
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        processed_df = process_df(dataframe)
        perform_eda(processed_df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error(f"Testing perform_eda: Failed with error - {err}")


def test_encoder_helper():
    """
    Test encoder helper
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        processed_df = process_df(dataframe)
        category_columns = list(processed_df.select_dtypes(include='object').columns)
        response = 'churn'
        encode_categorical_columns(processed_df, category_columns, response)
        logging.info("Testing encode_categorical_columns: SUCCESS")
    except Exception as err:
        logging.error(f"Testing encode_categorical_columns: Failed with error - {err}")


def test_perform_feature_engineering():
    """
    Test perform_feature_engineering
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        processed_df = process_df(dataframe)
        perform_feature_engineering(processed_df, 'churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(f"Testing perform_feature_engineering: Failed with error - {err}")


def test_train_models():
    """
    Test train_models
    """
    try:
        dataframe = import_data("data/bank_data.csv")
        processed_df = process_df(dataframe)
        x_train, x_test, y_train, y_test = perform_feature_engineering(processed_df)

        train_models(x_train, x_test, y_train, y_test)

        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error(f"Testing train_models: Failed with error - {err}")


if __name__ == "__main__":
    logging.info("Starting test suite for churn_library.py file")

    logging.info("Testing import_data function")
    test_import()

    logging.info("Testing process_df function")
    test_processer()

    logging.info("Testing perform_eda function")
    test_eda()

    logging.info("Testing encode_categorical_columns function")
    test_encoder_helper()

    logging.info("Testing perform_feature_engineering function")
    test_perform_feature_engineering()

    logging.info("Testing train_models function")
    test_train_models()

    logging.info("Test suite completed")
