# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project involves an initial exploratory analysis to understand variable behavior and identify attributes strongly linked to credit card service cancellations. Subsequently, resource engineering techniques are applied, followed by the implementation of a machine learning algorithm to determine optimal resources for model construction. The ultimate goal is to develop a predictive machine learning model that can anticipate customer departure from the credit card service based on [credit card customer dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Files and data description

1. **Guide.ipynb**
  Provides project guidance and troubleshooting tips

2. **churn_notebook.ipynb**
  Contains code to be refactored

3. **churn_library.py**
  File for defining functions related to the churn analysis

4. **churn_script_logging_and_tests.py**
  Script for tests and logs related to churn analysis code

5. **README.md**
  Provides project overview and code usage instructions

6. **data/bank_data.csv**
The dataset from https://leaps.analyttica.com/home assists a bank in predicting credit card customer churn among 10,000 entries with 18 features

7. **images/eda**
  Directory for storing EDA result images

8. **images/results**
  Directory for storing various churn analysis result images

9. **logs**
  Directory for storing logs generated during code execution

10. **models**
   Directory for storing generated models

## Running Files
To run the churn analysis code, you need to follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the directory where the code is located 
```
cd <path-directory>
```

3. create a new environment with a specific Python version `python==3.6`
```bash
conda create -n <env-name> python==3.6
```
*you can change it to Python 3.8* 

4. Activate the new environment
```bash
conda activate <env-name>
```

5. Install required packages and libraries 

For Python 3.6
```bash
pip install -r requirements_py3.6.txt
```

For Python 3.8
```bash
pip install -r requirements_py3.8.txt
```

6. Run the churn analysis script by executing the following command:

```bash
python churn_library.py
```

7. For testing and checking logging messages, run:
```bash
python churn_script_logging_and_tests.py
```

8. Both `churn_script_logging_and_tests.py` and `churn_library.py` files achieve a score of **+7** in `pylint test`. To check the result, run:
```bash
pylint churn_script_logging_and_tests.py
```
*and*
```bash
pylint churn_library.py
```