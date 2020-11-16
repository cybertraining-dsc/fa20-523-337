# Online Store Customer Revenue Prediction

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)

- [ ] please follow our template
- [ ] Please add references 
- [ ] Please correct the images with correct markdown syntax. 

Balaji Dhamodharan, bdhamodh@iu.edu, fa20-523-337
Anantha Janakiraman, ajanakir@iu.edu, fa20-523-337

[Edit](https://github.com/cybertraining-dsc/fa20-523-337/edit/master/project/project.md)

{{% pageinfo %}}

## Abstract
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies. We're challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** ecommerce,  regression analysis, big data

## 1. Introduction

Our objective is to predict the natural log of total revenue per customer which is a real valued continuous output and linear regression would be an ideal algorithm in such a setting to predict the response variable that is continuous using a set of predictor variables given the basic assumption that there is a linear relationship between the predictor and response variables.

## 2. Datasets

- <https://www.kaggle.com/c/ga-customer-revenue-prediction>

The dataset we used is from the Kaggle Competition. The dataset contains two csv files.
- Train.csv, User transactions from August 1st, 2016 to August 1st, 2017, 903.6K records
- Test.csv, User transactions from August 2nd, 2017 to April 30th, 2018, 804.6K records


The metrics we will use for this project is root mean squared error (RMSE). The root mean squared error function forms our objective/cost function which will be minimized to estimate the optimal parameters for our linear function through Gradient Descent. We will conduct multiple experiments to obtain convergence using different “number of iterations” value and other hyper-parameters (e.g. learning rate).

RMSE is defined as:

![Figure 2.1](https://github.com/cybertraining-dsc/fa20-523-337/blob/master/project/Images-and-plots/Loss_Func.png)  

where y-hat is the natural log of the predicted revenue for a customer and y is the natural log of the actual summed revenue value plus one as seen below.


## 3. Machine Learning Algorithm and Implementation

We used CRISP-DM process methodology for this project. The  high-level representation of the implementation steps explained in detail below:

- Get Data

The data was obtained from Kaggle Competition. Please find the details of the dataset in the above section

- Load Data
	-  <https://storage.googleapis.com/gstoretrain/train.csv>
	-  <https://storage.googleapis.com/gstoretrain/test.csv>

	The data we obtained from Kaggle was over 2.6 GB (for Train and Test). As the size of the dataset was large, I have created a bucket in Google Store where the data can be downloaded. The URL is provided here for your reference:

- Data Exploration
The dataset we obtained contained 54 Independent Variable and 1 Dependent variable. The Dependent Variable is "totals.transactionRevenue". The end goal of this project is to predict the revenue of the Online Store Customer as close as possible.

- Data Pre-Processing

- Feature Engineering

- Build the data pipeline

- Model Algorithms and Optimization Methods

We are planning to explore different algorithms as shown below:

- Linear Regression Model

- XGBoost Regressor

- LightGBM Regressor

- Lasso Regression

- Ridge Regressor

- Feature Importance
	- Xgboost Regressor
	- LightGBM Regressor
	- Lasso Regressor Regressor
	- Ridge Regressor
- Results Validation

## 5. Technologies

- Python
- Jupyter Notebook (Google Colab)
- Packages: Pandas, Numpy, Matplotlib, sklearn

## 6. Project Timeline

### October 16

- Explore the data set
- Explore ML Framework
- Perform Basic EDA

### November 2

- The dataset was large. Identified the problem was a regression problem
- Explored different variables and their distributions
- Build base line model

### November 9

- Build ML models using LR, XGBoost, LightGBM
- Review Results

### November 16
- Report and document findings

## 7. References

[^1]: Kaggle Competition, 2019, Predict the Online Store Revenue [online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/rules> [Accessed 21 October 2020]

[^2]: Kaggle Competition, 2019,  Simple Exploration Baseline, [online] Available at: <https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue> [Accessed 21 October 2020] 

[^3]: Towards DataScience, 2020,  Sweetviz- Powerful EDA, [online] Available at: <https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34> [Accessed 21 October 2020] 















