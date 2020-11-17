# Online Store Customer Revenue Prediction

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)

- [ ] please follow our template
- [ ] Please add references 
- [ ] Please correct the images with correct markdown syntax. 

Balaji Dhamodharan, bdhamodh@iu.edu, fa20-523-337
Anantha Janakiraman, ajanakir@iu.edu, fa20-523-337

[Edit](https://github.com/cybertraining-dsc/fa20-523-337/edit/main/project/project.md)

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


The metrics we will use for this project is root mean squared error (RMSE). The root mean squared error function forms our objective/cost function which will be minimized to estimate the optimal parameters for our linear function through Gradient Descent. We will conduct multiple experiments to obtain convergence using different "number of iterations" value and other hyper-parameters (e.g. learning rate).

RMSE is defined as:

![Figure 2.1](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/loss.png)  

where y-hat is the natural log of the predicted revenue for a customer and y is the natural log of the actual summed revenue value plus one as seen below.


## 3. Machine Learning Algorithm and Implementation

We used CRISP-DM process methodology for this project. The  high-level representation of the implementation steps explained in detail below:

- Get Data

The data was obtained from Kaggle Competition. Please find the details of the dataset in the above section

- Load Data
	-  <https://storage.googleapis.com/gstoretrain/train.csv>
	-  <https://storage.googleapis.com/gstoretrain/test.csv>

	The data we obtained from Kaggle was over 2.6 GB (for Train and Test). As the size of the dataset was large, I created a storage bucket in GCP to host the data. The URL is provided above for your reference.Also for this project we are using only the Train data to build our models and test the results because the real end goal is to test the effectiveness of algorithm. Since the test set doesnt contain the Target Variable (rightly so!), we won't be consuming the test set for our analysis.	

- Data Exploration

The dataset obtained for this project is large.  It had contained over 900k records. The dataset also contained 12 Independent Variable and 1 Dependent variable. The Dependent Variable is totals.transactionRevenue. 
The end goal of this project is to predict the revenue of the Online Store Customer as close as possible.

- Target Variable

The Target Variable is totals.transactionRevenue has the transaction value of each visit. But, this column contains 98.72% of missing values for revenue (no purchase). The Target variable had a skewed distribution, we performed a lognormal distribution, so the target variable has normal distribution.

![Target Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/y_after_transformation.png)  

- Exploratory Data Analysis

	-	Browser
		
		The most popular browser is Google Chrome. Also, we noticed second best users were using safari browser and firefox was placed third
		
		![Browser Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_browser.png)  
				
	-	Device Category
	
		About 70% of users were accessing online store via desktop
		
		![Device Category Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_category.png)		
		
	-	OperatingSystem
		
		Windows is still the popular operating system among the Google Store visitors among the desktop users. However, among the mobile users, what's interesting is, almost equal number of ios users (slightly lower) as android users access google play store.
		
		![Device Category Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_operating_system.png)		
		
	-	GeoNetwork-City
		
		Mountainview, California tops the cities list. However in the top 10 cities, 4 cities are from California. 
		
		![GeoNetwork-City](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_city.png)  
				
	-	GeoNetwork-Country
		
		Customers from US are way ahead of other customers from different countries. May be this could be due to the Online Store data that was provided to us was pulled from US Google Play Store (Possible BIAS!!!)
		
		![GeoNetwork-Country](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_country.png)		
	
	-	GeoNetwork-Region
		
		No surprise here, as we are already aware that majority of the customers are from US, so America region tops the list
		
		![GeoNetwork-Region](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_continent.png)
			
	-	GeoNetwork-Metro
		
		SFO tops the list for all metro cities, followed by New York and then London
		
		![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_metro.png)	
		
	-	Ad Sources
	
		Google Merchandise and Google Online Store are the top sources where the traffic is coming from to the Online Store. 
		
		![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/traffic_source_adcontent.png)	
	
- Data Pre-Processing

Data Pre-Processing is an important step to build a Machine Learning Model. The data pre-processing step typically consists of data cleaning, transformation and feature selection, so only the most cleaner and accurate data is fed to the models. The dataset we obtained for this project was noticed to contain several data issues such as missing values, less to no variance (zero variance) in the data and also identified the target variable not having random distribution.  The variables such as "totals_newVisits", "totals_bounces", "trafficSource_adwordsClickInfo_page", "trafficSource_isTrueDirect", "totals_bounces", "totals_newVisits" had missing values. We handled the missing values with "zeroes", so the ML algorithms doesn't face any issues. This is a very important step in building the Machine Learning Pipeline. 

- Feature Engineering

Created sophisticated functions to encode categorical variables. As this is regression problem, we need the dataset to be encoded categorically, so all the categories can be converted as numeric fields. We also created other functions to process date time, extract sum, count and mean from numeric variables such as "geoNetwork_networkDomain", "totals_hits" 

- Model Algorithms and Optimization Methods

I tested the following algorithms 

- Linear Regression Model

- XGBoost Regressor

- LightGBM Regressor

- Lasso Regression

- Ridge Regressor

- Results Validation

I found the XGBoost Method to have the lowest RMSE error. 

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

## 7. Conclusion

- Will be populated soon

## 8. References

[^1]: Kaggle Competition,2019,Predict the Online Store Revenue,[online]. Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/rules>

[^2]: Kaggle Competition,2019,Simple Exploration Baseline,[online]. Available at: <https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue>

[^3]: Towards DataScience,2020,Sweetviz- Powerful EDA,[online]. Available at: <https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34>















