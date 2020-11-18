# Online Store Customer Revenue Prediction

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)

- [X] please follow our template
- [ ] Please add references - Facing issues with adding reference. Need assitance
- [X] Please correct the images with correct markdown syntax. 

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


## 3. Methodology

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

Data Pre-Processing is an important step to build a Machine Learning Model. The data pre-processing step typically consists of data cleaning, transformation, standardization and feature selection, so only the most cleaner and accurate data is fed to the models. The dataset we obtained for this project was noticed to contain several data issues such as missing values, less to no variance (zero variance) in the data and also identified the target variable not having random distribution.  The variables such as "totals_newVisits", "totals_bounces", "trafficSource_adwordsClickInfo_page", "trafficSource_isTrueDirect", "totals_bounces", "totals_newVisits" had missing values. We handled the missing values with "zeroes", so the ML algorithms doesn't face any issues. This is a very important step in building the Machine Learning Pipeline. 

- Feature Engineering

Feature Engineering is the process of extracting the hidden signals/features, so the model can use these features to increase the predictive power. This step is the fundamental difference between a good model and a bad model. Also, there is no one-size-fits all approach for Feature Engineering. It is extremely time consuming and requires a lot of domain knowledge as well. 
For this project, we created sophisticated functions to extract date related values such as Month, Year, Data, Weekday, WeekofYear. We also noticed that browser and operating systems are redundant features. Instead of removing them, we combined them to create a combinational feature, so we believe this will increase the predictive power. We also calculated "mean", "sum" and "count" for pageviews and hits, so it can provide more powerful extraction of the feature to the model.

- Feature Selection

Feature Selection refers to selection of features in your data that would improve your machine learning model. There is subtle variation between Feature Selection and Feature Engineering. The Feature Engineering technique is designed to extract more feature from the dataset and the feature selection technique allows only relevant features into the dataset. Also, how do we know what are the relevant features? There are several methodologies and techniques that are developed over the years but there is no one-size-fits-all methodology. 

Feature Selection like Feature Engineering is more of an art than science. There are several iterative procedure that uses Information Gain, Entropy and Correlation scores to decide which feature gets into the model. There are also advanced Deep learning models that can be built or tree based models that can also provide us which of these variables are significant after the model is built. Similar to Feature Engineering, Feature Selection should also require domain specific knowledge to develop festure selection strategies.

For this project, we dropped the features which had no variance in the data and the features that had a lot of null values as well. These features would have not added any value to the dataset. Also, depending on the final result, we can try different strategies in the future to improve the performance of the model.

- Prepare the data

Scikit learn has inbuilt libraries to handle Train/Test Split as part the "model_selection" package. We split the dataset randomly with 80% Training and 20% Testing datasets. 

- Model Algorithms and Optimization Methods

We tested several different Machine Learning Models as shown below. The results of each step and their settings are explained in their respective sections:

- Linear Regression Model
	
	SKLearn Linear Regression performed well overall. We used 5 fold CV. The best RMSE Score for this model we obtained was: 1.89. Also the training and test scores seems to be very close and so we don't suspect any overfitting.  
	
	![Linear Regression](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/linear_regression1.png)	

- XGBoost Regressor
	
	XGBoost Linear Regression performed very well. It was our top performing model with the lowest RMSE error of 1.619. Also the training and test scores seems to be little different and so might have produce overfitting. I tested multiple rounds with different parameters, this was the best performing model overall.  
		
	![XGBoost](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/xgboost1.png)

- LightGBM Regressor	
	
	LightGBM Regression was the second best performing model interms of RMSE Scores.  Also the training and test scores seems to be little different and so might have produce overfitting. I tested multiple rounds with different parameters, this was the best performing model overall.  
	
	![lightgbm](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lightgbm1.png)
	
- Lasso Regression
	
	Lasso performed a bit better than baseline model. However, XGBoost seemed to have performed better than Lasso
	
	![lasso](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lasso1.png)

- Ridge Regressor
	
	Ridge performed a bit better than baseline model. However, XGBoost seemed to have performed better than Ridge
	
	![Ridge](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/ridge1.png)


## 5. Software Technologies

- Python
- Jupyter Notebook (Google Colab)
- Packages: Pandas, Numpy, Matplotlib, sklearn

## 6. Project Timelines

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

[^1]: Kaggle Competition,2019,Predict the Online Store Revenue,[online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/rules>

[^2]: Kaggle Competition,2019,Simple Exploration Baseline,[online] Available at: <https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue>

[^3]: Towards DataScience,2020,Sweetviz- Powerful EDA,[online] Available at: <https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34>















