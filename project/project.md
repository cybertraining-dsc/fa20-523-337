# Online Store Customer Revenue Prediction

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)
Status: in progress
- [x] please follow our template
- [x] Please add references - Facing issues with adding reference. Need assistance
- [ ] Please correct the images with correct markdown syntax. 
- [x] usage of italic vs quotes
- [ ] wrong indentation level when doing lists, student does not do markdown
- [ ] wrong indentation while using paragraphs they must not be indented
- [x] wrong indentation when using images, images must not be in bullet lists but stands long, images must be refered to in text as Figure x
- [ ] missing empty line before captions
- [x] future considerations should be renamed to Limitations. 
- [ ] there are no : in headers such as in future considerations:
- [ ] use grammerly
- [ ] your tables are unreadable. There are different ways on how to do this. You can include the parameters as text and the rest as markdown table
- [ ] hid from second author wrong
- [ ] you are not looking at the output of the check report script some errors are listed there
- [ ] the word below and above must never be used in a formal paper to refer to figures and tables and sections, use numbers as we posted in piszza
- [ ] bullet lists must not be used in substitution for subsections. You could **bf**. them and do not use a bullet similar to LaTeX paragraphs if you do not want to use subsections.SUbsections show up in the TOC, **bf**. does not
- [ ] no explanation is provided what teh different regressiosn are, no citations provided
- [ ] all figures must have captions (below)
- [ ] all tables must have captions (above)
- [ ] This is not a ppt presentations. for example

  * Kaggle - Customer Revenue Prediction

  is not a full sentence and must not be used to start a section
  
Balaji Dhamodharan, bdhamodh@iu.edu, fa20-523-337
Anantha Janakiraman, ajanakir@iu.edu, fa20-523-337  

[Edit](https://github.com/cybertraining-dsc/fa20-523-337/edit/main/project/project.md)

{{% pageinfo %}}

## Abstract

**Situation** The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies. The objective of this project is to explore different machine learning techniques and identify an optimized model that can help the marketing team understand customer behavior and make informed decisions.

**Task** The challenge is to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. Hopefully this exploration will lead to actionable insights and help allocating marketing budgets for those companies who choose to use data analysis on top of GA data [^1]. 

**Action** This exploration is based on a Kaggle competition and there are two datasets available in Kaggle. One is the test dataset (test.csv) and the other one is the training dataset (train.csv) and together the datasets contain customer transaction information ranging from August 2016 to April 2018. The action plan for this project is to first conduct data exploration that includes but not limited to investigating the statistics of data, examining the target variable distribution and other data distributions visually, determining imputation strategy based on the nature of missing data, exploring different techniques to scale the data, identifying features that may not be needed - for example, columns with constant variance, exploring different encoding techniques to convert categorical data to numerical data and identifying features with high collinearity. The preprocessed data will then be trained using a linear regression model with basic parameter setting and K-Fold cross validation. Based on the outcome of this initial model further experimentation will be conducted to tune the hyper parameters including regularization and also add new derived features to improve the accuracy of the model. Apart from linear regression other machine learning techniques like ensemble methods will be explored and compared.

**Result** The best performing model determined based on the RMSE value will be used in the inference process to predict the revenue per customer. The Kaggle competition requires to predict the natural log of sum of all transactions per customer

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** ecommerce,  regression analysis, big data


## 1. Introduction

The objective of this exploration is to predict the natural log of total revenue per customer which is a real valued continuous output and an algorithm like linear regression will be ideal to predict the response variable that is continuous using a set of predictor variables given the basic assumption that there is a linear relationship between the predictor and response variables.


## 2. Datasets

As mentioned in the earlier sections, the dataset used in this model exploration is from Kaggle [^1]. The link to the dataset is provided below. The training contains more than 900K observations and based on the size of the dataset it would be ideal to use mini-batch or gradient descent optimization techniques to identify the coefficients that best describe the model. The target variable as observed in the dataset is a continuous variable which implies that the use case is a regression problem. As mentioned earlier, there are several machine learning techniques that can be explored for this type of problem including regression and ensemble methods with different parameter settings. The sparsity of potential features in the datasets indicates that multiple experimentations will be required to determine the best performing model. Also based on initial review of the datasets, it also observed that some of the categorical features exhibit low to medium cardinality and if these features are going to be retained in the final dataset used for training then it is important to choose the right encoding technique. 

- Train.csv, User transactions from August 1st, 2016 to August 1st, 2017 [^2]
- Test.csv, User transactions from August 2nd, 2017 to April 30th, 2018  [^2]

#### Metrics

The metrics used for evaluation in this analysis is the root mean squared error (RMSE). The root mean squared error function forms the objective/cost function which will be minimized to estimate optimal parameters for the linear function using Gradient Descent. The plan is to conduct multiple experiments with different iterations to obtain convergence and try different hyper-parameters (e.g. learning rate).

RMSE is defined as:

![Figure 1.1](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/loss.png)  

**Figure 1:** RMSE

where y-hat is the natural log of the predicted revenue for a customer and y is the natural log of the actual summed revenue value plus one as seen below.

## 3. Methodology

The CRISP-DM process methodology was followed in this project. The  high-level implementation steps are shown in the image below:

![Methodology](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/methodology.png)

**Figure 2:** Project Methodology

### 3.1 Load Data:

The data we obtained from Kaggle was over 2.6 GB (for Train and Test). As the size of the dataset was large, I created a storage bucket in GCP to host the data. The URL is provided above for your reference.Also for this project we are using only the Train data to build our models and test the results because the real end goal is to test the effectiveness of algorithm. Since the test set doesnt contain the Target Variable (rightly so!), we won't be consuming the test set for our analysis.	

	- [Kaggle - Customer Revenue Prediction - Train Dataset](https://storage.googleapis.com/gstoretrain/train.csv)
	- [Kaggle - Customer Revenue Prediction - Test Dataset](https://storage.googleapis.com/gstoretrain/test.csv)	

### 3.2 Data Exploration:

The dataset obtained for this project is large.  It had contained over 900k records. The dataset also contained 12 Independent Variable and 1 Dependent variable. The Dependent Variable is totals.transactionRevenue.  The end goal of this project is to predict the revenue of the Online Store Customer as close as possible.

**Target Variable:** The Target Variable is totals.transactionRevenue has the transaction value of each visit. But, this column contains 98.72% of missing values for revenue (no purchase). The Target variable had a skewed distribution, we performed a lognormal distribution, so the target variable has normal distribution.

![Target Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/y_after_transformation.png) 

**Figure 3:** Target Variable
 

- Exploratory Data Analysis

	-	Browser
		
		The most popular browser is Google Chrome. Also, we noticed second and third best users were using safari and firefox respectively
		
		![Browser Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_browser.png)		
				
		**Figure 4:** Browser Variable
		
	-	Device Category
	
		Almost 70% of users were accessing online store via desktop
		
		![Device Category Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_category.png)		
		
		**Figure 5:** Device Category Variable
		
	-	OperatingSystem
		
		Windows is the popular operating system among the desktop users. However, among the mobile users, what's interesting is, almost equal number of ios users (slightly lower) as android users accessed google play store. The reason why this is interesting is because, google play store is primarily used by android users and ios users almost always use Apple Store for downloading apps to their mobile devices. So it is interesting to know, almost equal number of ios users visit google store as well.
		
		![OperatingSystem](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_operating_system.png)
			
		**Figure 6:** Operating System	
		
	-	GeoNetwork-City
		
		Mountain View, California tops the cities list for the users who accessed online store. However in the top 10 cities, 4 cities are from California. 
		
		![GeoNetwork-City](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_city.png)  
				
		**Figure 7:** GeoNetwork City	
		
	-	GeoNetwork-Country
		
		Customers from US are way ahead of other customers from different countries. May be this could be due to the fact that online store data that was provided to us was pulled from US Google Play Store (Possible BIAS!!!).
		
		![GeoNetwork-Country](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_country.png)		
	
		**Figure 8:** GeoNetwork Country
		
	-	GeoNetwork-Region
		
		We are already aware that majority of the customers are from US, so America region tops the list.
		
		![GeoNetwork-Region](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_continent.png)
		
		**Figure 9:** GeoNetwork Region
		
	-	GeoNetwork-Metro
		
		SFO tops the list for all metro cities, followed by New York and then London.
		
		![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_metro.png)	
		
		**Figure 10:** GeoNetwork Region
		
	-	Ad Sources
	
		Google Merchandise and Google Online Store are the top sources where the traffic is coming from to the Online Store. 
		
		![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/traffic_source_adcontent.png)	
		
		**Figure 11:** Ad Sources
	
- Data Pre-Processing

Data Pre-Processing is an important step to build a Machine Learning Model. The data pre-processing step typically consists of data cleaning, transformation, standardization and feature selection, so only the most cleaner and accurate data is fed to the models. The dataset we obtained for this project was noticed to contain several data issues such as missing values, less to no variance (zero variance) in the data and also identified the target variable not having random distribution.  The variables such as totals_newVisits, totals_bounces, trafficSource_adwordsClickInfo_page, trafficSource_isTrueDirect, totals_bounces, totals_newVisits had missing values. We handled the missing values with zeroes, so the ML algorithms doesn't face any issues. This is a very important step in building the Machine Learning Pipeline. 

- Feature Engineering

Feature Engineering is the process of extracting the hidden signals/features, so the model can use these features to increase the predictive power. This step is the fundamental difference between a good model and a bad model. Also, there is no one-size-fits all approach for Feature Engineering. It is extremely time consuming and requires a lot of domain knowledge as well. 
For this project, we created sophisticated functions to extract date related values such as Month, Year, Data, Weekday, WeekofYear. We also noticed that browser and operating systems are redundant features. Instead of removing them, we combined them to create a combined feature and believe this will increase the predictive power. We also calculated mean, sum and count for pageviews and hits, so it can provide more powerful extraction of the feature to the model.

- Feature Selection

Feature Selection refers to selection of features in your data that would improve your machine learning model. There is subtle variation between Feature Selection and Feature Engineering. The Feature Engineering technique is designed to extract more feature from the dataset and the feature selection technique allows only relevant features into the dataset. Also, how do we know what are the relevant features? There are several methodologies and techniques that are developed over the years but there is no one-size-fits-all methodology. 

Feature Selection like Feature Engineering is more of an art than science. There are several iterative procedure that uses Information Gain, Entropy and Correlation scores to decide which feature gets into the model. There are also advanced Deep learning models that can be built or tree based models that can also provide us which of these variables are significant after the model is built. Similar to Feature Engineering, Feature Selection should also require domain specific knowledge to develop festure selection strategies.

For this project, we dropped the features which had no variance in the data and the features that had a lot of null values as well. These features would have not added any value to the dataset. Also, depending on the final result, we can try different strategies in the future to improve the performance of the model.

- Prepare the data

Scikit learn has inbuilt libraries to handle Train/Test Split as part the model_selection package. We split the dataset randomly with 80% Training and 20% Testing datasets. 

- Model Algorithms and Optimization Methods

We tested several different Machine Learning Models as shown below. The results of each step and their settings are explained in their respective sections:

- Linear Regression Model

Linear regression is a supervised learning model that follows a linear approach in that it assumes a linear relationship between one ore more predictor variables (x) and a single target or response variable (y). The target variable can be calculated as a linear combination of the predictors or in other words the target is the calculated by weighted sum of the inputs and bias where the weighted is estimated through different optimization techniques. Linear regression is referred to as simple linear regression when there is only one predictor variable involved and referred to as multiple linear regression when there is more than one predictor variable is involved. The error between the predicted output and the ground truth is generally calculated using RMSE (root mean squared error). This is one of the classic modeling techniques that will be explored in this project because the target variable (revenue per customer) is a real valued continuous output and exhibits a significant linear relationship with the independent variables or the input variables.

In the exploration, SKLearn Linear Regression performed well overall. We used 5 fold Cross validation and the best RMSE Score for this model observed was: 1.89. The training and test RMSE error values are very close indicating that there is no overfitting the data.
	
![Linear Regression](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/linear_regression3.png)	
	
**Figure 12:** Linear Regression Model 
	
REFERNCE IN TEXT TO Figure 1 missing

- XGBoost Regressor

XGBoost regression is a gradient boosting regression technique and one of the popular gradient boosting frameworks that exists today. It follows the ensemble principle where a collection of weak learners improves the prediction accuracy. The prediction in the current step S is weighed based on the outcomes from the previous step S-1. Weak learning is slightly better than random learning and that is one of the key strengths of gradient boosting technique. The XGBoost algorithm was explored for this project for several reasons including it offers built-in regularization that helps avoid overfitting, it can handle missing values effectively and it also does cross validation automatically. The feature space for the dataset that we are using is sparse and believe the potential to overfit the data is high which is one of the primary reasons for exploring XGBoost.
    
XGBoost Linear Regression performed very well. It was our top performing model with the lowest RMSE error of 1.619. Also the training and test scores are reasonably close and it doesn't look like there was the problem of over fitting the training data. We tested multiple rounds with different parameters, this was the best performing model overall. 
		
![XGBoost](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/xgboost3.png)
	
**Figure 13:** XGBoost Model 	
	
- LightGBM Regressor	

LightGBM is a popular gradient boosting framework similar to XGBoost and is gaining popularity in the recent days. The important difference between lightGBM and other gradient boosting frameworks is that LightGBM grows the tree vertically or in other words it grows the tree leafwise compared to other frameworks where the trees grow horizontally. In this project the lightGBM framework was experimented primarily because this framework works well on large dataset with more than 10K observations. The algorithm also has a high throughput while using reasonably less memory but there is one problem with overfitting the data which we have controlled in our exploration using appropriate hyper parameter setting and optimized the performance.

LightGBM Regression was the second best performing model in terms of RMSE Scores.  Also the training and test scores seems to be little different and so might have produce overfitting. I tested multiple rounds with different parameters, this was the best performing model overall.  
	
![lightgbm](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lightgbm3.png)
	
**Figure 14:** LightGBM Model 
	
- Lasso Regression

Lasso is a regression technique that uses L1 regularization. In statistics, lasso regression is a method to do automatic variable selection and regularization to improve prediction accuracy and performance of the statistical model. Lasso regression by nature makes the coefficient for some of the variables zero meaning these variables are automatically eliminated from the modeling process. The L1 regularization parameter helps control overfitting and will need to be explored for a range of values for the specific problem. When the regularization penalty tends to be zero there is no regularization, and the loss function is mostly influenced by the squared loss and in contrary if the regularization penalty tends to be closer to infinity then the objective function is mostly influenced by the regularization part. It is always ideal to explore a range of values for the regularization penalty to improve the accuracy and avoid overfitting. 

In this project, Lasso is one of the important techniques that was explored primarily because the problem being solved is a regression problem and there is possibility to overfit the data due to the number of observations and feature space. As a team, we explored different ranges for regularization penalty and identified the appropriate value that helped achieve maximum reduction in the total RMSE score.

Lasso performed a bit better than baseline model. However, XGBoost seemed to have performed better than Lasso
	
![lasso](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lasso3.png)
	
**Figure 15:** Lasso Model 

- Ridge Regressor

Ridge is a regression technique that uses L2 regularization. Ridge regression does not offer automatic variable selection in the sense that it is not make the weights zero on any of the variable used in the model and the regularization term in a ridge regression is slightly different than the lasso regression. The regularization term is the sum of the square of coefficients multiplied by the penalty whereas in lasso it is the sum of the absolute value of the coefficients. The regularization term is a gentle trade-off between fitting the model and overfitting the model and like in lasso it helps improve prediction accuracy as well as performance of the statistical model. The L2 regularization parameter helps control overfitting and will need to be explored for a range of values for the specific problem. The regularization parameter also helps reduce multicollinearity in the model. Similar to Lasso, when the regularization penalty tends to be zero there is no regularization, and the loss function is mostly influenced by the squared loss and in contrary if the regularization penalty tends to be closer to infinity then the objective function is mostly influenced by the regularization part. It is always ideal to explore a range of values for the regularization penalty to improve the accuracy and avoid overfitting. 

In this project, Ridge regression is one of the important techniques that was explored again primarily because the problem being solved is a regression problem and there is possibility to overfit the data due to the number of observations and feature space. As a team, we explored different ranges for regularization penalty and identified the appropriate value that helped achieve maximum reduction in the total RMSE score. Ridge performed a bit better than baseline model. However, XGBoost seemed to have performed better than Ridge
	
![Ridge](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/ridge3.png)
	
**Figure 16:** Ridge Model 


## 5. Software Technologies

In this project we used Python and Google Colab Jupyter Notebook. We used several Python pancakes to execute this project such as Pandas, Numpy, Matplotlib, sklearn

## 6. Project Timelines

* October 16
  - Explore the data set
  - Explore ML Framework
  - Perform Basic EDA
* November 2
  - The dataset was large. Identified the problem was a regression problem
  - Explored different variables and their distributions
  - Build base line model
* November 9
  - Build ML models using LR, XGBoost, LightGBM
  - Review Results
* November 16
  - Report and document findings

## 7. Conclusion

As a project team the intention was to create a template that can be utilized for any ML project. The dataset that was used for this project was challenging in a way that it required a lot of data cleaning, flattening and transformation to get the data into the required format. 

### 7.1 Model Pipeline:

In this project, multiple regression and tree based models from scikit learn library were explored with various hyper parameter setting and other methods like the lightGBM. The goal of the model pipeline was to explore and examine data, identify data pre-processing methods, imputation strategies, derive features and try different feature extraction methods was to perform different experiments with different parameter setting and identify the optimized with low RMSE that can be operationalized in Production. The parameters were explored within the boundary of this problem setting using different techniques.

### 7.2 Feature Exploration and Pre-Processing:

As part of this project a few features were engineered and included in the training dataset. The feature importance visualizations that were generated after the model training process indicate that these engineered features were part of the top 30% of high impact features and they contributed reasonably to improving the overall accuracy of the model. We as a team discussed on the possibility of including few other potential features that could be derived from the dataset during additional experimentation phase and included those additional features as well to the dataset that was used during model training. Although these features did not contribute largely to reducing the error it gave us as a team an opportunity to share ideas and methods to develop these new features. Also, during the discussions we also tried to explore other imputation strategies, identify more outliers and tried different encoding techniques for categorical variables and ultimately determined that label encoder or ordinal encoder is the best way forward. We also tried to exclude some of the low importance features and retrained the model to validate if the same or better RMSE value could be achieved.

### 7.3 Outcome of Experiments:

Multiple modeling techniques were explored as part of this project like Linear regression, gradient boosting algorithms and linear regression regularization techniques. The techniques were explored with basic parameter setting and based on the outcome of those experiments, the hyper parameters were tuned using grid search to obtain the best estimator evaluated on RMSE. Also, during grid search K-Fold cross validation of training data was used and the cross validated results were examined through a results table. The fit_intercept flag played a significant role resulting in an optimal error. As part of the different experimentations that were performed, random forest algorithm was also explored but it suffered performance issues and it seemed like it would require more iterations to converge which is why it was dropped from our results and further exploration. Although random forest was not explored, gradient boosting techniques were part of the experimentations and the best RMSE from XGBoost. The LightGBM regressor was also explored with different parameter settings but it did not produce better RMSE score than XGBoost. 

In the case of XGBoost, there was improvement to the RMSE score as different tree depths, feature fraction, learning rate, number of children, bagging fraction, sub-sample were explored. There was significant improvement to the error metric when these parameters were adjusted in an intuitive way. Also, linear regression with regularization techniques were explored and although there was some improvement to the error metric compared to the basic linear regression model they did not perform better than the gradient boosting method that was explored. So, based on different explorations and experimentations we are able to reasonably conclude that gradient boosting technique performed better for the given problem setting and generated the best RMSE score. Based on the evaluation results of XGBoost on the dataset used, the recommendation would be to test the XGBoost model with real time data and the performance of the model can be evaluated in real-time scenario too and additionally, if needed, hyper parameter tuning can be performed on the XGBoost model specifically for the real-time scenario.

We also performed feature engineering in this dataset to get more predictive value out of the features and built a pipeline, so it can be easily fed into different models. We tested five different models, such as, Linear Regression, XGBoost, Light GBM, Lasso and Ridge. The summary of all the models is shown below for your reference:

![Model_Results](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/model_results_new2.png)

**Figure 17:** Model Results Summary 


### 7.4 Limitations:

Due to the limited capacity of our Colab Notebook setup, we were unable to perform Cross Validation for XGBoost and Light GBM. We recommend to perform cross validation for these models, check the RMSE Scores and potentially avoid overfitting, if any. The tree based models performed well in this dataset. In the future try additional tree based models like Random Forest to evaluate their performance

## 8. Previous Explorations

The Online GStore customer revenue prediction problem is a Kaggle competition with more than 4100 entries. It is one of the popular challenges in Kaggle with a prize money of $45,000. Although as a team we did not make it to the top in the leader board, the challenge gave us a huge opportunity to explore different methods, techniques, tools and resources. The one important difference between many of the previous of explorations versus what we achieved is the number of different machine learning algorithms that we explored and observed the performance. We reviewed many submissions in Kaggle and found only a very few entries exploring different parameter settings and making intuitive adjustments to them to make the model perform at an optimum level like what we accomplished. The other uniqueness that we brought to our submission was identifying techniques that offered good performance and consumed less system resources in terms of operationalization. We will continue to do further exploration and attempt other techniques to identify the best performing model.


## 9. References

[^1]: Kaggle Competition,2019,Predict the Online Store Revenue,[online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/rules>

[^2]: Kaggle Competition,2019,Predict the Online Store Revenue, Data, [online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/data>

[^3]: Towards DataScience,2020,Sweetviz- Powerful EDA,[online] Available at: <https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34>

[^4]: Towards Datascience,2018,Bhattacharya, Ridge and Lasso regression, [online] Available at: <https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b>

[^5]: Datacamp,2019,Oleszak, Regularization: Ridge, Lasso and Elastic Net, [online] Available at: <https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net>

[^6]: Medium 2017,Mandot, What is LightGBM, [online] Available at: <https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc>

[^7]: Kaggle 2018,Kaggle, Google Analytics Customer Revenue Prediction, [online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/data>

[^8]: XGBoost 2020,xgboost developers, XGBoost, [online] Available at: <https://xgboost.readthedocs.io>

[^9]: Datacamp,2019,Pathak, Using XGBoost in Python, [online] Available at: <https://www.datacamp.com/community/tutorials/xgboost-in-python>

[^10]: Towards Datascience,2017,Lutins, Ensemble Methods in Machine Learning, [online] Available at: <https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f>

[^11]: Machine Learning Mastery,2016,Brownlee, Linear Regression Model, [online] Available at: <https://machinelearningmastery.com/linear-regression-for-machine-learning>

[^12]: Kaggle 2018,Lee, Google Analytics Customer Revenue Prediction, [online] Available at: <https://www.kaggle.com/youhanlee/which-encoding-is-good-for-time-validation-1-4417>

[^13]: Kaggle 2018,Daniel, Google Analytics Customer Revenue Prediction, [online] Available at: <https://www.kaggle.com/fabiendaniel/lgbm-starter>















