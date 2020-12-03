# Online Store Customer Revenue Prediction

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-337/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-337/actions)
Status: final
- [x] please follow our template
- [x] Please add references - Facing issues with adding reference. Need assistance
- [x] Please correct the images with correct markdown syntax. 
- [x] usage of italic vs quotes
- [x] wrong indentation level when doing lists, student does not do markdown
- [x] wrong indentation while using paragraphs they must not be indented
- [x] wrong indentation when using images, images must not be in bullet lists but stands long, images must be refered to in text as Figure x
- [x] missing empty line before captions
- [x] future considerations should be renamed to Limitations. 
- [x] there are no : in headers such as in future considerations:
- [x] use grammerly
- [x] your tables are unreadable. There are different ways on how to do this. You can include the parameters as text and the rest as markdown table
- [x] hid from second author wrong
- [x] you are not looking at the output of the check report script some errors are listed there
- [x] the word below and above must never be used in a formal paper to refer to figures and tables and sections, use numbers as we posted in piszza
- [x] bullet lists must not be used in substitution for subsections. You could **bf**. them and do not use a bullet similar to LaTeX paragraphs if you do not want to use subsections.SUbsections show up in the TOC, **bf**. does not
- [x] no explanation is provided what the different regression are, no citations provided
- [x] all figures must have captions (below)
- [x] all tables must have captions (above)
- [x] This is not a ppt presentations. for example

  * Kaggle - Customer Revenue Prediction

  is not a full sentence and must not be used to start a section

Balaji Dhamodharan, bdhamodh@iu.edu, fa20-523-337
Anantha Janakiraman, ajanakir@iu.edu, fa20-523-351

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

As mentioned in the earlier sections, the dataset used in this model exploration was downloaded from Kaggle [^1] and available in CSV file format. The training contains more than 872K observations and based on the size of the dataset it would be ideal to use mini-batch or gradient descent optimization techniques to identify the coefficients that best describe the model. The target variable as observed in the dataset is a continuous variable which implies that the use case is a regression problem. As mentioned earlier, there are several machine learning techniques that can be explored for this type of problem including regression and ensemble methods with different parameter settings. The sparsity of potential features in the datasets indicates that multiple experimentations will be required to determine the best performing model. Also based on initial review of the datasets, it also observed that some of the categorical features exhibit low to medium cardinality and if these features are going to be retained in the final dataset used for training then it is important to choose the right encoding technique. 

- Train.csv User transactions from August 1st, 2016 to August 1st, 2017 [^2]
- Test.csv User transactions from August 2nd, 2017 to April 30th, 2018  [^2]

### 2.1 Metrics

The metrics used for evaluation in this analysis is the root mean squared error (RMSE). The root mean squared error function forms the objective/cost function which will be minimized to estimate optimal parameters for the linear function using Gradient Descent. The plan is to conduct multiple experiments with different iterations to obtain convergence and try different hyper-parameters (e.g. learning rate).

RMSE is defined as:

![Figure 1.1](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/loss.png)

**Figure 1:** RMSE

where y-hat is the natural log of the predicted revenue for a customer and y is the natural log of the actual summed revenue value plus one as seen in Figure-1.

### 2.2 Source Code

Follow this [link](https://github.com/cybertraining-dsc/fa20-523-337/blob/main/project/code/project.ipynb) to the source code for subsequent sections in this report.


## 3. Methodology

The CRISP-DM process methodology was followed in this project. The high-level implementation steps are shown in Figure-2.

![Methodology](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/methodology.png)

**Figure 2:** Project Methodology

### 3.1 Load Data

The data that was obtained from Kaggle was over 2.6 GB (for Train and Test). As the size of the dataset was significantly large, it was hosted onto a storage bucket in Google Cloud Platform and ingested into the modeling process through standard application API libraries. Also in this project from the available datasets, the Train dataset was used to build the models and test the results because the real end goal is to test the effectiveness of algorithm. Since the test set doesn't contain the Target Variable (rightly so!), it will not be consumed during the testing and evaluation phase in this exploration.	

### 3.2 Data Exploration

The dataset obtained for this project is large and it contains over 872k records. The dataset also contains 12 predictor variables and 1 target variable. The target Variable is totals.transactionRevenue and the objective of this exploration is to predict the total transaction revenue of an online store customer as accurately as possible.

**Target Variable:** The Target Variable is totals.transactionRevenue has the transaction value of each visit. But, this column contains 98.72% of missing values for revenue (no purchase). The Target variable had a skewed distribution originally and after performing a lognormal distribution on the target variable it has a normal distribution.

![Target Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/y_after_transformation.png) 

**Figure 3:** Target Variable

#### 3.2.1 Exploratory Data Analysis

**Browser:** The most popular browser is Google Chrome. Also, it was observed during the analysis that second and third best users were using safari and firefox respectively.
		
![Browser Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_browser.png)		
				
**Figure 4:** Browser Variable
		
**Device Category:** Almost 70% of users were accessing online store via desktop
		
![Device Category Variable](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_category.png)		
		
**Figure 5:** Device Category Variable
		
**OperatingSystem:** Windows is the popular operating system among the desktop users. However, among the mobile users, what's interesting is, almost equal number of ios users (slightly lower) as android users accessed google play store. The reason why this is interesting is because, google play store is primarily used by android users and ios users almost always use Apple Store for downloading apps to their mobile devices. So it is interesting to know, almost equal number of ios users visit google store as well.
		
![OperatingSystem](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/device_operating_system.png)
			
**Figure 6:** Operating System	
		
**GeoNetwork-City:** Mountain View, California tops the cities list for the users who accessed online store. However in the top 10 cities, 4 cities are from California. 
		
![GeoNetwork-City](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_city.png)
				
**Figure 7:** GeoNetwork City	
		
**GeoNetwork-Country:** Customers from US are way ahead of other customers from different countries. May be this could be due to the fact that online store data that was provided was pulled from US Google Play Store (Possible BIAS!).
		
![GeoNetwork-Country](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_country.png)		
	
**Figure 8:** GeoNetwork Country
		
**GeoNetwork-Region:** It is already known that majority of the customers are from US, so America region tops the list.
		
![GeoNetwork-Region](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_continent.png)
		
**Figure 9:** GeoNetwork Region
		
**GeoNetwork-Metro:** SFO tops the list for all metro cities, followed by New York and then London.
		
![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/geo_network_metro.png)	
		
**Figure 10:** GeoNetwork Region
		
**Ad Sources:** Google Merchandise and Google Online Store are the top sources where the traffic is coming from to the Online Store. 
		
![GeoNetwork-Metro](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/traffic_source_adcontent.png)	
		
**Figure 11:** Ad Sources
	
### 3.3 Data Pre-Processing

Data Pre-Processing is an important step to build a Machine Learning Model. The data pre-processing step typically consists of data cleaning, transformation, standardization and feature selection, so only the most cleaner and accurate data is fed to a model. The dataset that was downloaded for this project contains several issues with formatting, lot of missing values, less to no variance (zero variance) in some of features and it was also observed the target variable does not have random distribution.  The variables such as totals_newVisits, totals_bounces, trafficSource_adwordsClickInfo_page, trafficSource_isTrueDirect, totals_bounces, totals_newVisits had missing values. The missing values were imputed with zeroes, so the machine learning algorithm is able to execute without errors and there are no issues during categorical to numerical encoding. This is a very important step in building the Machine Learning Pipeline. 

### 3.4 Feature Engineering

Feature Engineering is the process of extracting the hidden signals/features, so the model can use these features to increase the predictive power. This step is the fundamental difference between a good model and a bad model. Also, there is no one-size-fits all approach for Feature Engineering. It is extremely time consuming and requires a lot of domain knowledge as well. 
For this project, a set of sophisticated functions to extract date related values such as Month, Year, Data, Weekday, WeekofYear have been created. It was observed that browser and operating systems are redundant features and instead of removing them, they were merged to create a combined feature which will potentially increase the predictive power. Also as part of feature engineering several features were derived like mean, sum and count of pageviews and hits that should help increase the feature space and ultimately reduce the total error increasing the overall accuracy of the model.

### 3.4 Feature Selection

Feature Selection refers to selection of features in your data that would improve your machine learning model. There is subtle variation between Feature Selection and Feature Engineering. The Feature Engineering technique is designed to extract more feature from the dataset and the feature selection technique allows only relevant features into the dataset. Also, how does anyone know what are the relevant features? There are several methodologies and techniques that are developed over the years but there is no one-size-fits-all methodology. 

Feature Selection like Feature Engineering is more of an art than science. There are several iterative procedure that uses Information Gain, Entropy and Correlation scores to decide which feature gets into the model. There are also advanced Deep learning models that can be built or tree based models that can help observe variables of high importance after the model is built. Similar to Feature Engineering, Feature Selection should also require domain specific knowledge to develop festure selection strategies.

In this project, the features that had constant variance in the data were dropped and also the features that had mostly null values with only one Non-null value were dropped too. These features do not possess any statistical significance and add very less value to the modeling process. Also, depending on the final result, different techniques and strategies can be explored to optimize and improve the performance of the model.

#### 3.4.1 Data Preparation

Scikit learn has inbuilt libraries to handle Train/Test Split as part the model_selection package. The dataset was split randomly with 80% Training and 20% Testing datasets. 

### 3.5 Model Algorithms and Optimization Methods

Different Machine Learning algorithms and techniques were explored in this project and the outcome of the exploration along with different parameter settings have been discussed in the following sections.

#### 3.5.1 Linear Regression Model

Linear regression is a supervised learning model that follows a linear approach in that it assumes a linear relationship between one ore more predictor variables (x) and a single target or response variable (y). The target variable can be calculated as a linear combination of the predictors or in other words the target is the calculated by weighted sum of the inputs and bias where the weighted is estimated through different optimization techniques. Linear regression is referred to as simple linear regression when there is only one predictor variable involved and referred to as multiple linear regression when there is more than one predictor variable is involved. The error between the predicted output and the ground truth is generally calculated using RMSE (root mean squared error). This is one of the classic modeling techniques that will be explored in this project because the target variable (revenue per customer) is a real valued continuous output and exhibits a significant linear relationship with the independent variables or the input variables [^10].

In the exploration, SKLearn Linear Regression performed well overall. A 5 fold cross validation was performed and the best RMSE Score for this model observed was: 1.89. As shown in the Figure-12, the training and test RMSE error values are very close indicating that there is no overfitting the data.
	
![Linear Regression](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/linear_regression3.png)	
	
**Figure 12:** Linear Regression Model 
	
#### 3.5.2 XGBoost Regressor

XGBoost regression is a gradient boosting regression technique and one of the popular gradient boosting frameworks that exists today. It follows the ensemble principle where a collection of weak learners improves the prediction accuracy. The prediction in the current step S is weighed based on the outcomes from the previous step S-1. Weak learning is slightly better than random learning and that is one of the key strengths of gradient boosting technique. The XGBoost algorithm was explored for this project for several reasons including it offers built-in regularization that helps avoid overfitting, it can handle missing values effectively and it also does cross validation automatically. The feature space for the dataset being used is sparse and believe the potential to overfit the data is high which is one of the primary reasons for exploring XGBoost [^7][^8][^9].

XGBoost Regressor performed very well. It was the best performing model with the lowest RMSE score of 1.619. Also the training and test scores are reasonably close and it doesn't look like there was the problem of over fitting the training data. Multiple training iterations of this model were explored with different parameters and most of the iterations resulted in significant error reduction compared to the other model making it the best performing model overall. 
		
![XGBoost](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/xgboost3new.png)
	
**Figure 13:** XGBoost Model 	
	
#### 3.5.3 LightGBM Regressor	

LightGBM is a popular gradient boosting framework similar to XGBoost and is gaining popularity in the recent days. The important difference between lightGBM and other gradient boosting frameworks is that LightGBM grows the tree vertically or in other words it grows the tree leafwise compared to other frameworks where the trees grow horizontally. In this project the lightGBM framework was experimented primarily because this framework works well on large dataset with more than 10K observations. The algorithm also has a high throughput while using reasonably less memory but there is one problem with overfitting the data which was controlled in our exploration using appropriate hyper parameter setting and achieved optimized performance [^6][^11].

LightGBM Regression was the second best performing model in terms of RMSE scores. Also the training and test scores observed were different indicating a potential problem of overfitting. Again, multiple training iterations of this model were explored with different parameter settings and although it achieved reasonable error reduction compared to most of the other models that were explored, it still did not outperform the XGBoost regressor making it the second best performing model.
	
![lightgbm](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lighgbm3new.png)
	
**Figure 14:** LightGBM Model 
	
#### 3.5.4 Lasso Regression

Lasso is a regression technique that uses L1 regularization. In statistics, lasso regression is a method to do automatic variable selection and regularization to improve prediction accuracy and performance of the statistical model. Lasso regression by nature makes the coefficient for some of the variables zero meaning these variables are automatically eliminated from the modeling process. The L1 regularization parameter helps control overfitting and will need to be explored for a range of values for the specific problem. When the regularization penalty tends to be zero there is no regularization, and the loss function is mostly influenced by the squared loss and in contrary if the regularization penalty tends to be closer to infinity then the objective function is mostly influenced by the regularization part. It is always ideal to explore a range of values for the regularization penalty to improve the accuracy and avoid overfitting [^4][^5].

In this project, Lasso is one of the important techniques that was explored primarily because the problem being solved is a regression problem and there is possibility to overfit the data due to the number of observations and feature space. During the model training phase different ranges for regularization penalty were explored and the appropriate value that helped achieve maximum reduction in the total RMSE score was identified.

Lasso performed a bit better than baseline model. However, it did not outperform lightGBM or XGBoost or tree based models in general.
	
![lasso](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/lasso3new.png)
	
**Figure 15:** Lasso Model 

#### 3.5.5 Ridge Regressor

Ridge is a regression technique that uses L2 regularization. Ridge regression does not offer automatic variable selection in the sense that it is not make the weights zero on any of the variable used in the model and the regularization term in a ridge regression is slightly different than the lasso regression. The regularization term is the sum of the square of coefficients multiplied by the penalty whereas in lasso it is the sum of the absolute value of the coefficients. The regularization term is a gentle trade-off between fitting the model and overfitting the model and like in lasso it helps improve prediction accuracy as well as performance of the statistical model. The L2 regularization parameter helps control overfitting and will need to be explored for a range of values for the specific problem. The regularization parameter also helps reduce multicollinearity in the model. Similar to Lasso, when the regularization penalty tends to be zero there is no regularization, and the loss function is mostly influenced by the squared loss and in contrary if the regularization penalty tends to be closer to infinity then the objective function is mostly influenced by the regularization part. It is always ideal to explore a range of values for the regularization penalty to improve the accuracy and avoid overfitting [^4][^5].

In this project, Ridge regression is one of the important techniques that was explored again primarily because the problem being solved is a regression problem and there is possibility to overfit the data due to the number of observations and feature space. During the model training phase different ranges for regularization penalty were explored and the appropriate value that helped achieve maximum reduction in the total RMSE score was identified. Ridge performed a bit better than baseline model. However like Lasso, it did not outperform lightGBM or XGBoost or tree based models in general.
	
![Ridge](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/ridge3new.png)
	
**Figure 16:** Ridge Model 

## 4. Benchmark Results

There are some interesting observations from the benchmark results seen in Figure-17. As expected, the data exploration and pre-processing performed very well and the data load and flattening of JSON took only a few hundred milliseconds on a platform like Google Colab compared to more than a minute running the same code locally on a desktop. The grid search for linear regression as expected took more time than the grid search for regularization techniques which is an interesting finding. The model training phase using ridge regression took only half the time approximately 250 seconds compared to 811 seconds for lasso regression. The highest training time of approximately 1800 seconds was recorded with XGBoost regressor and although the original assumption was that this modeling technique would consume time and significant system resources to complete the training process, the total time of 1800 seconds was certainly in the higher end. But, considering the fact that random forest regressor took more than 90 minutes to complete the training process during the experimentation phase, XGBoost performed much better than random forest. The other interesting observation was between LightGBM and XGBoost where LightGBM took significantly less time than XGBoost regressor and if performance and high availability are key considerations during operationalization with slight compromise on model performance then LightGBM would be an ideal candidate for real time operationalization.

![Benchmark](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/benchmark1.png)

**Figure 17:** Benchmark Results 

## 5. Software Technologies

In this project tools like Python and Google Colab Jupyter Notebook were used. Also several Python packages were employed in this exploration such as Pandas, Numpy, Matplotlib, sklearn

## 6. Conclusion

As a project team the intention was to create a template that can be utilized for any ML project. The dataset that was used for this project was challenging in a way that it required a lot of data cleaning, flattening and transformation to get the data into the required format. 

### 6.1 Model Pipeline

In this project, multiple regression and tree based models from scikit learn library were explored with various hyper parameter setting and other methods like the lightGBM. The goal of the model pipeline was to explore and examine data, identify data pre-processing methods, imputation strategies, derive features and try different feature extraction methods was to perform different experiments with different parameter setting and identify the optimized with low RMSE that can be operationalized in Production. The parameters were explored within the boundary of this problem setting using different techniques.

### 6.2 Feature Exploration and Pre-Processing

As part of this project a few features were engineered and included in the training dataset. The feature importance visualizations that were generated after the model training process indicate that these engineered features were part of the top 30% of high impact features and they contributed reasonably to improving the overall accuracy of the model. During additional experimentation phase, the possibility of including few other potential features that could be derived from the dataset was explored and those additional features were included in the final dataset that was used during model training. Although these features did not contribute largely to reducing the error it gave an opportunity to share ideas and methods to develop these new features. Also during feature exploration phase other imputation strategies were evaluated, attempted to identify more outliers and tried different encoding techniques for categorical variables and ultimately determined that label encoder or ordinal encoder is the best way forward. Also some of the low importance features were excluded and the model was retrained to validate if the same or better RMSE value could be achieved.

### 6.3 Outcome of Experiments

Multiple modeling techniques were explored as part of this project like Linear regression, gradient boosting algorithms and linear regression regularization techniques. The techniques were explored with basic parameter setting and based on the outcome of those experiments, the hyper parameters were tuned using grid search to obtain the best estimator evaluated on RMSE. Also, during grid search K-Fold cross validation of training data was used and the cross validated results were examined through a results table. The fit_intercept flag played a significant role resulting in an optimal error. As part of the different experimentations that were performed, random forest algorithm was also explored but it suffered performance issues and it seemed like it would require more iterations to converge which is why it was dropped from our results and further exploration. Although random forest was not explored, gradient boosting techniques were part of the experimentations and the best RMSE from XGBoost. The LightGBM regressor was also explored with different parameter settings but it did not produce better RMSE score than XGBoost. 

In the case of XGBoost, there was improvement to the RMSE score as different tree depths, feature fraction, learning rate, number of children, bagging fraction, sub-sample were explored. There was significant improvement to the error metric when these parameters were adjusted in an intuitive way. Also, linear regression with regularization techniques were explored and although there was some improvement to the error metric compared to the basic linear regression model they did not perform better than the gradient boosting method that was explored. So, based on different explorations and experimentations a reasonably conclusion can be made that gradient boosting technique performed better for the given problem setting and generated the best RMSE score. Based on the evaluation results of XGBoost on the dataset used, the recommendation would be to test the XGBoost model with real time data and the performance of the model can be evaluated in real-time scenario too and additionally, if needed, hyper parameter tuning can be performed on the XGBoost model specifically for the real-time scenario [^9]. The feature engineering process on the dataset helped derive features with additional predictive value and a pipeline was built to reuse the same process in different modeling techniques. Five different models were tested including Linear Regression, XGBoost, Light GBM, Lasso and Ridge. The summary of all the models can be seen in Figure-17.

![Model_Results](https://github.com/cybertraining-dsc/fa20-523-337/raw/main/project/images/model_results_new2new.png)

**Figure 18:** Model Results Summary 

### 6.4 Limitations

Due to the limited capacity of our Colab Notebook setup, there was difficulty in performing cross Validation for XGBoost and LightGBM. The KFold cross validation with different parameter settings would have helped identify the best estimator for these models, helped achieve even better rmse scores and potentially avoid overfitting, if any. The tree based models performed well in this dataset and it would be beneficial to explore other tree based models like Random Forest in the future and evaluate/compare the performance.

## 7. Previous Explorations

The Online GStore customer revenue prediction problem is a Kaggle competition with more than 4100 entries. It is one of the popular challenges in Kaggle with a prize money of $45,000. Although the goal was not to make it to the top in the leader board, the challenge gave a huge opportunity to explore different methods, techniques, tools and resources. The one important difference between many of the previous of explorations versus what has been achieved in this exploration is the number of different machine learning algorithms that was explored and the performance for each of those different techniques were examined. Based on review of several submissions in Kaggle there were only a very few kernel entries that explored different parameter settings and making intuitive adjustments to them to make the model perform at an optimum level like what has been accomplished in this project. The other uniqueness that was brought to this submission was identifying techniques that offered good performance and consumed less system resources in terms of operationalization. There is lot of scope to continue exploration and attempt other techniques to identify the best performing model.

## 8. Acknowlegements

The team would like to thank Dr. Gregor Von Laszewski, Dr. Geoffrey Fox, and the other instructors in the Big Data Applications course for their guidance and support through the course of this project and advise on documenting the results of various explorations. 


## 9. References

[^1]: Kaggle Competition,2019,Predict the Online Store Revenue,[online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/rules>

[^2]: Kaggle Competition,2019,Predict the Online Store Revenue, Data, [online] Available at: <https://www.kaggle.com/c/ga-customer-revenue-prediction/data>

[^3]: Towards DataScience,2020,Sweetviz- Powerful EDA,[online] Available at: <https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34>

[^4]: Towards Datascience,2018,Bhattacharya, Ridge and Lasso regression, [online] Available at: <https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b>

[^5]: Datacamp,2019,Oleszak, Regularization: Ridge, Lasso and Elastic Net, [online] Available at: <https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net>

[^6]: Medium 2017,Mandot, What is LightGBM,[online] Available at: <https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc>

[^7]: XGBoost 2020,xgboost developers, XGBoost, [online] Available at: <https://xgboost.readthedocs.io>

[^8]: Datacamp,2019,Pathak, Using XGBoost in Python, [online] Available at: <https://www.datacamp.com/community/tutorials/xgboost-in-python>

[^9]: Towards Datascience,2017,Lutins, Ensemble Methods in Machine Learning, [online] Available at: <https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f>

[^10]: Machine Learning Mastery,2016,Brownlee, Linear Regression Model, [online] Available at: <https://machinelearningmastery.com/linear-regression-for-machine-learning>

[^11]: Kaggle 2018,Daniel, Google Analytics Customer Revenue Prediction, [online] Available at: <https://www.kaggle.com/fabiendaniel/lgbm-starter>















