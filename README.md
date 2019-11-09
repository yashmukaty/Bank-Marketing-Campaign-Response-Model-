# Bank-Marketing-Campaign-Response-Model-
This example uses data related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to assess if the product (bank term deposit) would be subscribed ('yes') or not ('no').
Bank client data:
age (numeric)
job: type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
marital: marital status (categorical: 'divorced', 'married', 'single', 'unknown'; note: 'divorced' means divorced or widowed)
education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
default: has credit in default? (categorical: 'no', 'yes', 'unknown')
housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
Related with the last contact of the current campaign:
contact: contact communication type (categorical: 'cellular', 'telephone')
month: last contact month of year (categorical: 'jan', 'feb', 'mar', ., 'nov', 'dec')
day_of_week: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')
duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
other attributes:
campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
previous: number of contacts performed before this campaign and for this client (numeric)
poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
social and economic context attributes
emp.var.rate: employment variation rate - quarterly indicator (numeric)
cons.price.idx: consumer price index - monthly indicator (numeric)
cons.conf.idx: consumer confidence index - monthly indicator (numeric)
euribor3m: euribor 3 month rate - daily indicator (numeric)
nr.employed: number of employees - quarterly indicator (numeric)
Output variable (desired target):
y - has the client subscribed a term deposit? (binary: "yes","no")
Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques.

https://archive.ics.uci.edu/ml/datasets/Bank+Marketing


NOTES ABOUT RESPONSE MODELS:
Targeting the Right Prospects: What are Response Models?
Response models use data mining to find similarities between responders from previous marketing campaigns to predict who is likely or not likely to respond to a future campaign. The model is then scored against the prospects of the new campaign and a marketer can choose to mail only those people that are most likely to purchase. This increases conversions and decreases costs by only mailing to those most likely to respond.

Direct Marketing Models: Good, Better, Best
Not all models are created equal. Here’s a quick summary of different types of direct marketing models:

GOOD. Recency, Frequency, Monetary (RFM) models:
simple, better than not modeling

Though very basic, many marketers still rely on RFM models. Technically RFM models aren’t actually response models since they are descriptive but not truly predictive. This method emphasizes customer behavior and segments by how recently a customer purchased, how often they purchase, and how much they spend.

RFM can identify good customers and provide a lift in response rates versus not using any targeting. Another benefit is that it is both simple and descriptive, so it is easily understood by business people.

Unfortunately, RFM doesn’t take into account life stage and assumes that customers are likely to continue to respond the same way. If RFM is the only targeting method, the most attractive segments are likely to be over-marketed to at the expense of other segments that could be invested in.

BETTER. Traditional Response or Regression Models:
more sophisticated and predictive than RFM

Regression models determine the correlation between variables. Unlike RFM models, regression takes into account that scores can quickly change when combined with other variables.

The model is developed specifically to predict a desired behavior, such as response. Response models require both responder and non-responder data to identify patterns of those likely to respond to a marketing campaign.

This is by far the most widely used approach for marketers and has been a mainstay of predictive analytics for decades.

BEST. Multi-Channel Customer Level Response Models:
A New Approach that Outperforms Traditional

This innovative approach identifies not only those prospects most likely to purchase, but also which marketing channel they are most likely to respond to. This allows marketers to optimize their marketing budgets most effectively by contacting the prospect in the channel(s) they prefer and are most likely to be moved by.

Multi-Channel Customer Level Response Models are different from traditional response models in that all of a prospect’s known activity is taken into account – email opens, web browsing, display ad click-throughs, mobile, purchase behavior – and not just direct mail behavior. With a more holistic view of the customer, a marketer can create the ideal customer contact strategy for each customer.

In a recent head-to-head in the mail test with a major cataloger, the Multi-Channel Customer Level Response Model outperformed the traditional response model with a more than +14% lift in response rate. This increase translates into millions of dollars in new annual revenue at the current spend. A new subject line or pretty picture won’t drive revenue like that!

Model Building Steps - Pre-Modeling Stage
Identification of Business Problem & Converting business problem into stats problem
- Diagnostics of data to identify the business problem or test the hypothesis
- Understand the pain points to address
- Regression vs. classification vs. Segmentation vs. Forecasting vs. Optimziaiton vs. Others
- Strategic vs. Operational
- Supervised vs. Unsupervised
- What kind of combination of problems
Understand the data availability
- Define Y & X variables
- Understand sources of data
- Understand Granularity of data availability
- Understand enterprise data model
Identify the right Technique for given problem
- Regression Problem (Available Algorithms)
    - Linear Regression (Traditional)
    - Lasso/Ridge/Elastic net Regression
    - Decision Tree Regressor
    - Bagging Regressor
    - Random Forest Regressor
    - Adaboost Regressor
    - GBM Regressor
    - XGBoost Regressor
    - KNN Regressor
    - Support Vector Regressor (Linear SVM, Kernal SVM)
    - ANN Regressor
- Classification Problem (Available Algorithms)
    - Logistic Regression (Traditional)
    - Decision Tree Classifier
    - Bagging Classifier
    - Random Forest Classifier
    - Adaboost Classifier
    - GBM Classifier
    - XGBoost Classifier
    - KNN Classifier
    - Support Vector Classifier (Linear SVM, Kernal SVM)
    - ANN Classifier
- Segmentation Problems (Available Algorithms)
    - Hueristic Approach (Value Based, Life stage, Loyalty, RFM Segmentation)
    - Scientific Approach 
        - Subjective - Kmeans/Kmedians/Hierarchical/DBSCAN
        - Objective - Decision Trees
- Forecasting Problems (Available Algorithms)
    - Univariate Time Series
        - Decomposition
        - Averages (MA, WMA)            
        - ETS Models (Holt Winters)
        - SARIMA (AR/MA/ARMA/ARIMA/SARIMA)
    - Multivariate Time Series
        - SARIMAX (ARIMAX/SARIMAX)
        - Regression
Acquiring Data from available data sources (Internal/External/Paid sources)
- Internal databases
- External databases (Credit Bueruo's, Experian, D&B, Axiom etc.)
- Social Media & Secondary research
Data Audit
- At File Level
    - Is the data sample/population?
        - If it is sample, do we have population metrics to compare the sample metrics with population metrics
        - Size of the sample
    - Type of Format (delimiters)
    - Missings 
    - Number of rows
    - Number of columns
    - File Size
    - Headers availability
    - Data dictionary availability & understand the tables & keys
    - Keys (Unique variables) or which variables making each observation as unique
    - How the tables are interelated 
    - is data encoded? or Masked?
 - At Variable Level
     - Data types - mismatch    
     - Categorical    Nomial/    ordinal
     - Which of the variables date variables? Can we create any Derived variables?
     - Does variables need renaming?    
     - Does data have Missing values?    
     - Does data have Outliers?    
     - Does data have Duplicate records    
     - Does Variables with multiple values - do we need to split/extract specific information from the variable?    
     - Does data have Special values - 0's, @, ?, #NA, #N/A, #Error, Currencies, Null values, -inf, inf, 99999    
     - Understand Distribution of data    
            - Percentile distribution
            - Histogram/Boxplot for continuous
            - Bar chart for categorical
     - Understand Relationships (correlations/associations)    
     - Perform detailed Exploratory analysis?    
     - Create Derived variables to define using data (KPI's)    
     - sample representing population or not?    
     - Identify variables with Near Zero variance?    
     - Identify variables required to do encoding (Label encoding/One hot encoding)
Model Building Steps - Modeling Stage
Data Preparation-1 (Process data based on data audit Report)
Handiling all the problems identififed in the data audit report

Converting data types into appropriate manner
Renaming variables as required (specially remove spaces, special characters from headers)
Handlign missing values (Imputation of Missing values)

Cross Sectional data
Numerical - Impute with mean/median/Regression/KNN/MICE approach
Categorical - Impute with mode/Regression/KNN/MICE approach
Time Series data
Impute with forward filling/backward filling/moving average/centered moving average
Handling Outliers

Cross sectional data
Capping & floring with upper cap/lower cap
P1, P99 or P5, P95
mean+/-3std
Q1-1.5IQR, Q3+1.5IQR
Time Series Data
Correct the peaks/slumps based on business understanding
Add dummy variable for that period of time if you are unable to address the reasons
MA (Yearly data) +/- 2.5/3*STd(with in Year)
Converting categorical variables into numercal variables

- Ordinal variables (Label encoding)
- Nominal Variables (One-Hot encoding)
Creating derived variables (Using KPI's/Using Date variabels/Split the existing ariables etc)

Data Preparation-2 (Assumptions of the techniques)
    - Linear Regression (Normality/Linearity/No outliers/No Multicollinieirty/Homoscedasticity)
    - Logistic Regression (Linearity/No outliers/No Multicollinieirty)
    - KMeans/KNN (data should be scaled/standardized/No outliers)
    - Forecasting (no missings)
    etc.
Data Preparation-3 (Feature selection/Feature engineering/Feature identification)
- Based on the data, you may drop variables based on below reasons
    - If the variable have lots of missings (>25%)
    - if categorical Variable with lots categories(>20)
    - Variable with near zero variance (CV<0.05)
    - Unique variables/Keys/Names/Emaiid's/Phone number
    - Using Business Logic (By keeping in implementation perspective
- Variable Selection/Reduction using relationships
    - Supervised Learning (Based on Y & X Relationships)
        - Using statistical methods
        - Univariate Regression (F- Regression)
        - RFE (Recursive Feature Elimination)
        - SelectKBest
        - DT/RF
        - Regularization
        - WOE (relationship between Y & Log(odds) - Binary Classification
    - Any technique (Based on relationship with in X's variables)
        - Correlation metrics
        - PCA/SVD
        - VIF (Variance Inflation factor >5)
Data Preparation-4
- Splitting the data into train & Test
    - Cross Sectional data - Random sample of split (70:30)
    - Time Series Data - Based on time index (Most recent data will be test)       
Modeling steps - Building Model
Build the models Using the techniques what we selected as part of problem solving

Linear Regression
- variable significance
- final mathematical equation
- Important Drivers (positive/negative)
- Check the driver's signs with orignal correlations with Y
Metrics
R-square/Adj Square
MSE/RMSE/MAPE/RMSPE
Corr(Actual, pred)
Decile Analysis (Rank Ordering)
Checks
Errors should follow normal distribution
Corr(Actual, Errors) ~0
no multicollineirity on final list of variables
Logistic Regression
- variable significance
- final mathematical equation
- Drivers (Positive/Negative)
Metrics

Metrics Based on the probability

AUC, Somerce D/Gini
Get the right cut-off

%1's in the data
Maximum of Senstivity+ Specicity
Maximum of accuracy
Decile Analysis - Using KS Value
Metrics Based on category as output (after cut-off)

Confusion metrics
Precision
Recall - sensitivity, specicity, accuracy
F1-score
Segmentation
Iterative process different values of K
Identifying segmentation is good or not
Best value of K
Based SC Score/Pseduo F-value
Based on Segment Distribution
Based on Profiling
Identifying segment characteristics from profiling
Forecasting
Decomposition - Irregular compoent should be low (close to 1 for multiplicative decomposition)
ARIMA
Identifying best values of (p,d,q)(P,D,Q)
using get_prediction() method, calculating training accuracy
MAPE/MSE/RMSE/RMSPE
Machine Learning
Find the right hyperparameters to tune using GridsearchCV
Perform Gridsearch
Identify best model with best parameters
Finalize the model
Metrics are same as similar to linear regression/logistic regression
Relative importance of variables
Not possible to identify significance of variables
Not possible to identify the drivers's signs
Model Validation
Supervised Learning

Score the data using finalized model on train data
Predicting values/forecasting values for test data
Calculate all the metrics related to different types of techniques on validation data
Check the problems related to Overfitting/Underfitting
Finalize model based on low overfitting & low underfitting
Unsupervised Learning (Segmentation)

Predict the segments for test data using centroids
Business validation by comparing charateristics
Calculate SC score for test data, compare it with train data's SC Score
Modeling Steps - Post Modeling
Deployment code /Implementation code

- Preprocessing code as similar to preprocessing steps followed in train data
- Apply the model to predict the values for new data
Deployment of application/WebGUI/Excel Documentation

Presentation/Word Document/Excel Document
Business Problem
Which technique considered
Which data you considered?
Definitions of new variables
What transformation applied & why?
What imputations done & why?
Train & test split data proportions
Model metrics for all the models you tried for train & test
Which model finalized & why?
Final list of variables considered
Checks you have done it?
Detailed audit report
Iterations & how the improvement?
Assumptions of data
Waterfall chart of variable reduction
Finalize the mathematical equation
Mathematical relation ship - Y = B1X1+B2X2+B3*X3+C
Model object (Pickle)
Pros & Cons of the model
Quantify the model output in terms KPI's
Model tracking system/Model Maintanance
