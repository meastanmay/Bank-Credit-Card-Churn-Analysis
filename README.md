# Bank-Credit-Card-Churn-Analysis

## Table of contents:


With the advance of digital technology, people are increasingly resorting to credit cards online for their transactions.
In this competitive market space, it is paramount for banks to retain customers to maintain the profit
margin. We aim to predict credit card customer churn using machine learning models to deal with customer churn
problems. We have applied four models including Logistic regression, Decision tree, Random Forest, and Light
GBM to our dataset which contains more than 10000 pieces and 20 features.

### Introduction
Given the fierce market competition, credit cards are a crucial part of a bank’s profit. Customers are therefore crucial
from a business perspective. As a result, customer turnover is a key area of attention for many banks. If we take a
look at the AARRR or HEART frameworks utilized by several institutions, it is accurate. According to studies, a
bank’s profits might rise by 85% when the retention rate goes up 5%. The purpose of this study is to forecast customer
churn. Once forecasted, banks would have enough time to take proactive steps to keep clients by providing better
services or more alluring discounts. Therefore, it is very important, particularly in today’s world where there is a
wealth of customer-related data, and with the widespread usage of big data, large users’ data have become priceless
jewels for businesses. Large volumes of data may be processed and analyzed using machine learning. While using
models to predict the outcome, some other articles primarily concentrate on unsupervised learning, which is generally
unreliable and has relatively low interpretability. In this study, we use Kaggle, which has over 10,000 data points and
21 different attributes, to get credit card holders’ information. To determine its distribution and display correlations
between attributes, we perform exploratory data analysis. The dataset was then divided into training and testing, and
standardization came next. To evaluate the performance of the models, Logistic Regression, Decision Tree, Random
Forest, and Light GBM are employed.

### Methods
  #### Logistic regression
  The method of modeling the likelihood of a discrete result given an input variable is known as logistic regression. The
  most common algorithm models binary outcome by classifying a sample to the class if the estimated probability is
  greater than 50%. The probability estimated by the model in vector form is given by:
  
  p = hθx = σxTθ
  
  Interestingly, because it uses a non-linear log transformation of the linear regression, logistic regression can handle
  non-linear correlations between the dependent and independent variables. A Logistic function or a logistic curve is a
  common s-shaped curve (sigmoid curve)

### Conclusion
From the above results we could conclude that:
1. Light GBM model trained on the under-sampled training data set gives the best performance.
2. The most important features to determine the Attrition Rate among the customers of the bank are:
  (a) Total Transaction Amount
  (b) Total Transaction Count
  (c) Total Amount Change Q4 to Q1
  (d) Total Count Change Q4 to Q1
  (e) Total Revolving Balance
  (f) Average Utilization Ratio
  (g) Total Relationship Count
3. All the above features are negatively correlated to the target feature Attrition Flag, that is, the higher the values
of the above features, the lower the chances of the customers getting churned.
4. The bank must engage with their customers more frequently and increase the relationship count with each
customer.
5. Bank must come up with some offers or policies where they could decrease the inactivity of the Existing customers
and promote the usage of their credit card services.

### Future Scope
1. The attrition rate is an extremely popular and widely researched problem statement in the industry. Each
business in the industry wants to retain their customers and reduce Customer Acquisition costs to increase their
profit margin.

2. In our project, we used only 4 different Machine Learning algorithms to predict the customers’ Attrition rate
namely: Logistic regression, Decision Tree Classifier, Random Forest Classifier, and Light GBM Classifiers. In
the industry other classification algorithms like Neural network algorithms, genetic programming approaches
using the AdaBoost model, and many other models have been utilized.
3. We were provided with the data set for a few customers to train our model on; it would be more accurate and
generalized if we supplied more data to the training model.
4. We can use more sophisticated ensembling techniques by combining two or more base classifiers.

### Author’s Conrtribution
  • Tanmay Agarwal: Exploratory Data Analysis, Data Preparation, Modelling, Conclusion and Future Scope.
  • Niveditha Channapatna Raju: Literature survey, Exploratory Data Analysis, Plagarism Check and Report
  Writing on Latex.
  • Sajid Hussain: Research Publications for understanding different models which are used in this project, Evaluation
  criteria, Report Drafts.

### References
[1] Gustafsson, A., Johnson, M.D., Roos, I.: The effects of customer satisfaction, relationship commitment dimensions,
and triggers on customer retention. Journal of Marketing 69(4), 210–218 (2005)
[2] Roberts, J.H.: Developing New Rules for New Markets. Journal of the Academy of Mar- keting Science 28(1),
31–44 (2000)
[3] Slater, S.F., Narver, J.C.: Intelligence Generation and Superior Customer Value. Journal of the Academy of
Marketing Science 28(1), 120–127 (2000)
[4] Kotler, P.: Marketing Management. Prentice-Hall, NJ (2000)
[5] Lu, J.: Predicting Customer Churn in the Telecommunications Industry -— An Application of Survival Analysis
Modeling Using SAS. Sprint Communications Company
[6] Glady, N., Baesens, B., Croux, C.: Modeling Churn Using Customer Lifetime Value. European Journal of
Operational Research (2008), doi:10.1016/j.ejor.2008.06.027
[7] Van den Poel, D., Larivi‘ere, B.: Customer attrition analysis for financial services using proportional hazard
models. European Journal of Operational Research 157(1), 196–217 (2004)
[8] Buckinx, W., Van den Poel, D.: Customer base analysis: Partial defection of behaviorally-loyal clients in a
non-contractual fmcg retail setting. European Journal of Operational Re-search 164(1), 252–268 (2005)
[9] Glady, N., Baesens, B., Croux, C.: Modeling Churn Using Customer Lifetime Value. European Journal of
Operational Research (2008), doi:10.1016/j.ejor.2008.06.027
[10] Neslin, S.A., Gupta, S., et al.: Defection Detection: Improving Predictive Accuracy of Customer Churn Models
(2004)
[11] G. L. Nie, W. Rowe, L. L. Zhang, Y. J. Tian, and Y.Shi, “Credit card churn forecasting by logistic regression
and decision tree,” Expert Systems with Applications, vol. 38, pp. 15273-15285, 2011.
[12] Q. F. Bi, K. E. Goodman, J. Kaminsky, and J. Lessler, “What is Machine Learning? A Primer for the
Epidemiologist,” American Journal of Epidemiology, vol. 188, pp. 2222-2239, October 2019.
[13] An Introduction to Logistic Regression Analysis and Reporting CHAO-YING JOANNE PENG KUK LIDA LEE
GARY M. INGERSOLL
