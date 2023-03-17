# Bank-Credit-Card-Churn-Analysis

## Overview
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
  
  #### Decision Tree
  Decision trees classify data by utilizing a tree structure that is built by segmenting the dataset into several subsets.
  The outcome is a tree containing leaf nodes and decision nodes. It has a tree-based structure to show the predictions
  that result from a series of feature-based splits which starts with a root node and ends with a decision made by leaves.
  Some terms used in the Decision tree:
    • Root Node: It is the starting point of the decision tree. The population starts to get split from this particular
    node based on the features.
    • Decision Node: The nodes we get after splitting the root node.
    • Leaf Nodes: The last node in the decision tree after which no further splitting of the node is possible.
    • Sub-tree: A part of the decision tree.
    • Pruning: Process of cutting down some nodes to stop overfitting.
  The Decision Tree Algorithm also tries to deal with the uncertainty in the dataset using ENTROPY. A Pure sub-split
  is the split in which the entropy of the subset becomes zero indicating that features are perfectly separated.
  
  <b>Formula of Entropy:</b>
  
  −p+ve.log(p+ve) − p−ve.log(pve)
  
  Entropy reveals a node’s impurity, but information gain, which chooses the feature for the root node, can be used to
  assess how an impurity changed after splitting.
  Information Gain = E(Parent) - E(Parent|Decisive feature)
  E(Parent)−→The entropy of the Parent Node.
  E(Parent|Decisive Feature)−→The weighted average of Entropies of each decisive node split out of a particular feature.
  The feature which has the largest information gain after the split is picked for root node
  In real-world data sets, there are a lot more features and it takes a lot of time to process the Decision tree algorithm
  as the tree gets more and more complicated. One of the drawbacks of the Decision tree algorithm is, this algorithm
  won’t stop until the entropy reaches 0 and tries to fit to each and every data point, even the noise in the data set.
  Thus, it leads to overfitting of the model on the training data set.
  
  #### Random Forest
  A Random Forest is a bagging ensemble learning technique which incorporates two fundamental ideas that give it the
  term random rather than just average the predictions of trees:
    • A Random sampling of training observations:
    In a random forest, each tree gains knowledge from a random selection of the training observations. As a result
    of the samples being drawn using replacement, or bootstrapping, certain samples will be utilized more than once
    in a single tree. The concept is that by training each tree on many samples, even though each tree may have a
    large variance relative to a specific set of training data, the overall variance of the forest will be reduced without
    increasing the bias.
    • Random Subsets of features for splitting nodes:
    The random forest’s second fundamental idea is that, while deciding whether to divide a node, each tree considers
    only a portion of all the features. This may be done in Sklearn by setting max features = p
    (n features) which means that if a node in a tree has 16 characteristics, only 4 random features will be taken into account when
    splitting the node.
    A Random Forest achieves low variance and low bias by merging predictions from multiple decision trees into a single
    model, resulting in forecasts that are more accurate on average than predictions from individual decision trees.
    
  #### Light GBM
  Gradient Boosting Decision Tree has a variant called Light GBM that is very effective. In terms of computation
  speed and memory use, it can perform noticeably better than XGBoost and SGB. Gradient boosted decision trees are
  implemented using the Light GBM framework. Because of Light GBM’s many benefits, including its quicker training
  speed, good accuracy with default parameters, parallel and GPU learning, small memory footprint, and capacity to
  handle huge datasets, we chose to employ it. In Light GBM, the train() technique is used to generate an estimator. It
  accepts dictionary and training dataset as estimator parameter inputs. The estimator is then trained, and a returned
  object of type Booster has a trained estimator that can be used to forecast the future.


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
