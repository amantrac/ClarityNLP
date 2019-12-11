# I - Conceptual Questions
## Would you rather have too many false positives, or too many false negatives? Explain.
<font size="3">The question relates also to the exercice part (as we see further). In a typicall system taking decision in production usually one cares about its precision, i.e. among the items classified as positive how many are correctly classified. The number of false positives would measure how many are wrongly, in other words more false positive we have less the precision of the system. On the other hands, the real positives are either correctly classified by the system or wrongly in which case they are false negatives. Therefore, too many false negatives means the system is recovering few of the items it could. As the precision of the sytem increases, as it gets more strict about its decision to assign positive labels, and as its recall decreases. Depending on the system one may give more importance to precision, or recall. In practise, the business unit has to define based on the porduct at hand the level of requirement. In the remainder, we will see that to put the categorizer in production we suggest a precision level of  99%. That's of course hypothetical and based on our assumptions of how the categorizer will be used in production.</font>

## Give an example from your work experience of selection bias. Was it important? In general, how can you avoid it?
<font size="3">Selection bias is of course crucial as it may introduce a difference between offline metrics as typically measured on a test set, and online metrics when running an A/B test. This can typically happen when you have for example a recommender system running in production such that sampling directly from the data observed in production leads to a bias of selection imposed by the recommender system. Imagine that the recommender system is flowed, and never suggest any item from class A, this may lead if we are not carefull to produce a training data set without any item from class A since its never displayed by the system in production.

In the context of this exercice for instance, we are given  a training set, and a test set but nothing is said about how this data is selected. As we will see further, the class distributions are very different between training and test and indicates a selection procedure especially on the training (a stratified sampling as indicated by round numbers). It's very likely that the person who produced the training data sampled each category in a specific way to ensure all categories are present in the training set at sufficent level (this is typicall in ML), even though the real observed distribution differ (it may be that the test set follow the real distribution, but we have not enough information to be assertive on this). 

To avoid selection bias, one have to use different selection strategy, as stratified sampling based on specific features as category, or CTR, etc. The idea would be to cover all cases that may lead to a bias. Another technique would consist of introducing randomization in the production system. For instance, one way could be to sample the recommendation based on the probability score and so always give the chance to not recommended items to still show up. </font>

## What is the difference between Bayesian Estimate and Maximum Likelihood Estimation (MLE)? Can you think of an example where Bayesian estimate is the most appropriate method? Another example in which MLE is the most appropriate method?</font>
<font size="3">in MLE one would usually directly maximize the likelihood of a class, or its conditional likelihood in order to infer a model (values of parameters). A typical example is when training a logistic regression model that consists of finding the parameter values that maximize the conditional likelihood of the data. This is very appropriate when we have enough data, such that each feature is correctly estimated. However, when we do not have enough data  (leading to the parameters estimation to lack confidence) one would rather use a bayesian approach. In that context, we usually have a believe (for instance on parameter values) that come in form of a prior distribution. By so doing, one can take decision based on an evidence that is marginalized on the full distribution of the parameter set and hence account better for the variance of the estimates. A simple way to see this is that if you consider a normal prior, as one starts having more and more data, confidence increases and leads to nearly a null variance, in which case an MLE is great. However, for the lack of data case, the prior believe will smoothen the decision to avoid a wrong estimation based on few observations. In practise, a Bayesian model can be very effective for the long tail and rare events, while the MLE would work fine on the head (which if we are facing a power law as is usually the case will cover 80 to 90% of the domain).</font>

## II - Conceptual Problem
<font size="3"> We are given the average performance of 2 stores in $, along with the stand. deviation and the number of points of which both measure are calculated. If we make the hypothesis, that the performance follow a normal distribution we can easily derive a confidence interval to compare both scores with a 95\% confidence. </font>

```python
import numpy as np
import scipy.stats


def confidence_interval(m, std, n, confidence=0.95):
    ci_1, ci_2 = scipy.stats.norm.interval(confidence, loc=0, scale=1)
    h = (ci_1*std/(n**(0.5)))
    return m-h, m, m+h
    
first_store = confidence_interval(800000,100000,50)
print(first_store)
second_store= confidence_interval(780000,30000,50)
print(second_store)

```
```
(827718.0764869936, 800000, 772281.9235130064)
(788315.422946098, 780000, 771684.577053902)
```
We firt notice, that both confidence intervals overlap therefore we can not conclude that one performs better. We confirm this as well with a simple t-test, where we try to reject the null hypothesis of mean equalities of the population

```python
import numpy as np
from scipy import stats

def stat_test_diff_normal(mean_a , std_a, mean_b, std_b, N ):
    var_a = std_a**2
    var_b = std_b**2
    N = 50
    s = np.sqrt((var_a + var_b)/2)
    ## Calculate the t-statistics
    t = (mean_a - mean_b) /(s*np.sqrt(2/N))
    ## Compare with the critical t-value
    #Degrees of freedom
    df = 2*N - 2
    #p-value after comparison with the t 
    p = 1 - stats.t.cdf(t,df=df)
    return 2*p 

print(f" p-value = {stat_test_diff_normal(800000,100000,780000,30000, 50)}")
```
```
p-value = 0.17866908114208235
```

# III. Practical Problem
<font size="3">
It is asked to design a classification pipeline in the context of a business decision for classifying similar new articles. We start by analyzing the existing data set, before discussing assumptions, and derive solutions that we implement for the problem
</font>

# Business Understanding - What is the problem
## Questions one should answer to design a Text Classifier in production

<font size="3">The important question one should ask in order to design a text classifier system to be put in production is:
</font>

### (a) What is the scale of training set:

<font size="3"> When a system is put in production one has to keep in mind that the pipeline should be fully automatized, and the training should be done each time additional annotated data are collected and injected in the system. In our problem however, we are facing a dataset for which very likely the annotations are done by human editors. That means that the speed of annotations will remain small and hence we should expect a small scale in terms of number of training samples --- some millions of samples maximum. 
</font>

### (b) How often should the system be (re)trained and how:

<font size="3"> As explained earlier, since we are dealing with human annotations the pace of retraining will be dictated by the rythm at which new annotations comes in. We can easely make the assumption that we will get new annotations to be added to production once a week. In other word, we could envisage heavy learning systems that will take up to 1 week for training (
</font>

### (c) Life-cycle of the training and test set:

<font size="3"> In the real world, the training and test set will be evolving, mostly to keep injecting new examples not covered 
well so far. To do so, one will generally select news articles by relying on metrics as their popularity and make sure that the most popular articles are perfectly well covered, and check performance specifically on the part of the traffic that matters. Many times, metrics are not well designed, or do not reflect the reality of the traffic that matters. On top of that, the performance should be analyzed on the torso, and tail of the category distribution as the system keep improving.
During that cycle, false positive, and false negatives should be discussed along with the annotation team, and additional example should be extracted (using keyword search for instance) that resemble to the false positives and false negatives and used to improve the training set, and test set.
</font>

### (d) Serving at runtime or Offline:


<font size="3"> This question is crucial, as very likely if the model will be used online for serving the maximum time allowed for a score prediction for all classes is a maximum of few ms. This is not necessarly the case if the automatic annotations is done by an offline job, in which case we can consider  heavy models. In the remainder, we consider that the categorization are done by an offline job that populates a database, and would consider issues related to serving a model at runtime out of the scope of this assignement, however we remain open to discuss them if needed.</font>

    
### (e) Infra-cost vs Accuracy in the context of Business metrics
<font size="3">For instance, we could imagine a hadoop spark job that will run an estimator on each news feed into the system, in this last case our constraints are mostly how many news do we need to categorize, and what is the (infra)-cost we are ready to incur for it. This comes typically with a trade-off between computing time (and infra-cost) versus accuracy. So, one would need to run experiments offline to analyze the win in performance of running heavy models (for instance BERT), and the extent to which the final business metrics benefit of it.
</font>

# Business Understanding - What is available / What does a solution look like

<font size="3">Having highlighted  the main aspect of problems, let's see what is given to us. Basically, we are provided with two json files, rouglhy small (65000 samples for the training, 30K of the test, 10 Categories). What we propose is to first start with a simple baseline relying on Bag-Of-Words modeling, using a logistic regression. We propose to go through the exercice using this first modeling, and iteratively see what drives the performance (in terms of the features), and then tunning the model. We wil then propose additional steps to go beyond a simple logistic and see what are the pros and cons of such a choice.</font>


# Technology choices

<font size="3">For data preparation we decide to use pandas as it is a fast way to load the data and do cleaning operation on it, as check for Nans values, be sure the intersection between Training and Test is void, having a look at the category distribution. Also, since the dataset is pretty small this can be done directly on the laptop itself. Also, pandas interface well wit numpy, scikit, and keras that we will use a machine learning tools for this project. They allow us to easily train and test models as logistic regression, and neural networks, and also do hyperparameter tunning using internal cross validation. They will also provide facilities for plotting, and computing metrics (as accuracy for example).</font>



# Data Preparation

## Load data set. 

```python
import pandas as pd
import os
%matplotlib inline
import matplotlib.pyplot as plt
datafilename_train = "News_category_train.json"
datafilename_test = "News_category_test.json"
path_to_file = "../datasets/"
data_train = pd.read_json(os.path.join(path_to_file,datafilename_train))
data_test = pd.read_json(os.path.join(path_to_file,datafilename_test))
```
## check if any column of the dataset  is missing
```python
print(f"Is there any missing value in the training data set: {data_train.isnull().values.any()}")
print(f"Is there any missing value in the test data set: {data_test.isnull().values.any()}")
```
```
Is there any missing value in the training data set: False
Is there any missing value in the test data set: False
```
## Check the distribution of the classes in Training and Test
```python
data_train.category.value_counts().plot(kind='bar')
```
![Training categories distribution](images/distribution-train-categories.png)

```python
data_test.category.value_counts().plot(kind='bar')
```
![Test categories distribution](images/distribution-test-categories.png)
