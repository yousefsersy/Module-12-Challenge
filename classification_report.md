# Credit Risk Analysis Report

## Table of Contents: 

1. [Overview of the Analysis](#overview-of-the-analysis)

2. [Model Results](#results)

3. [Summary](#summary)

</br>

## **Overview of the Analysis**

Lending companies provide funds or properties to borrowers with the expectation of repayment or asset return. Credit Risk arises when borrowers fail to meet these expectations, resulting in financial losses for the lender. Various methods exist to measure credit risk, and in this analysis, we use Machine Learning to evaluate a dataset from a peer-to-peer lending service, aiming to develop a model that can assess borrowers' creditworthiness.

</br>

* Our objective is to use a machine learning model to differentiate between healthy (low-risk) and non-healthy (high-risk) loans based on the loan status data provided by the lending company.

  * The Logistic Regression Algorithm is chosen for this task due to its widespread use in predicting the probability of target variables in classification problems.

</br>

* Using the provided dataset, I created a Logistic Regression Model, which achieved an accuracy score of 95%. Despite this high accuracy, the model's recall value for non-healthy loans (0.91) was lower than for healthy loans (0.99). This disparity suggests the model is better at predicting healthy loans than non-healthy ones, likely due to the imbalanced nature of the dataset, where healthy loans significantly outnumber non-healthy loans.

</br>

`Analyzing the data in step 3 [Split the Data into Training and Testing Sets], we see the imbalance using the value_counts function:`

```
# code
y.value_counts()

# output
0    75036
1     2500
Name: loan_status, dtype: int64
```

`According to the confusion matrix in step 3 [Create a LRM w/ Original Imbalanced Data]:`

* Out of 18,765 healthy loan statuses, the model correctly predicted 18,663 and incorrectly predicted 102.

* Out of 619 non-healthy loan statuses, the model correctly predicted 563 and incorrectly predicted 56.

</br>

`To improve accuracy and enhance the model's ability to classify non-healthy loans, we employed oversampling using the RandomOverSampler module from the imbalanced-learn library, creating a balanced dataset by increasing the minority class (non-healthy loans).`

```
# code
y_oversampled.value_counts()

# output
0    56271
1    56271
Name: loan_status, dtype: int64
```

  * Using the balanced dataset, I developed a Logistic Regression Model which achieved an accuracy score of 99%, surpassing the imbalanced model. The oversampled model showed improved performance due to the balanced dataset, with the recall for non-healthy loans rising from 0.91 to 0.99, indicating it significantly reduces misclassifications of non-healthy loans as healthy.

`According to the confusion matrix in step 3 [Create a LRM w/ Resampled(oversampled) Data]:`

* Out of 18,765 healthy loan statuses, the model correctly predicted 18,649 and incorrectly predicted 116.

* Out of 619 non-healthy loan statuses, the model correctly predicted 615 and incorrectly predicted 4.

</br>

## **Results**

</br>

### Logistic Regression Model fitted with Imbalanced Data: 

</br>

`The Logistic Regression model trained on the imbalanced dataset correctly predicted healthy loans 100% of the time and non-healthy loans 85% of the time.`

</br>

* The model trained on imbalanced data is more likely to make these errors:

  * Classifying a healthy loan (low-risk) as non-healthy (high-risk).

  * Classifying a non-healthy loan (high-risk) as healthy (low-risk).


</br>

`The recall scores indicate that the model made 1% errors when predicting healthy loans and 9% errors when predicting non-healthy loans.`

`Although the model achieved a 95% accuracy score, it could be improved due to dataset imbalance.`


</br>

### Logistic Regression Model fitted with Balanced (oversampled) Data:

</br>

`The Logistic Regression model trained on the oversampled dataset correctly predicted healthy loans 100% of the time and non-healthy loans 99% of the time.`

</br>

* The model trained on balanced data has a much lower likelihood of these errors:

  * Classifying a healthy loan (low-risk) as non-healthy (high-risk).
  * Classifying a non-healthy loan (high-risk) as healthy (low-risk).
</br>

`The recall scores show that the model made 1% errors when predicting both healthy and non-healthy loans.`

`The model achieved a 99% accuracy score due to the balanced dataset.`


</br>

## **Summary**

* A lending company would benefit from a model that accurately classifies healthy and non-healthy loans to minimize risks:

  * Misclassifying healthy loans as non-healthy can result in customer loss.

  * Misclassifying non-healthy loans as healthy can lead to financial losses for the lender.

`The Logistic Regression model trained on oversampled data outperformed the one trained on imbalanced data, achieving higher accuracy and recall scores, thus reducing errors in classifying non-healthy loans.`

</br>

`Lending companies prefer fewer False Positives to avoid misclassifying non-healthy loans as healthy, reducing financial risks. The confusion matrices below illustrate the model's performance in correctly/incorrectly predicting loan statuses:`

* Model with Imbalanced Data: 
  
  * 56 (FALSE POSITIVES) --> The actual value is healthy, predicted as non-healthy.


  * 102 (FALSE NEGATIVES) --> The actual value is non-healthy, predicted as healthy.

</br>

* Model fitted with Balanced Data: 
  
  * 4 (FALSE POSITIVES) --> The actual value is healthy, predicted as non-healthy.


  * 116 (FALSE NEGATIVES) --> The actual value is healthy, predicted as non-healthy.
  
`The confusion matrices show a significant reduction in False Positives with the balanced model, suggesting it is more reliable in classifying both healthy and non-healthy loans. Based on this analysis, I recommend using the Logistic Regression Model trained on balanced (oversampled) data.`

---