Practical Machine Learning Project 
=========

# Project Summary

The goal of your project is to predict the manner in which they did the exercise which is that accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.


```r
require(dplyr)
```

```
## Loading required package: dplyr
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(randomForest)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(1234)
```

## Processing data cleaned

Data in training and testing files loaded.

- mpl_training is the training data set
- mpl_testing is the test set


```r
mpl_training <- read.csv("/Users/truongpham/Documents/Coursera/8 Practical Machine Learning/data/pml-training.csv")
mpl_testing <- read.csv("/Users/truongpham/Documents/Coursera/8 Practical Machine Learning/data/pml-testing.csv")
```

Some varaiables is near to zero which make them useless in prediction, so they are removed from our data.


```r
zerodata <- nearZeroVar(mpl_training[,-160])
mpl_training_cleaned <- mpl_training[, -zerodata]
mpl_testing_cleaned <- mpl_testing[, -zerodata]
```

Some missing data shouldn't be included in predicted data. Therefore, they should be removed too.


```r
nas <- colSums(is.na(mpl_training_cleaned)) > 0
mpl_training_cleaned <- mpl_training_cleaned[, !nas]
mpl_testing_cleaned <- mpl_testing_cleaned[, !nas]
```

The first six variables in data looks unuseful in prediction, so they are deleted in dataset. They are X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, and num_window.


```r
mpl_training_cleaned <- mpl_training_cleaned[, 7:59]
mpl_testing_cleaned <- mpl_testing_cleaned[, 7:59]
```

Now, prediction is conducted.

## Cross-validation

Training dataset is divided into two parts of 70% and 30% for validationg.


```r
in_train <- createDataPartition(mpl_training_cleaned$classe, p = 0.7, list=FALSE)
training <- mpl_training_cleaned[in_train, ]
testing <- mpl_training_cleaned[-in_train, ]
```

## Prediction

Cross-validation uses four partions with method of random forest.



```r
fit <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method = "cv", number = 4))
test_pred <- predict(fit, newdata=testing)
```

Cross-tabulation will be used for comparision of observed and predicted classes.



```r
confusionMatrix(test_pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674   11    0    0    0
##          B    0 1127    4    0    1
##          C    0    1 1018    6    1
##          D    0    0    4  957    3
##          E    0    0    0    1 1077
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.989    0.992    0.993    0.995
## Specificity             0.997    0.999    0.998    0.999    1.000
## Pos Pred Value          0.993    0.996    0.992    0.993    0.999
## Neg Pred Value          1.000    0.997    0.998    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.163    0.183
## Detection Prevalence    0.286    0.192    0.174    0.164    0.183
## Balanced Accuracy       0.999    0.994    0.995    0.996    0.998
```

The model accuracy is 99.46%, hence the error of ~ 0.54%. 

Prediction of test is followed as below.


```r
actual_predict <- predict(fit, newdata= mpl_testing_cleaned)
```

Ouput will be streamlized into file.


```r
# function for writing the output files
write_file = function(x){
  n = length(x)
  for(i in 1:n){
    fname = paste0("/Users/truongpham/Documents/Coursera/8 Practical Machine Learning/data/problem_id_",i,".txt")
    write.table(x[i],file=fname,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


write_file(actual_predict)
```

There are 20 files created for prediction of 20 sample tests.
