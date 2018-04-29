<style type="text/css">
body{ /* Normal  */
      font-size: 12px;
  }
td {  /* Table  */
  font-size: 12px;
}
h1.title {
  font-size: 38px;
  color: DarkRed;
}
h1 { /* Header 1 */
  font-size: 28px;
  color: DarkBlue;
}
h2 { /* Header 2 */
    font-size: 22px;
  color: DarkBlue;
}
h3 { /* Header 3 */
  font-size: 18px;
  font-family: "Times New Roman", Times, serif;
  color: DarkBlue;
}
code.r{ /* Code block */
    font-size: 12px;
}
pre { /* Code block - determines code spacing between lines */
    font-size: 14px;
}
</style>
``` r
knitr::opts_chunk$set(fig.width=12, fig.height = 8, warning = FALSE, message = FALSE)
require(caret); require(ggplot2);
list_h2=0
```

Introduction
------------

This is a final report of the course projet from Coursera course **Pratice Machine Learning**, as part of **Data Science Specialization**.<br/>
The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. Those Six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.
<br/> More information of the project background is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).
</br> The report contains:<br/> \* Question
\* Getting input data
\* how to select featurew
\* how the model has been built
\* how to do cross validation
\* what the expected out of sample error is
\* use the final prediction model to evaluate 20 different test cases

Question
--------

To use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

Getting input data
------------------

The data for this project were download from following site and stored on the **data** folder under the project:<br/> \* The training data (`pml.train`): <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>.
\* The test data (`pml.test`): <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>.

### Create partition

I will allocate 60% of data from `pml.train` dataset as training set for building model and the rest 40% as testing set for preforming cross validation. The `pml.test` dataset will be used on final prediction.

``` r
    ## create partition
    set.seed(3583)
    inTrain  <- createDataPartition(pml.train$classe, p = 0.6)[[1]]
    training <- pml.train[inTrain,]
    testing  <- pml.train[-inTrain,]
```

Selecting variables (features)
------------------------------

### Removing un-useful variables

The first 7 variables are adminstrative variables amd not useful for prediction (X, user\_name, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, cvtd\_timestamp, new\_window, num\_window), these variables are removed from data set.

``` r
    ## remove the first 7 variables
    training <- training[,-(1:7)]; testing<- testing[,-(1:7)]; predicting  <- pml.test[,-(1:7)]
```

### Removing Near-zero variance variables

There are many dummy variables or constant variables in dataset, We can easily remove those variables from dataset using `nearZeroVar` command.

``` r
    ## identify near zero variance variables
    y_var <- ncol(training)
    isnzv <- nearZeroVar(training[,-y_var], saveMetrics = T)
    training<-training[,names(training)[!isnzv$nzv]]
    testing<-testing[,names(testing)[!isnzv$nzv]]
    predicting<-predicting[,names(predicting)[!isnzv$nzv]]
```

### Removing valiables are mostly NA

For those varibles that has large amout of measurment (say 97.5%) were missing. These variables also are removed.

``` r
    ## identify variables that has 95% NA observation
    nearna <- sapply(training, function(x) mean(is.na(x))) > 0.975
    trainingF<-training[,!nearna]
    testingF<-testing[,names(testing)[!nearna]]
    predictingF<-predicting[,names(predicting)[!nearna]]
```

After removed adminstrative variables, near zero variance variables, and missing measurment variables following variables will be used on predition.

    ##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
    ##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
    ##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
    ## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
    ## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
    ## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
    ## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
    ## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
    ## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
    ## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
    ## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
    ## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
    ## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
    ## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
    ## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
    ## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    ## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    ## [52] "magnet_forearm_z"     "classe"

Exploratory data analysis
-------------------------

### Correlation Analysis

The correlation among variables as above is analyzed before fitting model. I calculate the correlation between all predictor variables. And I take its absolute value.

``` r
    require(corrplot)
    y_var <- ncol(trainingF)
    corMatrix<-abs(cor(trainingF[,-y_var], use='pair'))
    diag(corMatrix) <- 0
    corrplot(corMatrix)
```

![](PML_CourseProject_files/figure-markdown_github/analysis-1.png)

The highly correlated variables are shown dark color in above plot. You can see the detail of high correlation variables on **Appendix** section.

Building Model
--------------

In the project I will use 3 training method to build model:
\* Random forests method
\* Decision tree
\* Boosting

### Random forests

First of all I am using **Random forests** method to build model. Because random forests method has problem in `caret packate`, I use the method from `randomForest package` directly.

``` r
    require(randomForest)
    set.seed(3593)
    trainingF$classe <- as.factor(trainingF$classe)
    modFit_rdf <- randomForest(classe~ .,data=trainingF, importance=FALSE)
```

### Decision tree

Next I am trying to use **Decision tree** method to fit the model.

``` r
    require(rattle)
    set.seed(3593)
    modFit_rpart <- train(classe~ .,data=trainingF, method="rpart", trControl=control_rf)
```

### Boosting

The third I am using **Boosting with trees** method to fit the model.

``` r
    set.seed(3593)
    modFit_gbm <- train(classe~ .,data=trainingF, method="gbm", trControl=control_rf, verbose=FALSE)
```

### Evaluation of models

    ##             Accuracy     Kappa AccuracyLower AccuracyUpper AccuracyNull
    ## eval_rdf   0.9946470 0.9932284     0.9927711     0.9961394    0.2844762
    ## eval_gbm   0.9602345 0.9496940     0.9556727     0.9644503    0.2844762
    ## eval_rpart 0.4956666 0.3411313     0.4845431     0.5067933    0.2844762
    ##            AccuracyPValue McnemarPValue
    ## eval_rdf                0           NaN
    ## eval_gbm                0  9.529942e-09
    ## eval_rpart              0           NaN

After applied `testing` set to models for cross validation the **Random forests** model achieved accuracy of **99.46%**, the **Boosting with trees** model yield accuracy of **96.02%**, and the **Decision tree** model has the worst accuracy of **49.57%**.

### Out of sample error

    ##                Out-of-Sample Error
    ## oosError_rdf           0.005353046
    ## oosError_gbm           0.039765486
    ## oosError_rpart         0.504333418

From the above, the **Random forests** model is the best model with **0.54%** out-of-sample error rate, and then the **Boosting with trees** model with **3.98%** out-of-sample error rate. Again, the **Decision tree** model yield the highest out-of-sample error **50.43%** rate.

Result and Predicting
---------------------

The **Random forests** model is our final chooice for predicting. I applied it to the 20 predicting set and the result are shown as below.

``` r
    predict(modFit_rdf, newdata=predictingF)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Appendix
--------

### High correlation variables

``` r
    cor80<-which(corMatrix > 0.8, arr.ind=T); print(cor80)
```

    ##                  row col
    ## yaw_belt           3   1
    ## total_accel_belt   4   1
    ## accel_belt_y       9   1
    ## accel_belt_z      10   1
    ## accel_belt_x       8   2
    ## magnet_belt_x     11   2
    ## roll_belt          1   3
    ## roll_belt          1   4
    ## accel_belt_y       9   4
    ## accel_belt_z      10   4
    ## pitch_belt         2   8
    ## magnet_belt_x     11   8
    ## roll_belt          1   9
    ## total_accel_belt   4   9
    ## accel_belt_z      10   9
    ## roll_belt          1  10
    ## total_accel_belt   4  10
    ## accel_belt_y       9  10
    ## pitch_belt         2  11
    ## accel_belt_x       8  11
    ## gyros_arm_y       19  18
    ## gyros_arm_x       18  19
    ## magnet_arm_x      24  21
    ## accel_arm_x       21  24
    ## magnet_arm_z      26  25
    ## magnet_arm_y      25  26
    ## accel_dumbbell_x  34  28
    ## accel_dumbbell_z  36  29
    ## pitch_dumbbell    28  34
    ## yaw_dumbbell      29  36

``` r
    qplot(x=trainingF[,cor80[1,1]],y=trainingF[,cor80[1,2]], color=trainingF$classe,
          xlab=names(trainingF)[cor80[1,1]], ylab=names(trainingF)[cor80[1,2]])
```

![](PML_CourseProject_files/figure-markdown_github/analysis2-1.png)

### fancyRpartPlot of decision tree

``` r
    require(rattle)
    fancyRpartPlot(modFit_rpart$finalModel)
```

![](PML_CourseProject_files/figure-markdown_github/fit_rpart2-1.png)

R Markdown
----------

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.
