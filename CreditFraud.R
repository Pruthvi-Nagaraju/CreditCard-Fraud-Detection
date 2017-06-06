
#loading the libraries required
library(corrgram)
library(ggplot2)
library(caret)
library(ROCR)
library(e1071)
library(PRROC)
library(pROC)
library(randomForest)
library(DMwR)
library(unbalanced)
library(ROSE)
library(xgboost)

#reading the dataset
CreditData= read.csv("C:/SummerProjects/CreditCardFraud/creditcard.csv")
View(data)

# checking for missing values in each column
missing<-function(x){
  n=ncol(x)
  missing=0
  cname=1
  for(i in 1:n){
    missing[i]=sum(is.na(x[,i]))
    cname[i]=colnames(x)[i]
  }
  df=data.frame(cname,missing)
  return(df)
}

missing(CreditData)
# there are no missing values in any column

# changing target variable to factor
CreditData$Class= as.factor(CreditData$Class)

#checking for number of observations in each level for response variable
table(CreditData$Class)

#0          1 
#284315    492
#This shows the data is highly imbalanced

prop.table(table(CreditData$Class))

#0      1 
#0.9983 0.0017

#plotting the target variable
ggplot(CreditData,aes(x = CreditData$Class, fill="red")) + 
  geom_bar(position = "dodge", alpha = 0.5, col ="black") +
  scale_x_discrete( name = "Is it Fraud?") +
  scale_y_continuous() + 
  ggtitle("Fraud Case Classes") +
  theme(plot.title = element_text(hjust = 0.5))


#using createDataPartition from caret package to split the data (70% -train, 30% - test)

trainIndex=createDataPartition(CreditData$Class, p = 0.7,list = F,times = 1)

dataTrain= CreditData[trainIndex,]
dataTest= CreditData[-trainIndex,]
table(dataTrain$Class)

#0          1 
#199021    345 

table(dataTest$Class)
#0        1 
#85294   147 


# building a baseline logistic regression model

model1= glm(dataTrain$Class ~ ., family = binomial , data= dataTrain)
#predicting the model on test data
model1_predict= predict(model1, dataTest,type='response')
model1_predClass= ifelse( model1_predict >= 0.5, 1, 0)

table(model1_predClass,dataTest$Class)
caret::confusionMatrix(model1_predClass,dataTest$Class,positive='1')

#      Reference
#Prediction     0     1
#0           85276    54
#1             18    93

#Accuracy : 0.999
#Sensitivity : 0.632653        
#Specificity : 0.999789 
#Kappa : 0.7205


#plotting ROC curve
roc.curve(model1_predClass, dataTest$Class, curve=T)
#Area under the curve (AUC): 0.928

# using k fold cross validation technique
k=10
n=floor(nrow(CreditData)/k)
log_accuracy<-c()
#using 10-fold cross validation
for (i in 1:k){
  s1 = ((i-1)*n+1)
  s2 = (i*n)
  subset = s1:s2
  log_train<- CreditData[-subset,]
  log_test<- CreditData[subset,]
  log_fit<-glm(Class ~ ., family=binomial, data = log_train)
  log_pred <- predict(log_fit, log_test)
  log_pred_class <- ifelse(log_pred>0.5, 1, 0)
  print(paste("Logistic Accuracy: ",1-sum(log_test$Class!=log_pred_class)/nrow(log_test)))
  log_accuracy[i]<- 1- (sum(log_test$Class!=log_pred_class)/nrow(log_test))
}
#taking the mean of all the 10 model to estimate the accuracy of the model
print(paste("The accuracy of the logistic Model is: ",mean(log_accuracy)))
#The base model gives an accuracy of 0.999125702247191
roc.curve(log_pred_class, log_test$Class, curve=T)
#Area under the curve (AUC): 0.863


# Balancing the data using SMOTE 

# Creating SMOTE dataset with new train and test split


train_smote=SMOTE(Class~.,data=dataTrain, perc.over=2000,perc.under=100)
test_smote=SMOTE(Class~.,data=dataTest, perc.over=2000,perc.under=100)

train_smote$Class= as.factor(train_smote$Class)
test_smote$Class= as.factor(test_smote$Class)


# glm using balanced sample
model1glm= glm(Class ~ ., family = binomial , data= train_smote)
model1_predict= predict(model1glm, test_smote)
model1_predClass= ifelse( model1_predict >= 0.5, 1, 0)
caret::confusionMatrix(model1_predClass,test_smote$Class,positive='1')

roc.curve(model1_predClass, test_smote$Class, curve=T)
#Sensitivity : 0.866         
#Specificity : 0.988
#Kappa : 0.852
#Area under the curve (AUC): 0.931


#Random forest with balanced data

rf_model_smote = randomForest(Class~.,train_smote,ntree=500,mtry=2,sampsize=c(50,250),nodesize=1, rules = TRUE)
rf_test_predictions = predict(rf_model_smote, test_smote, type="class")
conf_matrix_rf = table(test_smote$Class,rf_test_predictions)
confmatrix =confusionMatrix(conf_matrix_rf,positive = '1')
randomforest_smote_roc = roc.curve(test_smote$Class,rf_test_predictions)
randomforest_smote_recall = confmatrix$byClass[1]

#Sensitivity : 0.930         
#Specificity : 0.922
#Kappa : 0.852
#Area under the curve (AUC): 0.924


#XGBoost with the balanced dataset

xgb.data.train <- xgb.DMatrix(as.matrix(train_smote[, colnames(train_smote) != "Class"]), label = as.numeric(levels(train_smote$Class))[train_smote$Class])
xgb.data.test <- xgb.DMatrix(as.matrix(test_smote[, colnames(test_smote) != "Class"]), label = as.numeric(levels(test_smote$Class))[test_smote$Class])

xgb.model <- xgb.train (data = xgb.data.train, params = list(objective = "binary:logistic",
                             eta = 0.1, max.depth = 3,min_child_weight = 100,
                             subsample = 1, colsample_bytree = 1, nthread = 3,
                             eval_metric = "auc"), nrounds = 500,early_stopping_rounds = 40,
                             watchlist = list(test = xgb.data.test))

xg_test_predictions = predict(xgb.model, data.matrix(test_smote[,colnames(test_smote) != "Class"]))
model1_predClass= ifelse( xg_test_predictions >= 0.5, 1, 0)
table(model1_predClass)
conf_matrix_rf = table(test_smote$Class,model1_predClass)
confmatrix =confusionMatrix(test_smote$Class,model1_predClass,positive = '1')

#Sensitivity : 0.979         
#Specificity : 0.885 
#Kappa : 0.857 

xgboost_smote_roc = roc.curve(test_smote$Class,xg_test_predictions)
#Area under the curve (AUC): 0.986

#XGBoost gives the best AUC of 0.986

