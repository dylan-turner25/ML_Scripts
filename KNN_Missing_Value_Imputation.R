# clear console
rm(list=ls())

# load libraries
library(class) #contains knn function

#create a function that normalizes the values to between 0 and 1 (normalization is critical for KNN !!!)
normalize <- function(x){
  x.norm <- (x - min(x))/(max(x) - min(x))
  return(x.norm)
}

# omits observations with missin values in the specified columns
completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

# load data: outcome variable should be in the first column of the data frame
df <- data.frame(mp.purch, rental, retreat, wealth.share, knew.zone,armor,
                 college,e.first.aid,e.ia, e.loan, e.pa, employed.ft, female,sentiment)

# make sure outcome variable is numeric
df[,1] <- as.numeric(df[,1])

# identify the outcome variable
outcome.var <- df[,1]

# identify which columns the predictor variables are in 
predictor.vars <- c(2:ncol(df))

# remove any observations that are missing for the predictor variables
df <- completeFun(df,predictor.vars)

# normalize the predictor variables
for( i in predictor.vars){
  df[,i] <- normalize(df[,i])
}

# extract observations that have missing values for the outcome variables
# these are the observations that we need to impute values for
missing.rows <- which(is.na(df[,1]) == T) # row locations for missing outcome values
df.missing <- df[c(missing.rows),]

# extract observations that are not missing for the outcome variable 
# this is the complete data we will use for training the KNN algorithm
df.complete <- df[-c(missing.rows),]

# split the complete data into two stratified samples 
set.seed(123456) # set seed for reproducability 
split.ratio <- .8 # define what percentage of the data you want to be used for training
stratifiedSplit <- stratified(df.complete, df.complete[,1] , split.ratio, keep.rownames = F, bothSets = T)

# extract the training split
train <- stratifiedSplit$SAMP1[,-1] # remove the outcome variable from the training set
outcome_train <- as.vector(as.data.frame(stratifiedSplit$SAMP1[,1])[,1])#  extract the outcome variable for the train data

# extract the test split
test <- stratifiedSplit$SAMP2[,-1] # remove the outcome variable from the test set
outcome_test <- as.vector(as.data.frame(stratifiedSplit$SAMP2[,1])[,1])#  extract the outcome variable for the train data

# run knn function
predictions <- knn(train = train, test = test ,cl = outcome_train, k = 10)

# create a confusion matrix
confusion <- table(predictions,outcome_test)
confusion

# calculate the out of sample accuracy, if it is satisfactorily high, 
# then proceed to impute missing data, otherwise go back and refine model
accuracy <- summary(as.numeric(predictions == outcome_test))[4]
print(paste("Out of Sample Accuracy = ", round(accuracy*100,2),"%",sep = ""))

# predict missing values for the missing data
predictions.missing <- knn(train = train, test = df.missing[,2:ncol(df.missing)] ,cl = outcome_train, k = 10)

# add predictions to the data frame that had missing values in the outcome
df.missing[,1] <- predictions.missing

# merge missing and complete data frames back together
df.imputed <- rbind(df.missing, df.complete) 
