# Assignment 2 by Yuanyuan Qin (yqi43@uclive.ac.nz) on Aug.29

# Question 1 -- Linear regression and feature selection
# Load the library and open the data file
library(ggplot2)
library(leaps)
library(MASS)

load("Residen.RData")
attach(Residen)
# EDA of the data
names(Residen)
colnames(Residen)
dim(Residen)
# Understand the correlation
data <- Residen[, !(names(Residen) %in% c("V105"))]
dim(data)

# Explore the graphic tools to display the correlation between variables
#method1
library(corrplot)
corr_matrix <- cor(data)
dim(corr_matrix)
corrplot(corr_matrix, method = "color", tl.cex = 0.2, tl.col = "black")
#method2
library(graphics)
par(cex.main = 0.8)
heatmap(corr_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),  # Color palette
        breaks = seq(-1, 1, length.out = 101),  # Breaks for color scale
        main = "Correlation Heatmap",
        cexCol = 0.2, cexRow = 0.2)  # Title

#Build y data frame with V104
y <- Residen$V104
head(y)

# Build the x data frame without V105 and V104
x <- Residen[, !(names(Residen) %in% c("V104", "V105"))]
head(x)

# check the density distribution of y
ggplot(Residen, aes(y)) + geom_density(fill="blue")
ggplot(Residen, aes(log(y))) + geom_density(fill="blue")
ggplot(Residen, aes(sqrt(y))) + geom_density(fill="blue")

# Build model1 as linear regression model
model1 <- lm(y~., data=x)
summary(model1)

#model2 <- lm(log(y)~., data=x)
#summary(model1)
# No much improvement from y

set.seed(202302)
# Sample the dataset. Returns a list of row indices. 80:20 split.
row.number <- sample(1:nrow(data), 0.5*nrow(data))

# create the train and test data sets
train <- data[row.number,]
test <- data[-row.number,]

# show dimensions of the train and test sets.
dim(train)
dim(test)

# backwards
start_time <- Sys.time()
model2 <- stepAIC(model1, direction="backward")
end_time <- Sys.time()
backward_modeling_time <- end_time - start_time
backward_modeling_time

#stepwise selection
# first create null model.
model0 <- lm(V104~1, data=train)
summary(model0)

# Automatic forward search using AIC
start_time <- Sys.time()
model3 <- stepAIC(model0, direction="forward", scope=list(upper=model1, lower=model0))
end_time <- Sys.time()
stepwise_modeling_time <- end_time - start_time
stepwise_modeling_time

# model summary
summary(model3)

# handout mean square error of backward method
pred1 <- predict(model2, newdata = test)
head(pred1)
head(test$V104)
mse2 <- sum((pred1 - test$V104)^2)/length(test$V104)
mse2
c(MSE = mse2, R2 = summary(model2)$r.squared)

# handout mean square error of stepwise method
pred1 <- predict(model3, newdata = test)
head(pred1)
head(test$V104)
mse3 <- sum((pred1 - test$V104)^2)/length(test$V104)
mse3
c(MSE = mse3, R2 = summary(model3)$r.squared)

# K-fold cv error comparision of backward and stepwise

k <- 10
set.seed(202303)

# assign every row in Hitters a random fold number (1-10)
folds <- sample(1:k, nrow(data), replace=TRUE)

# a vector to hold MSE values for each fold.
cv.mse1 <- numeric(k)
cv.mse2 <- numeric(k)

for (j in 1:k) {
  train_data <- data[folds != j, ]  # K-1 is training sets
  test_data <- data[folds == j, ]   # one fold for test set
  
  # use K-1 folds to build the backward model
  model4 <- lm(V104 ~ ., data = train_data)
  model5 <- stepAIC(model1, direction = "backward")
  
  # Calculate the MES on the test fold
  pred <- predict(model5, newdata = test_data)
  cv.mse1[j] <- mean((test_data$V104 - pred)^2)
}

# Calculate the mean of MSE
mean_cv_mse_backward <- mean(cv.mse1)
mean_cv_mse_backward 


for (j in 1:k) {
  train_data <- data[folds != j, ]  # K-1 is training sets
  test_data <- data[folds == j, ]   # one fold for test set
  
  # use K-1 folds to build the backward model
  model6 <- lm(V104 ~ ., data = train_data)
  model7 <- lm(V104~1, data = train_data)
  model8 <- stepAIC(model7, direction="forward", scope=list(upper=model6, lower=model7))
  
  # Calculate the MES on the test fold
  pred <- predict(model8, newdata = test_data)
  cv.mse2[j] <- mean((test_data$V104 - pred)^2)
}

# Calculate the mean of MSE
mean_cv_mse_stepwise <- mean(cv.mse2)
mean_cv_mse_stepwise 

# Compare the two models
# Create a data frame to store the comparison results
comparison_df <- data.frame(
  Model = c("Backward Model", "Stepwise Model"),
  Computation_Time = c(backward_modeling_time, stepwise_modeling_time),
  Holdout_MSE = c(mse2, mse3),
  CV_MSE_k_10 = c(mean_cv_mse_backward, mean_cv_mse_stepwise),
  R_squared = c(summary(model2)$r.squared, summary(model3)$r.squared)
)

# Print the comparison data frame
print("Model Comparison:")
print(comparison_df)

# Ridge and Lasso regression plus lambda selection using cross-validation method
set.seed(202304)

# create matrix for glmnet, because this method cannot take dataframe for modelling 

X <- model.matrix(V104 ~ ., data)[,-1]
y <- data$V104
head(X)
dim(X)

train <- sample(1:nrow(X), nrow(X)/2)
test <- -train

# using default grid...Ridge
cv.out <- cv.glmnet(X[train,],
                    y[train],
                    alpha=0,
                    nfolds=10,
                    thresh=1e-12)

cv.out$lambda.min
plot(cv.out)

# Calculate the MSE based on the best lambda from CV
bestlam <- cv.out$lambda.min
ridge.pred <- predict(cv.out, s=bestlam, newx=X[test,])
mean((ridge.pred - y[test])^2)
# End of uing default grid

start_time <- Sys.time()
# use the self-defined grid
grid <- 10^seq(5, -2, length=100)
grid

ridge.cv.out <- cv.glmnet(X[train,],
                          y[train],
                          alpha=0,
                          lambda=grid,
                          nfolds=10,
                          thresh=1e-12)

best_model <- glmnet(X[train,], y[train], alpha=0, lambda=bestlam)
coef(best_model)
end_time <- Sys.time()
ridge_modeling_time <- end_time - start_time
ridge_modeling_time

ridge.cv.out$lambda.min
plot(ridge.cv.out)
cv_mse_mean1 <- mean(ridge.cv.out$cvm)
cv_mse_mean1

bestlam <- ridge.cv.out$lambda.min
ridge.pred <- predict(ridge.cv.out, s=bestlam, newx=X[test,])
test_MSE_Ridge <- mean((ridge.pred - y[test])^2)
test_MSE_Ridge

# Use Lasso Linear Regression

start_time <- Sys.time()

# use the self-defined grid
grid <- 10^seq(5, -2, length=100)
grid

lasso.cv.out <- cv.glmnet(X[train,],
                          y[train],
                          alpha=1,
                          lambda=grid,
                          nfolds=10,
                          thresh=1e-12)

best_model <- glmnet(X[train,], y[train], alpha=1, lambda=bestlam)
coef(best_model)
end_time <- Sys.time()
lasso_modeling_time <- end_time - start_time
lasso_modeling_time

lasso.cv.out$lambda.min
plot(lasso.cv.out)
cv_mse_mean2 <- mean(lasso.cv.out$cvm)
cv_mse_mean2

bestlam <- lasso.cv.out$lambda.min
lasso.pred <- predict(lasso.cv.out, s=bestlam, newx=X[test,])
test_MSE_Lasso <- mean((lasso.pred - y[test])^2)
test_MSE_Lasso

# Question 2 -- Lasso regression for small sample size and large feature variables
Parkinson <- read.csv("parkinsons.csv",header=TRUE)
head(Parkinson)

X <- model.matrix(UPDRS ~ ., Parkinson)[,-1]
y <- Parkinson$UPDRS
head(X)
dim(X)
head(y)

set.seed(202305)
train_size <- 30
train <- sample(1:nrow(X), train_size)
test <- setdiff(1:nrow(X), train)

#scale the X and y
scaled_X <- scale(X)
scaled_y <- scale(y)

scaled_train_X <- scaled_X[train, ]
scaled_train_y <- scaled_y[train]

scaled_test_X <- scaled_X[test, ]
scaled_test_y <- scaled_y[test]

linear.mod <- lm(scaled_train_y ~ scaled_train_X)
# Linear model and summary
summary(linear.mod)
# The linear model is encountering multi-collinearity, where there is strong linear correlation among the independent variables. 
# This is causing the model to not obtain clear coefficient estimates;
# With few observations and many variables, the estimates of the regression coefficients can be highly unstable;
# Small changes in the data can lead to significant changes in the estimated coefficients;
# Small changes can be challenging to detect statistically significant relationships between the variables and the target;
grid <- 10^seq(3, -1, length=100)
grid

lasso.cv.out <- cv.glmnet(scaled_train_X,
                          scaled_train_y,
                          alpha=1,
                          lambda=grid,
                          nfolds=30,
                          thresh=1e-10)

bestlam <- lasso.cv.out$lambda.min
bestlam
best_model <- glmnet(scaled_train_X, scaled_train_y, alpha=1, lambda=bestlam)
coef(best_model)

plot(lasso.cv.out)
cv_mse_mean <- mean(lasso.cv.out$cvm)
cv_mse_mean

lasso.pred <- predict(lasso.cv.out, s=bestlam, newx = scaled_test_X)
test_MSE_Lasso <- mean((lasso.pred - scaled_test_y)^2)
test_MSE_Lasso

# Try other random split
set.seed(202306)
train_size <- 38
train <- sample(1:nrow(X), train_size)
test <- setdiff(1:nrow(X), train)

#scale the X and y
scaled_X <- scale(X)
scaled_y <- scale(y)

scaled_train_X <- scaled_X[train, ]
scaled_train_y <- scaled_y[train]

scaled_test_X <- scaled_X[test, ]
scaled_test_y <- scaled_y[test]

linear.mod <- lm(scaled_train_y ~ scaled_train_X)
# Linear model and summary
summary(linear.mod)
# The linear model is encountering multi-collinearity, where there is strong linear correlation among the independent variables. 
# This is causing the model to not obtain clear coefficient estimates;
# With few observations and many variables, the estimates of the regression coefficients can be highly unstable;
# Small changes in the data can lead to significant changes in the estimated coefficients;
# Small changes can be challenging to detect statistically significant relationships between the variables and the target;
grid <- 10^seq(3, -1, length=100)
grid

lasso.cv.out <- cv.glmnet(scaled_train_X,
                          scaled_train_y,
                          alpha=1,
                          lambda=grid,
                          nfolds=38,
                          thresh=1e-10)

bestlam <- lasso.cv.out$lambda.min
bestlam
best_model <- glmnet(scaled_train_X, scaled_train_y, alpha=1, lambda=bestlam)
coef(best_model)

plot(lasso.cv.out)
cv_mse_mean <- mean(lasso.cv.out$cvm)
cv_mse_mean

lasso.pred <- predict(lasso.cv.out, s=bestlam, newx = scaled_test_X)
test_MSE_Lasso <- mean((lasso.pred - scaled_test_y)^2)
test_MSE_Lasso

# Question 3 

library(caret)
library(glmnet)

set.seed(202306)

Charge <- read.csv("insurance.csv",header=TRUE)
head(Charge)

X <- model.matrix(charges ~ ., Charge)[,-1]
y <- Charge$charges
head(X)
dim(X)
head(y)

#scale the X and y
scaled_X <- scale(X)
scaled_y <- scale(y)

# create a grid that will allow us to investigate different models 
# with different combinations of alpha and lambda. 
# This is done using the “expand.grid” function. 
grid <- expand.grid(.alpha  = seq(0, 1, by = 0.1),
                    .lambda = 10^seq(5, -2, length = 100))
grid
head(grid, 11)

train_size <- floor(0.8 * nrow(X)) 
train <- sample(1:nrow(X), train_size)  
test <- setdiff(1:nrow(X), train)

scaled_train_X <- scaled_X[train, ]
scaled_train_y <- scaled_y[train]

scaled_test_X <- scaled_X[test, ]
scaled_test_y <- scaled_y[test]

# use caret for model training
control <- caret::trainControl(method = "cv", number = 10)

# build our models on the training set.
enet.train <- caret::train(x = scaled_X[train, ], 
                           y = scaled_y[train], 
                           method    = "glmnet",
                           trControl = control,
                           tuneGrid  = grid)
enet.train
# best alpha
enet.train$bestTune$alpha
# best lambda
enet.train$bestTune$lambda

enet <- glmnet(x      = scaled_X[train, ], 
               y      = scaled_y[train],  
               alpha  = enet.train$bestTune$alpha,
               lambda = enet.train$bestTune$lambda)

# get coefficients for the 'best' model.
enet.coef <- coef(enet, exact=T)
enet.coef

# make predictions on the test set. 
enet.pred.y <- predict(enet, newx = scaled_test_X)

# calculate residuals - predicted - actuals
enet.resid <- enet.pred.y - scaled_test_y
enet.resid
# calculate test MSE 
mean(enet.resid^2)

# use cv.glmnet to get the lambda.min and lambda.lse value
set.seed(202307)

# define a vector of lambda values to try.
.lambda <- 10^seq(5, -2, length = 100)
.lambda

enet.cv <- cv.glmnet(scaled_X[train, ],
                     scaled_y[train],
                     alpha  = enet.train$bestTune$alpha,    # enet.train$bestTune$alpha = 1
                     lambda = .lambda,
                     nfolds = 10,
                     thresh = 1e-12)
# plot CV results.
plot(enet.cv)


enet.cv$lambda.1se

# show coefficients for this model using lambda of lambda.1se.
# Note: some coefficients have been reduced to zero.
coef(enet.cv, s = "lambda.1se")

# make predictions using the less complex model "lambda.1se"
enet.y.cv.1se <- predict(enet.cv,
                         newx = scaled_X[test, ], 
                         s    = "lambda.1se")

# calculate residuals - predicted - actuals                 
enet.cv.resid.1se <- enet.y.cv.1se - scaled_y[test]
# calculate test MSE (compare with OLS, Ridge, LASSO, "lambda.min"),   MSE = 10203.91
mean(enet.cv.resid.1se ^ 2)

# Use lambda.min
enet.cv$lambda.min

# show coefficients for this model using lambda of lambda.min.
coef(enet.cv, s = "lambda.min")

# now make predictions using the model with the smaller MSE, but more complex model "lambda.min" 
enet.y.cv.min  <- predict(enet.cv,
                          newx = scaled_X[test, ], 
                          s    = "lambda.min")

# calculate residuals - predicted - actuals                 
enet.cv.resid.min <- enet.y.cv.min - scaled_y[test]

mean(enet.cv.resid.min ^ 2)
