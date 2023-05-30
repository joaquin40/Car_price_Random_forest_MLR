## ----message=FALSE--------------------------------------
pacman::p_load(tidyverse, car, DataExplorer, data.table, randomForest, missForest, glmnet, caret, doParallel, foreach, Metrics, cowplot, knitr)


## -------------------------------------------------------
df <- fread("./data/car_ad.csv")


## -------------------------------------------------------
df1 <- df %>% mutate(
          car = as.factor(car),
          body = as.factor(body),
          engType = as.factor(engType),
          registration = as.factor(registration),
          model = as.factor(model),
          drive = as.factor(drive))

df1$engType <- fct_relevel(df1$engType, "Diesel", "Petrol","Other", "Gas")

# set other to NA for engine type
df1$engType[df1$engType == "Other"] <-  NA

str(df1)


## -------------------------------------------------------
lapply(df1,function(x) { length(levels(x))})
df1


## -------------------------------------------------------
for(i in names(df1)){
  
  if (is.factor(df1[[i]])){
    levels(df1[[i]])[levels(df1[[i]]) == ""] = NA
    df1[[i]]<- factor(df1[[i]])
  }
}


## -------------------------------------------------------
plot_missing(df1)

# copy & save the PNG file and close the device
dev.copy(png, "./images/na_values.png")
dev.off()


## -------------------------------------------------------
lapply(df1,function(x) { length(which(is.na(x)))})


## -------------------------------------------------------
rf_na <- missForest(df1[,-c(1,9)],
                    ntree = 100,
                    variablewise = T,
                    verbose= T,
                    mtry = round(ncol(df1[,-c(1,9)])/3))


## -------------------------------------------------------
# df2 <- rf_na$ximp %>% 
#   mutate(car = df1$car,
#          model = df1$model)
df2 <- rf_na$ximp


## -------------------------------------------------------
plot_missing(df2)


## -------------------------------------------------------
summary(df2)
quantile(df2$price)
df3 <- df2[-which(df2$price == 0),]


## -------------------------------------------------------
summary(df3)


## -------------------------------------------------------
scatterplotMatrix(~price + mileage + engV, data = df3, col = "black")
dev.copy(png, "./images/scatterplot.png")
dev.off()

body <- df3 %>% 
  ggplot(aes(body, log(price) )) + 
  geom_boxplot() + 
  labs(x = "Car body type" , y= "log(Price)",
  title ="Relation between the
log of Price and car body")

# engine type
eng <- df3 %>% 
  ggplot(aes(drive, log(price), fill = engType)) + 
  geom_boxplot() + 
  labs(fill = "Type of fuel ", x = "Drive type", y = "log(Price)", title ="Relation between the
log of Price and Drive type")

cowplot::plot_grid(body, eng)

# copy & save the PNG file and close the device
dev.copy(png, "./images/relation_log_price.png")
dev.off()


## -------------------------------------------------------
fit <- lm(price ~ ., data = df3)
summary(fit)
avPlots(fit)
# copy & save the PNG file and close the device
dev.copy(png, "./images/avplots.png")
dev.off()


## -------------------------------------------------------
par(mfrow = c(1,2))
plot(fit, which = 2, cex = .5)
plot(fitted(fit1), rstandard(fit1),
xlab="Fitted values", ylab="Standardized Residuals", cex = .5)
dev.copy(png, "./images/assumption_model.png")
dev.off()


## -------------------------------------------------------
pt <- powerTransform(cbind(df3$mileage + 0.001, df3$engV) ~ 1 )
summary(pt)


## -------------------------------------------------------
engV1 <-  as.numeric(df3$engV)^(-0.5)
powerTransform(df3$price ~ df3$body + sqrt(df3$mileage +0.001) + engV1 + df3$engType + df3$registration + df3$year + df3$drive ) %>% 
  summary()


## -------------------------------------------------------
fit1 <- lm(log(price) ~ body + sqrt(mileage +0.001) + engV1 + engType + registration + year + drive, data = df3)
summary(fit1)


## -------------------------------------------------------
car::vif(fit1)


## -------------------------------------------------------
plot(rstandard(fit1) ~ df3$year, cex = .8,
    xlab="Years",
    ylab="Standardized Residuals")

dev.copy(png, "./images/residuals_year.png")
dev.off()


## -------------------------------------------------------
car::qqPlot(rstandard(fit1))
dev.copy(png, "./images/qnorm_trans.png")
dev.off()


## -------------------------------------------------------
hat <- hatvalues(fit1)
n <- nrow(df3)
p <- ncol(df3)
plot(hatvalues(fit1), rstandard(fit1),
xlab="Leverage", ylab="Standardized Residuals")
abline(v =  3*mean(hat) , lty = 2, lwd = 2, col = "red")
abline(h = c(-6,6), lty = 2, lwd = 2, col = "blue")

dev.copy(png, "./images/leverage_residuals.png")
dev.off()


## -------------------------------------------------------
df3[which(hatvalues(fit1) >3*mean(hat) & abs(rstandard(fit1)) >6 ,)]


## -------------------------------------------------------
hat <- hatvalues(fit1)
cooks <- cooks.distance(fit1)
index <- which(cooks > 4/(n - p - 1) )

# number of influential points
length(unique(index))

influenceIndexPlot(fit1,vars = c("hat", "Cook"),id = TRUE)

dev.copy(png, "./images/cooks.png")
dev.off()


## -------------------------------------------------------
df4 <- df3[-index,]

engV1 <-  as.numeric(df4$engV)^(-0.5)

fit2 <- lm(log(price) ~ body + sqrt(mileage +0.001) + engV1 + engType + registration + year + drive, data = df4)
summary(fit2)


## -------------------------------------------------------
plot(rstandard(fit2) ~ df4$year, cex = .8,
    xlab="Years",
    ylab="Standardized Residuals")


dev.copy(png, "./images/res_vs_years_no_infl.png")
dev.off()


## -------------------------------------------------------
car::qqPlot(rstandard(fit2))
dev.copy(png, "./images/qqplot_no.png")
dev.off()

plot(rstandard(fit2) ~ fitted(fit2), cex = .8, 
     xlab = "Fitted Values", ylab = "Standardize Residuals")
dev.copy(png, "./images/fitted_residuals_no.png")
dev.off()



## -------------------------------------------------------
hat <- hatvalues(fit2)
n <- nrow(df4)
p <- ncol(df4)
plot(hatvalues(fit2), rstandard(fit2),
xlab="Leverage", ylab="Standardized Residuals")
abline(v = 3*mean(hat) , lty = 2, lwd = 2, col = "red")
abline(h = c(-6,6), lty = 2, lwd = 2, col = "blue")

dev.copy(png, "./images/outliers_no.png")
dev.off()


## -------------------------------------------------------
summary(fit1)
-2.999e-01*100 # engine type
8.569e-01 *100 # drive


## -------------------------------------------------------
set.seed(1)
train_index <- createDataPartition(df3$price, p = .7, list = FALSE)
train <- df3[train_index,]
test <- df3[-train_index,]


## -------------------------------------------------------

# cross-validation method
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
tune <- expand.grid(alpha = 0, lambda = seq(0.01, 1, length = 100))

model <- train(log(price) ~ body + sqrt(mileage + 0.001) + engV + 
    engType + registration + year + drive,
                      data = train,
                      method = "glmnet",
                      trControl = control,
                      tuneGrid = tune)


## -------------------------------------------------------
pred <- predict(model , newdata = test)


## -------------------------------------------------------
rmse <- sqrt(mean((test$price - exp(pred) )^2)) 
mae <- Metrics::mae(test$price, exp(pred))

table <- data.frame(
"Residual Mean Square error" =  round(rmse,2),
      "Mean Absolute error" =  round(mae,2))

knitr::kable(table,align = "lccccc",caption = "Random Forest Prediction Results")



## -------------------------------------------------------
set.seed(1)
train_index <- createDataPartition(df3$price, p = .7, list = FALSE)
train <- df3[train_index,]
test <- df3[-train_index,]


## -------------------------------------------------------
cl <- makeCluster(detectCores())
registerDoParallel(cl)


## -------------------------------------------------------
# define models
algorithms <- list(
  "glmnet" = list(method = "glmnet"),
  "glm" = list(method = "glm"),
  "rf" = list(method = "rf")
)
algorithms <- c("glmnet", "glm", "rf")


# cross-validation method
control <- trainControl(method = "cv", number = 10)


# tuning parameters
tune_glmnet <- expand.grid(alpha = 0, lambda = seq(0.01, 1, length = 100))
rf_grid <- expand.grid(mtry = 1:4)
param_grids <- list(glmnet = tune_glmnet, glm = glm_grid, rf = rf_grid)

# train model
results <- foreach(alg = algorithms ,.packages = "caret", .combine = "rbind") %dopar% {
  
  if (alg == "glm"){
    model <- train(log(price) ~ body + sqrt(mileage + 0.001) + engV + 
    engType + registration + year + drive,
    data = train,
    method = alg,
    trControl = control)
  } else {
    model <- train(log(price) ~ body + sqrt(mileage + 0.001) + engV + 
    engType + registration + year + drive,
    data = train,
    method = alg,
    trControl = control,
    tuneGrid = param_grids[[alg]])
  }
  
  # Extract model performance metrics
  if (alg == "rf"){
    model_metrics <- list(
    model = model,
    RMSE = min(model$results$RMSE),
    R2 = max(model$results$Rsquared),
    MAE = min(model$results$MAE) )
  } else {
    model_metrics <- list(
    model = coef(model$finalModel),
    RMSE = min(model$results$RMSE),
    R2 = max(model$results$Rsquared),
    MAE = min(model$results$MAE)
  )
  }
  
  list(algorithm = alg, model = model, model_metrics = model_metrics)
  
  
  # Return model performance metrics
  model_metrics
}



## -------------------------------------------------------
results[3]


## -------------------------------------------------------
pred_rf <- predict(results[3], newdata = test) %>% unlist()


## -------------------------------------------------------
rmse <- sqrt(mean( (test$price - exp(pred_rf))  ^2) )
mae <- mean(abs(test$price - exp(pred_rf)))


## -------------------------------------------------------
table <- data.frame(
"Residual Mean Square error" =  round(rmse,2),
      "Mean Absolute error" =  round(mae,2))

knitr::kable(table,align = "lccccc",caption = "Random Forest Prediction Results")



## -------------------------------------------------------
# Stop the parallel backend
stopCluster(cl)


## -------------------------------------------------------
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

unregister_dopar()

