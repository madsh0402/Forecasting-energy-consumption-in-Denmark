### CNN Proff of Concept
rm(list=ls())
knitr::opts_chunk$set(echo = TRUE)
options(repos="https://cran.rstudio.com" )

list.of.packages <- c("tsibble","dplyr","zoo","lubridate","padr","keras","ggplot2","caret")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()
                                   [,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(tsibble)
library(dplyr)
library(zoo)
library(lubridate)
library(padr)
library(keras)
library(ggplot2)
library(caret)



setwd("C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Speciale/Data/Data fra energidataservice.dk")
df <- read.csv("Production and Consumption - Settlement.csv") 
summary(df)
df <- as.data.frame(c(df[2],df[3],df[23]))
head(df)

#tager summen for hver dk1 og dk2
df <- df %>%
  group_by(HourDK) %>%
  summarise(GrossConsumptionMWh = sum(GrossConsumptionMWh, na.rm = TRUE))

#laver om til tidsserie
df$HourDK <- as.POSIXct(df$HourDK, format = "%Y-%m-%dT%H:%M:%S")
ts.df <- as_tsibble(df,index = HourDK)
head(ts.df)
tail(ts.df)

############################Filling timeseries##################################
ts_filled <- ts.df %>%
  fill_gaps(GrossConsumptionMWh = NA)

#replace missing values using Interpolation
ts_filled$GrossConsumptionMWh <- na.approx(ts_filled$GrossConsumptionMWh)

################################################################################
#################### Omdanne Tidsserie til Gitter ##############################
################################################################################
# Sæt vinduesstørrelsen
window_size <- 24

# Generer rullende vinduer af tidsserien
xdata_temp <- rollapply(ts_filled$GrossConsumptionMWh, width = window_size, by = 1, function(x) x)

# Opbygning af xdata og ydata
N <- nrow(xdata_temp) - 2 # Reducerer med 1 mere, fordi vi nu skal forudsige en tid frem i stedet for den aktuelle tid

xdata <- array(NA, dim = c(N, window_size, window_size, 1))
for (i in 1:N) {
  xdata[i, , , 1] <- embed(xdata_temp[i:(i+window_size-1)], window_size)
}

ydata <- ts_filled$GrossConsumptionMWh[(window_size + 2):(N + window_size + 1)] # Nu forudsiger vi en tid frem

# Split data i trænings- og testsæt
Ntrain <- 130000
Ntest <- N - Ntrain

# First assignment
xtrain <- xdata[1:(Ntrain - 1), , , 1]  # Only take the first index of the 4th dimension
ytrain <- ydata[1:(Ntrain - 1)]

xtest <- xdata[Ntrain:N, , , 1]  # Same here, only the first index of the 4th dimension
ytest <- ydata[Ntrain:N]

#random image
# Select an image index
image_index <- sample(1:Ntrain, 1)  # or any valid index

# Display the image
image(t(apply(xtrain[image_index,,], 2, rev)))


# Standardiser xdata
standardafv <- sd(c(xtrain, xtest))
middelv <- mean(c(xtrain, xtest))

xtrain <- (xtrain - middelv) / standardafv
xtest <- (xtest - middelv) / standardafv

#opsæt model til regression
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu', input_shape = c(window_size,window_size,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 10, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'linear') %>%
  
  compile(
    loss = loss_mean_squared_error,
    optimizer = optimizer_rmsprop(),
  )

callback <- callback_early_stopping(monitor = "val_loss", patience = 5)

#laver fit 
fit <- model %>%
  fit(
    x = xtrain, 
    y = ytrain,
    batch_size = 128,
    epochs = 50,
    validation_split = 0.2,
    callbacks = list(callback)
  )

plot(fit)
summary(model)
fit

################################################################################
################################ Predicting ####################################
################################################################################

test_predictions <- model %>% predict(xtest,batch_size=Ntest)

# Calculate errors
mse <- mean((test_predictions - ytest)^2)
mae <- mean(abs(test_predictions - ytest))
rmse <- sqrt(mse)
mpe <- mean((test_predictions - ytest) / ytest) * 100
mape <- mean(abs((test_predictions - ytest) / ytest)) * 100

# Create a dataframe to hold the results
results <- data.frame(
  MSE = mse,
  RMSE = rmse,
  MAE = mae,
  MPE = mpe,
  MAPE = mape
)
print(results)

# Create a data frame for plotting
plot_data <- as.data.frame(data.frame(Actual = ytest, Predicted = test_predictions))
plot_data$Time <- ts_filled$HourDK[(Ntrain + window_size):(N + window_size)]
# Plot Actual vs Predicted values
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  ggtitle("Actual vs Predicted Values") +
  xlab("Actual Values") +
  ylab("Predicted Values") +
  theme_minimal()

# Create the plot for all time 
ggplot(plot_data, aes(x = Time)) +
  geom_line(aes(y = Actual, color = 'Actual')) +
  geom_line(aes(y = Predicted, color = 'Predicted')) +
  labs(x = "Time", y = "Value", color = "Line") +
  theme_minimal() +
  ggtitle("Actual vs Predicted Values")

last_24 <-tail(plot_data,24*7)

#create the plot for last x
ggplot(last_24, aes(x = Time)) +
  geom_line(aes(y = Actual, color = 'Actual')) +
  geom_line(aes(y = Predicted, color = 'Predicted')) +
  labs(x = "Time", y = "Value", color = "Line") +
  theme_minimal() +
  ggtitle("Actual vs Predicted Values")

