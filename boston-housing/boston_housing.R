# library(tidyverse)
boston <- readr::read_csv('housingdata.csv')
set.seed(1255126)

# missing completely at random
create_mcar <- function(dataframe, percentage_as_decimal){  # percentage should be a numeric value
  mcar_df <- dataframe  # create a copy of the original df
  random_rows <- sample(nrow(mcar_df),  # select a random sample from the rows in the df
                        round(nrow(mcar_df) * percentage_as_decimal))  # select the nearest whole number to n% where n = percentage
  for (i in random_rows){
      mcar_df[i, 14] <- NA  # for each location [i, response_variable_column] assign it an NA value
  }
  mcar_df  # return the new df with completely random missing values
}

mcar_df <- create_mcar(boston, .15)


# missing at random
create_mar <- function(dataframe, column_index, cutoff_values){
  internal_copy <- dataframe
  for (i in 1:nrow(dataframe)){
     if (internal_copy[[i, column_index]] > cutoff_values[1] && internal_copy[[i, column_index]] < cutoff_values[2]){
        internal_copy[[i, 14]] <- NA
    }
  }
  internal_copy
}

mar_df <- create_mar(boston, 5, c(.51, .56))
# sum(is.na(mar_df$MEDV)) / nrow(mar_df)


# missing not at random
create_mnar <- function(dataframe, response_column_index, cutoff_values){
  internal_copy <- dataframe
  for (i in 1:nrow(dataframe)){
      if ((internal_copy[[i, response_column_index]] > cutoff_values[1] && internal_copy[[i, response_column_index]] < cutoff_values[2])) {
          internal_copy[[i, response_column_index]] <- NA
     }
  }
  internal_copy
}

mnar_df <- create_mnar(boston, 14, c(16, 19.5))
# sum(is.na(mnar_df$MEDV)) / nrow(mnar_df)


# write the dataframes to CSVs
# write.csv(mcar_df, 'mcar.csv', row.names=T)
# write.csv(mar_df, 'mar.csv', row.names=T)
# write.csv(mnar_df, 'mnar.csv', row.names=T)


# models
# baseline model with all the data
baseline_lm <- lm(MEDV ~ ., data = boston)
summary(baseline_lm)

# model with mcar data
mcar_lm <- lm(MEDV ~ ., data = mcar_df)
summary(mcar_lm)

# model with mar data
mar_lm <- lm(MEDV ~ ., data = mar_df)
summary(mar_lm)

# model with mnar data
mnar_lm <- lm(MEDV ~ ., data = mnar_df)
summary(mnar_lm)


# test/train models
# split dataset
create_test <- function(dataframe) {
  split1 <- sample(c(rep(0, 450), rep(1, 56)))
  # table(split1)
  test <- dataframe[split1 == 1,]
  return(test)
}
create_train <- function(dataframe) {
  split1 <- sample(c(rep(0, 450), rep(1, 56)))
  # table(split1)
  train <- dataframe[split1 == 0,]
  return(train)
}

# baseline test/train/lm/predict
boston_test <- create_test(boston)
boston_train <- create_train(boston)
boston_train_lm <- lm(MEDV ~ ., data = boston_train)
boston_test_preds <- predict(boston_train_lm, boston_test) %>%
  print()
summary(boston_train_lm)

(boston_test_preds + sd(boston_test$MEDV, na.rm = T)) < boston_test$MEDV
(boston_test_preds - sd(boston_test$MEDV, na.rm = T)) > boston_test$MEDV

# mar test/train/lm/predict
mar_test <- create_test(mar_df)
mar_train <- create_train(mar_df)
mar_train_lm <- lm(MEDV ~ ., data = mar_train)
mar_test_preds <- predict(mar_train_lm, mar_test) %>%
  print()
summary(mar_train_lm)
(mar_test_preds + sd(mar_test$MEDV, na.rm = T)) < mar_test$MEDV
(mar_test_preds - sd(mar_test$MEDV, na.rm = T)) > mar_test$MEDV

# mcar test/train/lm/predict
mcar_test <- create_test(mcar_df)
mcar_train <- create_train(mcar_df)
mcar_train_lm <- lm(MEDV ~ ., data = mcar_train)
mcar_test_preds <- predict(mcar_train_lm, mcar_test) %>%
  print()
summary(mcar_train_lm)
(mcar_test_preds + sd(mcar_test$MEDV, na.rm = T)) < mcar_test$MEDV
(mcar_test_preds - sd(mcar_test$MEDV, na.rm = T)) > mcar_test$MEDV

# mnar test/train/lm/predict
mnar_test <- create_test(mnar_df)
mnar_train <- create_train(mnar_df)
mnar_train_lm <- lm(MEDV ~ ., data = mnar_train)
mnar_test_preds <- predict(mnar_train_lm, mnar_test) %>%
  print()
summary(mnar_train_lm)
(mnar_test_preds + sd(mnar_test$MEDV, na.rm = T)) < mnar_test$MEDV
(mnar_test_preds - sd(mnar_test$MEDV, na.rm = T)) > mnar_test$MEDV


# group assignments
# we got halosight2