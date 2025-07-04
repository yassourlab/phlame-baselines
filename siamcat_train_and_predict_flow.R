# install required packages if not already installed
cran_packages <- c("tidyverse", "dplyr", "ranger")

for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Install BiocManager if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Install SIAMCAT from Bioconductor if not already installed
if (!requireNamespace("SIAMCAT", quietly = TRUE)) {
  BiocManager::install("SIAMCAT")
}

# # get the task name as an argument
args <- commandArgs(trailingOnly = TRUE)
task_name <- args[1]
benchmark_datasets <- args[2]
sprintf("task name: %s",task_name)
sprintf("benchmark datasets dir: %s",benchmark_datasets)
siamcat_default <- 'siamcat_default/'

############# preparing the train and test data ############ 
# create siamcat_outputs directory if it does not exist:
if (!dir.exists(siamcat_default)) {
  dir.create(siamcat_default)
}

# get train and test data
train_path <- sprintf("%s/%s_train.csv",benchmark_datasets, task_name)
test_x_path <- sprintf("%s/%s_test.csv",benchmark_datasets, task_name)
test_y_path <- sprintf("%s/%s_test_gt.csv",benchmark_datasets, task_name)

train_data <- data.frame(read_csv(train_path, col_types = cols(label = col_character())))
rownames(train_data) <- train_data$sample_id

features <- t(train_data %>% select(-label, -sample_id, -subject_id)) /100
meta <- train_data %>% select(sample_id, subject_id, label)

# read the test data and set row names to be the sample_id:
test_data <- data.frame(read_csv(test_x_path, col_types = cols(label = col_character()), ))
test_labels <- data.frame(read_csv(test_y_path, col_types = cols(label = col_character()), ))
rownames(test_data) <- test_data$sample_id
rownames(test_labels) <- test_labels$sample_id

test_features <- t(test_data %>% select(-sample_id, -subject_id)) / 100
test_meta <- test_labels %>% select(sample_id, label)

test_set <- siamcat(
  feat = test_features,
  meta = test_meta,
  label = 'label',
  case = '1',
  verbose = 1
)

############# the helper function for training and potentially testing on an external datast ############ 

train_and_test_model <- function(features,
                                 meta,
                                 norm,
                                 num_folds,
                                 num_repeats,
                                 test_siamcat = NULL) {
  # system.time({
  siamcat_obj <- siamcat(
    feat = features,
    meta = meta,
    label = 'label',
    case = '1',
    verbose = 1
  )
  
  # Feature filtering
  siamcat_obj <- filter.features(
    siamcat_obj,
    verbose = 0
  )

  siamcat_obj <- normalize.features(siamcat_obj, norm.method = norm)
  
  if (num_folds > 1) {
    siamcat_obj <- create.data.split(
      siamcat_obj,
      num.folds = num_folds,
      num.resample = num_repeats,
      verbose = 0
    )
  }
  # Create cross-validation
  
  # Train model
  siamcat_obj <- train.model(
    siamcat_obj,
    verbose = 1,
  )
  
  # Make predictions
  
  siamcat_obj <- make.predictions(
    siamcat_obj,
    siamcat.holdout = test_siamcat,
  )
  
  
  # Evaluate predictions
  siamcat_obj <- evaluate.predictions(siamcat_obj, verbose = 0)
  
  # Get model evaluation
  eval <- eval_data(siamcat_obj)
  
  current_result <- data.frame(
    normalization = norm,
    auc = mean(eval$auroc),
    auc_sd = sd(eval$auroc)
  )
  
  # })
  
  return(list(current_result=current_result, siamcat_obj=siamcat_obj))
}


############# get the result for some default params with no parameter search ############ 
# read the train data and set row names to be the sample_id:
# for now, I chose the params that seemed fine on other datasets from the supplementary

no_search_results <- train_and_test_model(
  features,
  meta,
  "log.std", #
  5,
  1,

  test_siamcat = test_set
)

pred_matrix <- rowMeans(pred_matrix(no_search_results$siamcat_obj))
write.csv(pred_matrix, file = sprintf("%s/%s_predictions_on_test.csv",siamcat_default, task_name), row.names = TRUE)
