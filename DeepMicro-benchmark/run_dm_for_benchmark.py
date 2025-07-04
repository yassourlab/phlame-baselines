from dm_benchmark import DeepMicrobiome
import pandas as pd
import os
import sys

if __name__ == '__main__':
    # parse args
    task_name, benchmark_datasets_dir = arguments = sys.argv[1:]
    train_path = f"{benchmark_datasets_dir}/{task_name}_train.csv"
    test_x_path = f"{benchmark_datasets_dir}/{task_name}_test.csv"
    test_y_path = f"{benchmark_datasets_dir}/{task_name}_test_gt.csv"

    model_dir = f'/sci/backup/morani/lab/Projects/gut_mgx_benchmark/mgx-benchmark-code/benchmarking_flow/deepmicro_default'
    os.makedirs(model_dir, exist_ok=True)

    # create train and test sets:
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=['label', 'sample_id', 'subject_id']).values
    y_train = train_df['label'].values
    X_test_df = pd.read_csv(test_x_path)
    X_test = X_test_df.drop(columns=['sample_id', 'subject_id']).values
    y_test = pd.read_csv(test_y_path)['label'].values


    class Args:
        def __init__(self):
            self.dims = '50'
            self.act = 'relu'
            self.max_epochs = 2000
            self.aeloss = 'mse'
            self.ae_oact = 'store_true'
            self.patience = 20
            self.rf_rate = 0.1
            self.st_rate = 0.25
            self.no_trn = 'store_true'
            self.numFolds = 5
            self.numJobs = -1
            self.scoring = 'roc_auc'


    args = Args()

    dm = DeepMicrobiome(X_train, X_test, y_train, y_test, task_name, model_dir)
    dm.cae()

    rf_hyper_parameters = [{'n_estimators': [100],
                            'max_features': ['sqrt'],
                            'min_samples_leaf': [1],
                            'criterion': ['gini']
                            }, ]
    test_prob = dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
    pd.DataFrame(index=X_test_df.sample_id, data=test_prob)[[1]].to_csv(
        f"{model_dir}/{task_name}_predictions_on_test.csv")
