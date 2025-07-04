import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import os
import sys

seed = 17
if __name__ == '__main__':
    # parse args
    task_name, benchmark_datasets_dir, model_name = sys.argv[1:]
    train_path = f"{benchmark_datasets_dir}/{task_name}_train.csv"
    test_x_path = f"{benchmark_datasets_dir}/{task_name}_test.csv"

    train = pd.read_csv(train_path)
    test_x = pd.read_csv(test_x_path)

    models_dict = {'random_forest': RandomForestClassifier,
                   'xgboost': GradientBoostingClassifier,
                   'svm': SVC,
                   'logistic_regression': LogisticRegression}

    model_params = {
        'random_forest': {},
        'xgboost': {},
        'svm': {'probability': True},
        'logistic_regression': {'max_iter': 1000}
    }

    model_func = models_dict[model_name]
    model = model_func(random_state=seed, **model_params[model_name])
    model.fit(train.drop(columns=['label', 'sample_id', 'subject_id']), train.label)
    probabilities = model.predict_proba(test_x.drop(columns=['sample_id', 'subject_id']))
    model_dir = f'{model_name}_default'
    os.makedirs(model_dir, exist_ok=True)
    pd.DataFrame(index=test_x.sample_id, data=probabilities)[[1]].to_csv(
        f"{model_dir}/{task_name}_predictions_on_test.csv")