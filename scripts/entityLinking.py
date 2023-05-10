import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pickle
import time
import configparser

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

MODEL_TYPE = config.get('legacy', 'model')
DO_OVERSAMPLING = config.getboolean('legacy', 'do_oversampling')
DATASET_PATH = DATA_DIR + 'el training set.parquet'
DATASET_FORMAT = 'parquet'
EXPERIMENT_NAME = 'FTB'

start = time.time()

print('entity linking')

# Dataset should have header
if DATASET_FORMAT == 'tsv':
    data = pd.read_csv(DATASET_PATH, sep='\t', index_col=False)
else:
    data = pd.read_parquet(DATASET_PATH)

# stratifiedgroupkfold, train test split
train_idx, test_idx = next(GroupShuffleSplit(test_size=.3).split(X=data.iloc[:, 3:], y=data.iloc[:, 2], groups=data.iloc[:,0]))
x_train = data.iloc[train_idx, 3:]
y_train = data.iloc[train_idx, 2]
x_test = data.iloc[test_idx, 3:]
y_test = data.iloc[test_idx, 2]

# Oversampling for better train results
if DO_OVERSAMPLING:
    print('-applying oversampling')
    print(f'-train set shape: {x_train.shape}')
    x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)
    print(f'-train set shape after SMOTE: {x_train.shape}')
else:
    print('-skipping oversampling')

if MODEL_TYPE == 'r_forest':
    clf = RandomForestClassifier()
elif MODEL_TYPE == 'mlp':
    clf = MLPClassifier(early_stopping=True)
elif MODEL_TYPE == 'log_reg':
    clf = LogisticRegression(max_iter=1000)
elif MODEL_TYPE == 'd_tree':
    clf = DecisionTreeClassifier()

multiplier = 1  # adjust n_jobs dependent on ram/dataset size
if MODEL_TYPE == 'r_forest':
    random_search = RandomizedSearchCV(clf, param_distributions={'criterion': ["gini", "entropy"], 'n_estimators': range(5, 30)}, n_iter=5, scoring=metrics.make_scorer(metrics.f1_score, average='micro'), random_state=2, verbose=3, n_jobs=4*multiplier)
elif MODEL_TYPE == 'mlp':
    random_search = RandomizedSearchCV(clf, param_distributions={'hidden_layer_sizes': range(50, 300), 'learning_rate': ['constant', 'adaptive']}, n_iter=1, scoring=metrics.make_scorer(metrics.f1_score, average='micro'), random_state=2, verbose=3, n_jobs=4*multiplier)
elif MODEL_TYPE == 'log_reg':
    random_search = RandomizedSearchCV(clf, param_distributions={}, n_iter=1, scoring=metrics.make_scorer(metrics.f1_score, average='micro'), random_state=2, verbose=3, n_jobs=6*multiplier)
elif MODEL_TYPE == 'd_tree':
    random_search = GridSearchCV(clf, param_grid={'criterion': ["gini", "entropy"], 'class_weight': [None, {True: 1, False: 5}], 'splitter': ['best', 'random']}, scoring=metrics.make_scorer(metrics.f1_score, average='micro'), verbose=3, n_jobs=6*multiplier)

print(f'-fitting model {MODEL_TYPE}')
random_search.fit(x_train, y_train)
runtime = time.time() - start
print(f'-finishing training after: {runtime}')

random_search.score(x_test, y_test)

pred = random_search.predict(x_test)

# log model performance and metadata
with open(f'{DATA_DIR}class_report.txt', 'w', encoding='utf-8') as file:
    report = metrics.classification_report(y_test, pred)
    file.write(f'Performance model {MODEL_TYPE}:\n')
    file.write(f'On Dataset: {DATASET_PATH}\n')
    file.write(f"runtime: {time.strftime('%H:%M:%S', time.gmtime(runtime))}\n\n")
    file.write(report)
    print(report)


best_estim = random_search.best_estimator_

# save model
def save_model(model, type: str):
    FILENAME = f'{DATA_DIR}{type}.sav'
    with open(FILENAME, 'wb') as file:
        pickle.dump(model, file)

save_model(best_estim, MODEL_TYPE)