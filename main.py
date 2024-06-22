# %%
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, LeaveOneGroupOut
from tqdm import tqdm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
)
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from IPython.display import display

# %%
data = pd.read_csv("../Project/HR_data.csv", delimiter=",")
len(data["Individual"].unique())
# %%
groups = data["Individual"]
data.drop(
    columns=["Round", "Phase", "Cohort", "Puzzler", "Individual", "Unnamed: 0"],
    inplace=True,
)
data.head()

# %%
X = data.drop("Frustrated", axis=1)
X.head()

# %%
len(X.columns), len(X)

# %%
y = data["Frustrated"]
y.head()
y.plot.hist(bins=range(9))  # range(9) generates [0, 1, 2, ..., 8]

# %%
y.describe()

# %%
# Static threshold: 5 and above is frustrated, below 5 is not
static_threshold = 5
(y >= static_threshold).mean()
# we will have to handle class imbalance

# %%
# Define the pipelines
rf_pipe = ImbPipeline(
    [
        ("smote", SMOTE()),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(class_weight="balanced")),
    ]
)
log_pipe = ImbPipeline(
    [
        ("smote", SMOTE()),
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(class_weight="balanced")),
    ]
)

knn_pipe = ImbPipeline(
    [("smote", SMOTE()), ("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
)
ada_pipe = ImbPipeline(
    [("smote", SMOTE()), ("scaler", StandardScaler()), ("ada", AdaBoostClassifier())]
)
baseline_stratified_pipe = ImbPipeline(
    [
        ("smote", SMOTE()),
        ("scaler", StandardScaler()),
        ("dummy", DummyClassifier(strategy="stratified")),
    ]
)
baseline_pos_pipe = ImbPipeline(
    [
        ("smote", SMOTE()),
        ("scaler", StandardScaler()),
        ("dummy", DummyClassifier(strategy="constant", constant=1)),
    ]
)
baseline_neg_pipe = ImbPipeline(
    [
        ("smote", SMOTE()),
        ("scaler", StandardScaler()),
        ("dummy", DummyClassifier(strategy="constant", constant=0)),
    ]
)

s = display(
    rf_pipe,
    log_pipe,
    knn_pipe,
    ada_pipe,
    baseline_stratified_pipe,
    baseline_pos_pipe,
    baseline_neg_pipe,
)

# %%

# Define parameter grids
rf_param_grid = {
    "rf__n_estimators": [10, 50, 100, 200, 500, 1000],
    "rf__max_depth": [2, 5, 10, 20],
}
log_param_grid = {"log_reg__C": np.logspace(-3, 3, 7)}
knn_param_grid = {"knn__n_neighbors": [1, 3, 5, 7, 9]}
ada_param_grid = {"ada__n_estimators": [10, 50, 100, 200, 500]}

# Initialize lists to store results
(
    rf_y,
    log_y,
    knn_y,
    ada_y,
    baseline_stratified_y,
    baseline_pos_y,
    baseline_neg_y,
    y_true,
) = ([], [], [], [], [], [], [], [])
rf_prob, log_prob, knn_prob, ada_prob, prior_prob = [], [], [], [], []

# Initialize lists to store results
rf_test_scores = []
rf_best_params = []

log_test_scores = []
log_best_params = []

knn_test_scores = []
knn_best_params = []

ada_test_scores = []
ada_best_params = []

baseline_pos_scores = []
baseline_neg_scores = []
baseline_stratified_scores = []

y_true = []


class KFoldHelper:
    def __init__(
        self,
        kfold: sklearn.model_selection._split._BaseKFold,
        x: np.ndarray,
        classes: np.ndarray = None,
        groups: np.ndarray = None,
    ):
        self.iter = kfold.split(x, y=classes, groups=groups)

    def __iter__(self):
        for idxsTrain, idxsTest in self.iter:
            yield idxsTrain, idxsTest


# Outer cross-validation
CV = LeaveOneGroupOut()
# CV = KFold(n_splits=2, shuffle=True, random_state=42)
for train_index, test_index in tqdm(CV.split(X, y, groups=groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Print the unique individuals in the training and testing sets
    # print(X_train.loc[:,"Individual"].unique())
    # print(X_test.loc[:,"Individual"].unique())

    # Binarize the target variable using the static threshold
    y_train_binary = (y_train >= static_threshold).astype(int)
    y_test_binary = (y_test >= static_threshold).astype(int)

    inner_cv = KFold(n_splits=2)
    y_true.extend(y_test_binary)

    # Random Forest
    model = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        cv=inner_cv,
        n_jobs=-1,
        scoring="f1",
    )
    model.fit(X_train, y_train_binary)
    rf_predictions = model.predict(X_test)
    rf_y.extend(rf_predictions)
    rf_prob.extend(model.predict_proba(X_test)[:, 1])

    # Logistic Regression
    model = GridSearchCV(
        estimator=log_pipe,
        param_grid=log_param_grid,
        cv=inner_cv,
        n_jobs=-1,
        scoring="f1",
    )
    model.fit(X_train, y_train_binary)
    log_predictions = model.predict(X_test)
    log_y.extend(log_predictions)
    log_prob.extend(model.predict_proba(X_test)[:, 1])

    # KNN Classifier
    model = GridSearchCV(
        estimator=knn_pipe,
        param_grid=knn_param_grid,
        cv=inner_cv,
        n_jobs=-1,
        scoring="f1",
    )
    model.fit(X_train, y_train_binary)
    knn_predictions = model.predict(X_test)
    knn_y.extend(knn_predictions)
    knn_prob.extend(model.predict_proba(X_test)[:, 1])

    # AdaBoost Classifier
    model = GridSearchCV(
        estimator=ada_pipe,
        param_grid=ada_param_grid,
        cv=inner_cv,
        n_jobs=-1,
        scoring="f1",
    )
    model.fit(X_train, y_train_binary)
    ada_predictions = model.predict(X_test)
    ada_y.extend(ada_predictions)
    ada_prob.extend(model.predict_proba(X_test)[:, 1])

    # Baseline Stratisfied Classifier
    baseline_stratified_pipe.fit(X_train, y_train_binary)
    baseline_predictions = baseline_stratified_pipe.predict(X_test)
    baseline_stratified_y.extend(baseline_predictions)

    # Baseline Positive Classifier
    baseline_pos_pipe.fit(X_train, y_train_binary)
    baseline_pos_predictions = baseline_pos_pipe.predict(X_test)
    baseline_pos_y.extend(baseline_pos_predictions)

    # Baseline Negative Classifier
    baseline_neg_pipe.fit(X_train, y_train_binary)
    baseline_neg_predictions = baseline_neg_pipe.predict(X_test)
    baseline_neg_y.extend(baseline_neg_predictions)

# Calculate metrics for each classifier
models = [
    "RF",
    "LR",
    "KNN",
    "ADA",
    "Base[S]",
    "Base[+]",
    "Base[-]",
]

metrics = {
    "Classifier": models,
    "F1 Score": [
        f1_score(y_true, rf_y),
        f1_score(y_true, log_y),
        f1_score(y_true, knn_y),
        f1_score(y_true, ada_y),
        f1_score(y_true, baseline_stratified_y),
        f1_score(y_true, baseline_pos_y),
        f1_score(y_true, baseline_neg_y),
    ],
    "Balanced Accuracy": [
        balanced_accuracy_score(y_true, rf_y),
        balanced_accuracy_score(y_true, log_y),
        balanced_accuracy_score(y_true, knn_y),
        balanced_accuracy_score(y_true, ada_y),
        balanced_accuracy_score(y_true, baseline_stratified_y),
        balanced_accuracy_score(y_true, baseline_pos_y),
        balanced_accuracy_score(y_true, baseline_neg_y),
    ],
    "Precision": [
        precision_score(y_true, rf_y),
        precision_score(y_true, log_y),
        precision_score(y_true, knn_y),
        precision_score(y_true, ada_y),
        precision_score(y_true, baseline_stratified_y),
        precision_score(y_true, baseline_pos_y),
        precision_score(y_true, baseline_neg_y),
    ],
    "Recall": [
        recall_score(y_true, rf_y),
        recall_score(y_true, log_y),
        recall_score(y_true, knn_y),
        recall_score(y_true, ada_y),
        recall_score(y_true, baseline_stratified_y),
        recall_score(y_true, baseline_pos_y),
        recall_score(y_true, baseline_neg_y),
    ],
    "NPV": [
        precision_score(y_true, rf_y, pos_label=0),
        precision_score(y_true, log_y, pos_label=0),
        precision_score(y_true, knn_y, pos_label=0),
        precision_score(y_true, ada_y, pos_label=0),
        precision_score(y_true, baseline_stratified_y, pos_label=0),
        precision_score(y_true, baseline_pos_y, pos_label=0),
        precision_score(y_true, baseline_neg_y, pos_label=0),
    ],
    "MCC": [
        matthews_corrcoef(y_true, rf_y),
        matthews_corrcoef(y_true, log_y),
        matthews_corrcoef(y_true, knn_y),
        matthews_corrcoef(y_true, ada_y),
        matthews_corrcoef(y_true, baseline_stratified_y),
        matthews_corrcoef(y_true, baseline_pos_y),
        matthews_corrcoef(y_true, baseline_neg_y),
    ],
    "ROC-AUC": [
        roc_auc_score(y_true, rf_prob),
        roc_auc_score(y_true, log_prob),
        roc_auc_score(y_true, knn_prob),
        roc_auc_score(y_true, ada_prob),
        None,  # Baseline classifier does not provide probability scores
        None,  # Baseline classifier does not provide probability scores
        None,  # Baseline classifier does not provide probability scores
    ],
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics)
# set the number of decimals to 4
metrics_df = metrics_df.round(4)

metrics_df.to_latex("metrics.tex", float_format="%.4f")
# Display the table
display(metrics_df)


# %% Do a Cochran's Q test to see if the classifiers are significantly different
rf = np.array(rf_y)
log = np.array(log_y)
knn = np.array(knn_y)
ada = np.array(ada_y)
baseline_stratified = np.array(baseline_stratified_y)
baseline_pos = np.array(baseline_pos_y)
baseline_neg = np.array(baseline_neg_y)

y_true = np.array(y_true)

# Organize the predictions into a binary matrix
results = np.array(
    [
        rf_y == y_true,
        log_y == y_true,
        knn_y == y_true,
        ada_y == y_true,
        baseline_stratified_y == y_true,
        baseline_pos_y == y_true,
        baseline_neg_y == y_true,
    ]
).T

# Perform Cochran's Q test
cochran_q_test = sm.stats.cochrans_q(results)
print("Cochran's Q test result:", cochran_q_test)

# Interpret the p-value
if cochran_q_test.pvalue < 0.05:
    print("There is a statistically significant difference between the classifiers.")
else:
    print("There is no statistically significant difference between the classifiers.")

# Assuming 'models' is a list of model names and 'results' is a numpy array with the results of the classifiers


# %% Perform McNemar's test for each pair of classifiers
# Initialize a dictionary to store the p-values
p_values = []
diff_conf_int = []
# Perform McNemar's test for each pair of classifiers
for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        if i < j:
            # Make the contingency table
            table = pd.crosstab(results[:, i], results[:, j])
            # Perform McNemar's test
            result = sm.stats.mcnemar(table)
            # Store the result as a tuple
            p_values.append((model1, model2, result.pvalue))


# Convert the list of tuples into a pandas DataFrame
p_values_df = pd.DataFrame(p_values, columns=["Model 1", "Model 2", "P-Value"]).round(4)

# Output the DataFrame
pivot_tabel = (
    p_values_df.pivot(index="Model 1", columns="Model 2", values="P-Value")
    .reindex(index=models, columns=models)
    .T.dropna(axis=1, how="all")
    .dropna(axis=0, how="all")
)

pivot_tabel.to_latex("mc_nemar.tex", float_format="%.4f")
pivot_tabel
