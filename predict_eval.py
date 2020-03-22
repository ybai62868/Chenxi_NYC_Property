# Name: Chenxi Zhu

import logging
import time

# import lightgbm as lgb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import ensemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set random seed and logging for reproducibility
random_seed = 666

logging.basicConfig(
    filename="./logs/predict_eval_info.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)


# Function definitions
def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """Mean absolute percentage error regression loss."""
    """
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
        """
    output_errors = np.average(np.abs(y_pred / y_true - 1),
                               weights=sample_weight, axis=0)

    return(output_errors)


def median_absolute_percentage_error(y_true, y_pred):
    """Median absolute percentage error regression loss."""
    """
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
        """
    output_errors = np.median(np.abs(y_pred / y_true - 1), axis=0)

    return(output_errors)


def plot_model_performance(y_pred, y_test, model_name, zoom=False):
    """Save a scatter plot of the predicted vs actuals."""
    """zoom: Zoom in on the part of the distribution where most data lie."""

    if (zoom is True):
        axes_limit = 0.2 * 1e7
        path_suffix = "_zoom"
    else:
        axes_limit = y_pred.max()*1.1
        path_suffix = ""

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred, alpha=0.1)
    line = mlines.Line2D([0, 1], [0, 1], color="red")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    subplot_title = "{} \n Median AE: {:.0f}, Median APE: {:.3f}".format(
                model_name,
                median_absolute_error(y_test, y_pred),
                median_absolute_percentage_error(y_test, y_pred)
                )
    ax.set(title=subplot_title,
           xlabel="Actual selling price in $",
           ylabel="Predicted selling price in $",
           xlim=(0, axes_limit),
           ylim=(0, axes_limit))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    fig.savefig(
        "./figs/model_evaluation_{}{}.png".format(
            model_name,
            path_suffix),
        dpi=1000,
        bbox_inches="tight"
        )
    plt.close(fig)


def log_train_set_info_and_performance(model, model_name):
    """Log info on the tuning process and performance on train set."""
    logging.info("# Tuning hyper-parameters for {}".format(model_name))
    logging.info("Best parameters set found on train set:")
    logging.info(model.best_params_)
    logging.info("Grid scores on train set:")

    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in zip(
                                 means,
                                 stds,
                                 model.cv_results_["params"]):
        logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def log_test_set_performance(model_name, y_pred, y_test):
    """Log performance on test set."""
    logging.info("# Test set performance for {}".format(model_name))
    logging.info("Score on the test set:")
    logging.info("R2 score: " + str(round(r2_score(y_test, y_pred), 3)))
    logging.info("MAE: " + str(round(mean_absolute_error(y_test, y_pred), 2)))


def log_end_of_model():
    """Visually demarcate end of log info on a model."""
    logging.info("")
    logging.info("--------------------------------------")
    logging.info("")


# train-test split ------------------------------------------------------------
clean_data_path = './data/after_processing_rolling_sales.csv'
df_clean = pd.read_csv(clean_data_path)
X = df_clean.drop(columns=["sale_price"])
y = df_clean["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=random_seed
                                                    )

# Define models to be tuned----------------------------------------------------
scoring = "neg_mean_absolute_error"
models = []
models.append((
        "Lasso",
        ElasticNet(normalize=True, tol=0.1),
        [{"ttregressor__regressor__l1_ratio": [1],
          "ttregressor__regressor__alpha": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]}]
        )
)
models.append((
        "Ridge",
        ElasticNet(normalize=True, tol=0.1),
        [{"ttregressor__regressor__l1_ratio": [0],
          "ttregressor__regressor__alpha": [5e-3, 1e-3, 5e-4, 1e-4]}]
        )
)
models.append((
    "RF",
    RandomForestRegressor(random_state=random_seed),
    [{
        "ttregressor__regressor__max_features": [200, 300],
        "ttregressor__regressor__max_depth": [15],
        "ttregressor__regressor__n_estimators": [500]}]
        )
        )
models.append((
    "GB_lad",
    ensemble.GradientBoostingRegressor(random_state=random_seed,
                                       loss="lad",
                                       learning_rate=0.01,
                                       n_estimators=1500
                                       ),
    [{"ttregressor__regressor__max_features": [10, 50, 100]}]
))

# Tune and evaluate models-----------------------------------------------------
tuned_models = []
for name, model, grid in models:
    print("# Tuning hyper-parameters for {}".format(name))

    t0 = time.time()

    my_pipeline = sk.pipeline.Pipeline([
             ("scale", StandardScaler()),
             ("ttregressor",
              TransformedTargetRegressor(
                      regressor=model,
                      func=np.log,
                      inverse_func=np.exp
                      )
              )
    ])
    current_model = GridSearchCV(my_pipeline,
                                 grid,
                                 cv=3,
                                 scoring=scoring
                                 )
    current_model.fit(X_train, y_train)
    tuned_models.append((name, current_model.best_estimator_))
    y_pred = current_model.predict(X_test)

    t1 = time.time()

    # Log and plot results of tuning
    msg_time = "Tuning the {} model took {:.2f} seconds".format(name, t1 - t0)
    logging.info(msg_time)
    log_train_set_info_and_performance(current_model, name)
    log_test_set_performance(name, y_pred, y_test)
    log_end_of_model()
    plot_model_performance(y_pred=y_pred,
                           y_test=y_test,
                           model_name=name,
                           zoom=True
                           )
    plot_model_performance(y_pred=y_pred,
                           y_test=y_test,
                           model_name=name,
                           zoom=False
                           )

# Models that are not tuned----------------------------------------------------
# Linear Regression
lin_reg = Pipeline([("scaling", StandardScaler()),
                    ("ttregressor",
                     TransformedTargetRegressor(
                             regressor=LinearRegression(),
                             func=np.log,
                             inverse_func=np.exp
                             )
                     )
                    ])
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
log_end_of_model()
plot_model_performance(y_pred, y_test, "lin_reg")
plot_model_performance(y_pred, y_test, "lin_reg", zoom=True)

# LGBM
# create dataset for lightgbm using log-transform for selling price
# lgb_train = lgb.Dataset(X_train, np.log(y_train))
# lgb_test = lgb.Dataset(X_test, np.log(y_test))

# lgb_params = {
#     "task": "train",
#     "objective": "mae",
#     "boosting_type": "gbdt",
#     "num_leaves": 100,
#     "learning_rate": 0.01,
#     "feature_fraction": 0.9,
#     "bagging_fraction": 0.8,
#     "bagging_freq": 5,
#     "verbose": 0,
#     "seed": 42
# }

# t0 = time.time()
# print("# Tuning hyper-parameters for LGBM")
# evals_result = {}  # to record eval results for plotting
# gbm = lgb.train(lgb_params,
#                 lgb_train,
#                 num_boost_round=1500,
#                 # evals_result=evals_result,
#                 # early_stopping_rounds=5
#                 )
# t1 = time.time()
# msg_time = ("Tuning the Light GBM" +
#             "model took {} seconds.".format(str(round(t1 - t0, 2)))
#             )

# logging.info(msg_time)
# y_pred = np.exp(gbm.predict(X_test))
# logging.info("Score on the test set:")
# logging.info("R2 score: " + str(round(r2_score(y_test, y_pred), 3)))
# logging.info("MAE: " + str(round(mean_absolute_error(y_test, y_pred), 2)))

# plot_model_performance(y_pred, y_test, "lgbm")
# plot_model_performance(y_pred, y_test, "lgbm", zoom=True)
