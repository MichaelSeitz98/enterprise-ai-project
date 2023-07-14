import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import shap
from ctgan import CTGAN

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")


def determineHighCorrCols(df):
    df.columns = [
        re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), col)
        for col in df.columns
    ]
    df.columns = [
        col.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
        for col in df.columns
    ]
    important_num_cols = list(
        df.corr()["Object_price"][
            (df.corr()["Object_price"] > 0.20) | (df.corr()["Object_price"] < -0.20)
        ].index
    )
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    important_cols = important_num_cols + cat_cols + ["ConstructionYear"] + ["ZipCode"]
    print(important_cols)
    return important_cols


def preprocess_data_for_model(df, feature_set):
    print(df.columns)
    print(f"Used feature set for preprocessing:{feature_set}")
    df.columns = [
        re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), col)
        for col in df.columns
    ]
    df.columns = [
        col.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
        for col in df.columns
    ]
    df = df.replace('""', np.nan)
    df = df.dropna()
    df["LivingSpace"] = df["LivingSpace"].astype(float)
    df["Rooms"] = df["Rooms"].astype(float)
    df["ZipCode"] = df["ZipCode"].astype(str)
    df["LivingSpace"] = df["LivingSpace"].astype(float)
    df = df[feature_set]
    df = df.reindex()
    df = df.reset_index(drop=True)
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=cat_cols)
    return df


def data_split(df, train_size=0.8, random_state=42):
    y = df["Object_price"]
    X = df.drop("Object_price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=train_size, random_state=random_state
    )

    X_train.to_excel("data/X_train.xlsx")
    X_val.to_excel("data/X_val.xlsx")
    X_test.to_excel("data/X_test.xlsx")
    y_train.to_excel("data/y_train.xlsx")
    y_val.to_excel("data/y_val.xlsx")
    y_test.to_excel("data/y_test.xlsx")

    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_train.to_excel("data/df_train.xlsx")
    df_val.to_excel("data/df_val.xlsx")
    df_test.to_excel("data/df_test.xlsx")

    return X_train, y_train, X_val, y_val, X_test, y_test


def scrape_avg_rental_prices():
    url = "https://www.wohnungsboerse.net/mietspiegel-Wuerzburg/2772"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    script_tag = soup.find("script", string=lambda text: "pdfData" in text)
    rental_price = 0
    if script_tag:
        script_content = script_tag.string
        start_index = script_content.find("avg_rent_price: ") + len("avg_rent_price: '")
        end_index = script_content.find("',", start_index)
        rental_price = script_content[start_index:end_index]
        rental_price = (
            rental_price.replace("€/m2", "").replace(".", "").replace(",", ".")
        )
        rental_price = rental_price.strip()
        rental_price = float(rental_price)
        print(f"Extrcated rental price per square meter via scraper: {rental_price}")
    else:
        print("The script tag containing the rental price was not found.")
    return rental_price


def scrape_avg_buy_prices():
    url = "https://www.wohnungsboerse.net/immobilienpreise-Wuerzburg/2772"
    response = requests.get(url)
    response.raise_for_status(
    )
    soup = BeautifulSoup(response.text, "html.parser")
    p_element = soup.find("p", class_="mb-8")
    buy_price = 0
    if p_element:
        pattern = r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?€/m²"
        match = re.search(pattern, p_element.text)
        if match:
            buy_price = match.group()
            buy_price = buy_price.replace("€/m²", "").replace(".", "").replace(",", ".")
            print(f"Extrcated buy price per square meter via scraper: {buy_price}")
        else:
            print("Price not found")
    else:
        print("The element ontaining the buy price was not found.")
    return buy_price


def baseline_rent(val_X, val_y, runname="baseline_rent"):
    avg_price_per_sqm_rent = scrape_avg_rental_prices()
    print(f"Average rental price per sqm: {avg_price_per_sqm_rent}")
    return avg_price_per_sqm_rent


def baseline_buy(X_val, y_val, runname="baseline_buy"):
    avg_price_per_sqm_buy = scrape_avg_buy_prices()
    print(f"Average rental price per sqm: {avg_price_per_sqm_buy}")

    baseline_preds = X_val["LivingSpace"] * avg_price_per_sqm_buy
    baseline_mae = mean_absolute_error(y_val, baseline_preds)
    baseline_r2 = r2_score(y_val, baseline_preds)
    baseline_mse = mean_squared_error(y_val, baseline_preds)

    with mlflow.start_run(run_name=runname):
        mlflow.log_metric("mse", baseline_mse)
        mlflow.log_metric("mae", baseline_mae)
        mlflow.log_metric("r2", baseline_r2)

    print(f"Baseline Mae: {baseline_mae}")
    print(f"Baseline MSE: {baseline_mse}")
    print(f"Baseline R2 Score: {baseline_r2}")

    return avg_price_per_sqm_buy, baseline_mae, baseline_mse, baseline_r2

def train_and_eval_linear(X_train, y_train, X_val, y_val, runname="linear-regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_and_eval_lasso(X_train, y_train, X_val, y_val, runname="lasso-regression"):
    model = Lasso()
    model.fit(X_train, y_train)
    return model

def train_and_eval_ridge(X_train, y_train, X_val, y_val, runname="ridge-regression"):
    model = Ridge()
    model.fit(X_train, y_train)
    return model

def train_and_eval_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators=50,
    random_state=0,
    run_name="random-forest",
):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_and_eval_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    run_name="xgb",
    early_stopping_rounds=30,
    max_depth=6,
    n_estimators=1000,
):
    model = xgb.XGBRegressor(
        eval_metric=["rmse", "mae"],
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        max_depth=max_depth,
        n_estimators=n_estimators,
    )
    model.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)], verbose=True)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_train)
    # shap.plots.waterfall(shap_values[6])
    # plt.savefig("waterfall_0.png", bbox_inches="tight")
    # shap.plots.waterfall(shap_values[5])
    # plt.savefig("waterfall_1.png", bbox_inches="tight")
    # shap.plots.waterfall(shap_values[9])
    # plt.savefig("waterfall_2.png", bbox_inches="tight")
    # shap.plots.waterfall(shap_values[10])
    # plt.savefig("waterfall_3.png", bbox_inches="tight")
    # shap.plots.beeswarm(shap_values)
    # plt.savefig("beeswarm.png", bbox_inches="tight")
    return model


def train_and_eval_elasticnet(X_train, y_train, X_val, y_val, runname="elasticNet"):
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model



def pipeline_from_extracted(df, feature_set, model_name="lasso"):
    mlflow.end_run()
    model = None
    X, y = preprocess_data(df, feature_set)
    print("Done with preprocessing")
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y)
    print("Done with data split")

    if model_name == "xgb":
        mlflow.xgboost.autolog()
    else:
        mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=model_name):
        model, mae, mse, r2, mae_train, mse_train, r2_train = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        if model_name == "lasso":
            model = train_and_eval_lasso(X_train, y_train, X_val, y_val)
        elif model_name == "ridge":
            model = train_and_eval_ridge(X_train, y_train, X_val, y_val)
        elif model_name == "rf":
            model = train_and_eval_rf(X_train, y_train, X_val, y_val)
        elif model_name == "xgb":
            model = train_and_eval_xgb(X_train, y_train, X_val, y_val)
        elif model_name == "elasticnet":
            model = train_and_eval_elasticnet(X_train, y_train, X_val, y_val)
        elif model_name == "linear":
            model = train_and_eval_linear(X_train, y_train, X_val, y_val)
        elif model_name == "baseline-rent":
            avg_price = baseline_rent(X_val, y_val)
            baseline_preds = X_val["LivingSpace"] * avg_price
            baseline_preds_test = X_test["LivingSpace"] * avg_price
            mlflow.log_metric("mae", mean_absolute_error(y_val, baseline_preds))
            mlflow.log_metric("mse", mean_squared_error(y_val, baseline_preds))
            mlflow.log_metric("r2", r2_score(y_val, baseline_preds))
            mlflow.log_metric(
                "mae_test", mean_absolute_error(y_test, baseline_preds_test)
            )
            mlflow.log_metric(
                "mse_test", mean_squared_error(y_test, baseline_preds_test)
            )
            mlflow.log_metric("r2_test", r2_score(y_test, baseline_preds_test))
            return model, mae, mse, r2, mae_train, mse_train, r2_train
        else:
            print(
                "Model not found. Model_name must be 'lasso', 'ridge', 'rf', 'xgb', 'elasticnet', 'linear', 'baseline_buy' or 'baseline_rent' or conigure the pipeline manually."
            )

        pred_train = model.predict(X_train)
        preds = model.predict(X_val)
        pred_test = model.predict(X_test)

        mlflow.log_metric("mae_test", mean_absolute_error(y_test, pred_test))
        mlflow.log_metric("mse_test", mean_squared_error(y_test, pred_test))
        mlflow.log_metric("r2_test", r2_score(y_test, pred_test))

        mlflow.log_metric("mae_train", mean_absolute_error(y_train, pred_train))
        mlflow.log_metric("mse_train", mean_squared_error(y_train, pred_train))
        mlflow.log_metric("r2_train", r2_score(y_train, pred_train))

        mlflow.log_metric("mae", mean_absolute_error(y_val, preds))
        mlflow.log_metric("mse", mean_squared_error(y_val, preds))
        mlflow.log_metric("r2", r2_score(y_val, preds))

    print("Done with train")
    mlflow.end_run()
    return model, mae, mse, r2, mae_train, mse_train, r2_train



