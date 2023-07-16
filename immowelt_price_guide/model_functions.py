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
from tqdm import tqdm
import time
# from ctgan import CTGAN
import plotly.express as px
import plotly.express as px

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
import gradio as gr
from enum import Enum
from preprocessing_methods import *
from scrape_and_preprocess.apify_scrap import *
from datetime import datetime
import mlflow
import pickle
from model_functions import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    response.raise_for_status()
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
        print("The element containing the buy price was not found.")
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


class ImmoWeltUrls(Enum):
    BUY_FLATS_WUE_10km = "https://www.immowelt.de/liste/wuerzburg/wohnungen/kaufen?d=true&r=10&sd=DESC&sf=RELEVANCE&sp=1"
    # add price range to avoid "consulting"-offers without named price
    BUY_HOUSES_WUE_10km = "https://www.immowelt.de/liste/wuerzburg/haeuser/kaufen?d=true&pma=10000000&pmi=10&r=10&sd=DESC&sf=RELEVANCE&sp=1"
    RENT_FLATS_WUE_10km = "https://www.immowelt.de/liste/wuerzburg/wohnungen/mieten?d=true&r=10&sd=DESC&sf=RELEVANCE&sp=1"
    RENT_HOUSES_WUE_10km = "https://www.immowelt.de/liste/wuerzburg/haeuser/mieten?d=true&r=10&sd=DESC&sf=RELEVANCE&sp=1"


def getFeatureSetApp():
    return [
        "Object_price",
        "LivingSpace",
        "ZipCode",
        "Rooms",
        "altbau_(bis_1945)",
        "balkon",
        "barriefrei",
        "dachgeschoss",
        "einbaukueche",
        "neubau",
        "parkett",
        "stellplatz",
        "bad/wc_getrennt",
        "personenaufzug",
        "garten",
        "garage",
        "renoviert",
        "terrasse",
        "wanne",
        "zentralheizung",
        "abstellraum",
        "ferne",
        "fussbodenheizung",
        "gartennutzung",
        "kelleranteil",
    ]


def evaluate_model(model, X_train_recent, y_train_recent, X_val, y_val, X_test, y_test):
    pred_train = model.predict(X_train_recent)
    preds = model.predict(X_val)
    pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train_recent, pred_train)
    mse_train = mean_squared_error(y_train_recent, pred_train)
    r2_train = r2_score(y_train_recent, pred_train)

    mae_test = mean_absolute_error(y_test, pred_test)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    mae_val = mean_absolute_error(y_val, preds)
    mse_val = mean_squared_error(y_val, preds)
    r2_val = r2_score(y_val, preds)

    mlflow.log_metric("mae_test", mae_test)
    mlflow.log_metric("mse_test", mse_test)
    mlflow.log_metric("r2_test", r2_test)
    mlflow.log_metric("mae_train", mae_train)
    mlflow.log_metric("r2_train", mae_train)
    mlflow.log_metric("mse_train", mae_train)
    mlflow.log_metric("mae", mae_val)
    mlflow.log_metric("mse", mse_val)
    mlflow.log_metric("r2", r2_val)
    return (
        mae_val,
        mse_val,
        r2_val,
        mae_test,
        mse_test,
        r2_test,
        mae_train,
        mse_train,
        r2_train,
    )


def decode_col_names(df):
    df.columns = [
        re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), col)
        for col in df.columns
    ]
    df.columns = [
        col.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
        for col in df.columns
    ]
    return df


def getFeatureSetApp():
    return [
        "Object_price",
        "LivingSpace",
        "ZipCode",
        "Rooms",
        "altbau_(bis_1945)",
        "balkon",
        "barriefrei",
        "dachgeschoss",
        "einbaukueche",
        "neubau",
        "parkett",
        "stellplatz",
        "bad/wc_getrennt",
        "personenaufzug",
        "garten",
        "garage",
        "renoviert",
        "terrasse",
        "wanne",
        "zentralheizung",
        "abstellraum",
        "ferne",
        "fussbodenheizung",
        "gartennutzung",
        "kelleranteil",
    ]


def gradio_retrain_with_added_data(
    xgb, ridge, rf, elasticnet, linear, lasso, baseline, limit, progress=gr.Progress()
):
    progress(0.01, desc="Start pipeline")
    time.sleep(1)
    model_list = []
    if xgb:
        model_list.append("xgb")
    if ridge:
        model_list.append("ridge")
    if rf:
        model_list.append("rf")
    if elasticnet:
        model_list.append("elasticnet")
    if linear:
        model_list.append("linear")
    if lasso:
        model_list.append("lasso")
    if baseline:
        model_list.append("baseline-rent")

    result_df = trigger_retraining_with_added_data(
        limit=limit, model_list=model_list, progress=progress
    )

    progress(0.95, desc="Preparing model comparison")
    time.sleep(1.5)
    print("Done with retraining: ", result_df)

    print("Save results to excel")
    result_df.to_excel("retraining_results.xlsx")
    print("Done with saving results to excel")

    plot = px.bar(
        result_df,
        x="model",
        y="mae",
        title="Modellperformance mit erweitereten Trainingsdaten",
    )

    color_scale = px.colors.sequential.Greens[::-1] + px.colors.sequential.Reds
    plot = px.bar(
        result_df,
        x="model",
        y="mae",
        title="Modellperformance mit aktuellen Trainingsdaten",
        color="mae",
        color_continuous_scale=color_scale,
    )
    print("Done with plotting: ", plot)
    progress(0.99, desc="Done with pipeline")
    time.sleep(0.5)
    return result_df, gr.update(value=plot, visible=True)


def trigger_retraining_with_added_data(
    limit=3,
    model_list=["baseline-rent", "xgb", "ridge", "rf", "elasticnet", "linear", "lasso"],
    progress=gr.Progress(),
):
    url = "https://www.immowelt.de/liste/wuerzburg/wohnungen/mieten?d=true&r=10&sd=DESC&sf=RELEVANCE&sp=1"
    progress(0.05, desc=f"Scraping the first {int(limit)} pages from {url}")
    feature_set = getFeatureSetApp()
    print(url)
    print("started")
    retrain_data = get_dataset_items(url, limit)
    print("Retraining data successfully scraped.")

    write_data_to_excel(retrain_data, "data/retrain_train_data.xlsx")
    print(
        "Retraining data successfully written to excel under data/retrain_train_data.xslx"
    )

    progress(0.30, desc=f"Scraping done. Raw data preprocessing of new data...")
    time.sleep(1.5)

    new_df = pd.read_excel(r"data/retrain_train_data.xlsx")
    new_df = preprocess_data(new_df)

    print("Done with raw preprocessing.")
    new_df.to_excel("data/retrain_train_data_preprocessed.xlsx", index=False)

    ############################# Scraping done ##################################

    X_val = pd.read_excel("data/X_val.xlsx")
    X_val = X_val.drop("Unnamed: 0", axis=1)
    y_val = pd.read_excel("data/y_val.xlsx")
    y_val = y_val.drop("Unnamed: 0", axis=1)
    X_test = pd.read_excel("data/X_test.xlsx")
    X_test = X_test.drop("Unnamed: 0", axis=1)
    y_test = pd.read_excel("data/y_test.xlsx")
    y_test = y_test.drop("Unnamed: 0", axis=1)

    new_df = pd.read_excel("data/retrain_train_data_preprocessed.xlsx")
    new_df = decode_col_names(new_df)

    for feature in feature_set:
        if feature not in new_df.columns:
            new_df[feature] = 0

    train_recent = pd.read_excel("data/train_recent.xlsx")
    print("old shape of train_recent", train_recent.shape)
    old_shape = train_recent.shape[0]
    new_df = preprocess_data_for_model(new_df, feature_set)
    train_recent = pd.concat([train_recent, new_df], axis=0)
    train_recent = train_recent.drop_duplicates()
    print("new shape of train_recent", train_recent.shape)
    new_shape = train_recent.shape[0]
    amount_new_data = new_shape - old_shape
    progress(
        0.35,
        desc=f"{amount_new_data} new entries added to training base.",
    )
    time.sleep(3)
    print("Retraining data successfully added to training data.")
    train_recent = train_recent.fillna(0)
    train_recent.to_excel("data/train_recent_add.xlsx", index=False)

    y_train_recent = train_recent["Object_price"]
    X_train_recent = train_recent.drop(["Object_price"], axis=1)

    now = datetime.now()
    print(
        "!!!--------------------------------------START RETRAINING----------------------------------------------!!!"
    )

    progress(
        0.4,
        desc=f"Start retraining of models with new data...",
    )
    model = None
    experiment_name = f"retraining_{now.strftime('%Y-%m-%d_%H-%M')}"
    mlflow.set_experiment(experiment_name)

    results = pd.DataFrame()

    for model_name in progress.tqdm(model_list, desc=f"Retrain models and log to MLFlow: {experiment_name}"):
        if model_name == "xgb":
            mlflow.xgboost.autolog()
        else:
            mlflow.sklearn.autolog()

        with mlflow.start_run(run_name=f"{model_name}"):
            if model_name == "xgb":
                print("XGB------")
                print(f"train{X_train_recent.shape}")
                print(f"val:{X_val.shape}")
                print(f"y_train:{y_train_recent.shape}")
                print(f"y_val:{y_val.shape}")
                model = train_and_eval_xgb(X_train_recent, y_train_recent, X_val, y_val)
            elif model_name == "lasso":
                print("LASSO------")
                model = train_and_eval_lasso(
                    X_train_recent, y_train_recent, X_val, y_val
                )
            elif model_name == "ridge":
                print("RIDGE------")
                model = train_and_eval_ridge(
                    X_train_recent, y_train_recent, X_val, y_val
                )
            elif model_name == "rf":
                print("RF------")
                model = train_and_eval_rf(X_train_recent, y_train_recent, X_val, y_val)
            elif model_name == "elasticnet":
                print("ELASTICNET------")
                model = train_and_eval_elasticnet(
                    X_train_recent, y_train_recent, X_val, y_val
                )
            elif model_name == "linear":
                print("LINEAR------")
                model = train_and_eval_linear(
                    X_train_recent, y_train_recent, X_val, y_val
                )
            elif model_name == "baseline-rent":
                print("BASELINE-RENT------")
                avg_price = baseline_rent("", "")
                baseline_preds_val = X_val["LivingSpace"] * avg_price
                baseline_preds_test = X_test["LivingSpace"] * avg_price
                baseline_mae = mean_absolute_error(y_val, baseline_preds_val)
                baseline_r2 = r2_score(y_val, baseline_preds_val)
                baseline_mse = mean_squared_error(y_val, baseline_preds_val)
                baseline_mae_test = mean_absolute_error(y_test, baseline_preds_test)
                baseline_r2_test = r2_score(y_test, baseline_preds_test)
                baseline_mse_test = mean_squared_error(y_test, baseline_preds_test)
                print(f"Baseline Mae: {baseline_mae}")
                mlflow.log_metric("mse", baseline_mse)
                mlflow.log_metric("mae", baseline_mae)
                mlflow.log_metric("r2", baseline_r2)
                mlflow.log_metric("mse_test", baseline_mse_test)
                mlflow.log_metric("mae_test", baseline_mae_test)
                mlflow.log_metric("r2_test", baseline_r2_test)

                print(f"Baseline Mae: {baseline_mae}")
                print(f"Baseline MSE: {baseline_mse}")
                print(f"Baseline R2 Score: {baseline_r2}")

                results = results.append(
                    {
                        "model": model_name,
                        "mae": baseline_mae,
                        "mse": baseline_mse,
                        "r2": baseline_r2,
                        "mae_test": baseline_mae_test,
                        "mse_test": baseline_mse_test,
                        "r2_test": baseline_r2_test,
                    },
                    ignore_index=True,
                )
                results = results.round(2)

            else:
                print("Model not found.")

            print(f"Training {model_name} model done...")
            print(f"---EVALUATION AND LOGGING TO MLFLOW------ {model_name}")

            if model_name != "baseline-rent":
                (
                    mae_val,
                    mse_val,
                    r2_val,
                    mae_test,
                    mse_test,
                    r2_test,
                    mae_train,
                    mse_train,
                    r2_train,
                ) = evaluate_model(
                    model, X_train_recent, y_train_recent, X_val, y_val, X_test, y_test
                )
                results = results.append(
                    {
                        "model": model_name,
                        "mae": mae_val,
                        "mse": mse_val,
                        "r2": r2_val,
                        "mae_test": mae_test,
                        "mse_test": mse_test,
                        "r2_test": r2_test,
                        "mae_train": mae_train,
                        "mse_train": mse_train,
                        "r2_train": r2_train,
                    },
                    ignore_index=True,
                )
                results = results.round(2)

        mlflow.end_run()
    return results
