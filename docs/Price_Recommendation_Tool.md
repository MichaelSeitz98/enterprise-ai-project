# Price Recommendation Tool

Accurate price predictions for flat rents and house buys in Würzburg are crucial. They enable informed decision making, aid budget planning, support market analysis, ensure fair transactions, and reduce information asymmetry. Price predictions play a crucial role in facilitating well-informed choices and efficient transactions in Würzburg's real estate market.

To develop an ML system to enable this, we followed a structured process.  First, we scraped our own dataset by utilizing web scraping techniques. The details of the data extraction process can be found under the [data extraction](#data-extraction) section. Next, we conducted [exploratory data analysis](#exploratory-data-analysis) and performed [feature engineering](#feature-engineering) to prepare the dataset for modeling. We then proceeded to train and compare different models using the dataset. The [training](#model-training-and-evaluation) procemodel comparisons were documented and tracked using MLFlow. Once we selected the best model, we deployed it to the cloud for scalability and accessibility. The deployment process is described under [deployment to cloud](#deployment-to-cloud). Finally, we developed a [frontend](#frontend-application) application to provide an intuitive user interface for interacting with the ML system.

## Data Extraction 


## Exploratory Data Analysis
An exploratory data analysis was conducted using `ydata-profiling` to generate insights about the distribution of the extracted dataset. The analysis focused on the flat rent data set, and the results can be accessed publically via the following links. 

* [EDA: basic dataset of flats to rent in Würzburg](https://michaelseitz98.github.io/enterprise-ai-project/eda-wue-rent-all.html)
* [EDA: basic dataset of houses to buy in Würzburg](https://michaelseitz98.github.io/enterprise-ai-project/eda-wue-houses.html)

The purpose of the exploratory data analysis was to gain an overview of the variables, identify missing values, assess class imbalance, and explore correlations among different variables. Statistical measures and visualizations were employed to understand the dataset's structure, uncover patterns, and identify potential issues. The analysis serves as a crucial step in the data exploration process, providing a foundation for informed decisions related to feature engineering and modeling.


## Model Training and Evaluation

### Model Selection 

For this regression task, different models where trained, tuned and compared to each other. All training and model related code can be find in notebook `train_and_eval_models.ipynb`. To ensure reproducability and compariblity between the models, model training was executed as pipeline. For experiemnt, like finding suitable featres, date set, data augmentation methods, model archiekture, all runs are logged with MLFlow. 

- Linear Regression 
- Lasso Regression 
- Ridge Regression 
- Elasticnet Regression 
- Random Forest Regression 
- XGBoost Regressor

All of these models are benchmarked agaisnst our own simple benchmark model. This baseline model predicts the prices just on the living room information and the current average rent / buy price per square meter in Würzburg. The benchmark automatically scrapes the current price from [wohnungsboerse.net/mietspiegel-Wuerzburg](https://www.wohnungsboerse.net/mietspiegel-Wuerzburg/2772), where it is updated every month. So the benchmark is always up to date. In the same way, for our 2nd use case of buying houses, we have also a dynamicall benchmark, but with average buying price per square meter. Also scaped automatically from [wohnungsboerse.net/immobilienpreise-Wuerzburg](https://www.wohnungsboerse.net/immobilienpreise-Wuerzburg/2772). 



## Deployment to Cloud


## Frontend Application


## Outlook & Discussion
