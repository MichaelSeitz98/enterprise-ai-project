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


For this regression task, different models were trained, tuned and compared. All training and model related code can be found in the notebook `train_and_eval_models.ipynb`. To ensure reproducibility and comparability between models, model training was performed as a pipeline. For experimentation, such as finding suitable features, data set, data augmentation methods, model architecture, all runs are logged using MLFlow. 

- Linear regression 
- Lasso regression 
- Ridge regression 
- Elasticnet regression 
- Random Forest Regression 
- XGBoost Regressor
- Our own Benchmark 

All of these models are benchmarked against our own simple **benchmark model**. This baseline model predicts prices using only the living room information and the current average rental/purchase price per square metre in Würzburg. The benchmark automatically scrapes the current price from [wohnungsboerse.net/mietspiegel-Wuerzburg](https://www.wohnungsboerse.net/mietspiegel-Wuerzburg/2772), where it is updated every month. So the benchmark is always up to date. In the same way, for our 2nd use case, buying a house, we also have a dynamic benchmark, but with the average purchase price per square metre. This is also automatically taken from [wohnungsboerse.net/immobilienpreise-Wuerzburg](https://www.wohnungsboerse.net/immobilienpreise-Wuerzburg/2772).


### Data Augmentation 

As we did not have many data, we did use data augementation techniques to extend our data set. 
For this reason we used a Generative Adversial Generator (GAN) for tabular Data  `CTGAN`. It can be trained on a dataset and generate new data with similar characteristics and can extend the data amount in that way.  We build a complete pipeline, where the whole training and evaluation process can be extended with augemented data in method. The implementation was done also as pipeline, which runs the complete training and eval process, but adds augemented data. It can be seen here: [train_and_eval_modules.ipynb](https://github.com/MichaelSeitz98/enterprise-ai-project/tree/main/Immowelt/04_finetuning_approaches).

[Title](C:%255CUsers%255Cmichi%255CenterpriseAI_michi%255Centerprise-ai-project%255CImmmoWelt_Price_Guide%255Ctrain_and_eval_models.ipynb)

Important: the tabular GAN is always ONLY fed / trained with the training data. No information form validation and test dataset was used for generating the augemented extra sample. 


## Frontend Application

## Deployment to Cloud




## Outlook & Discussion



* extend data set data, regualary scraped. every week new scarping 
