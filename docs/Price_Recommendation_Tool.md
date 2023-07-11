# Price Recommendation Tool

Accurate price predictions for flat rents and house buys in Würzburg are crucial. They enable informed decision making, aid budget planning, support market analysis, ensure fair transactions, and reduce information asymmetry. Price predictions play a crucial role in facilitating well-informed choices and efficient transactions in Würzburg's real estate market.

To develop an ML system to enable this, we followed a structured process.  First, we scraped our own dataset by utilizing web scraping techniques. The details of the data extraction process can be found under the [data extraction](#Data-Extraction) section. Next, we conducted [exploratory data analysis](#Data Extraction :open_file_folder:) and performed [feature engineering](#Feature-Engineering) to prepare the dataset for modeling. We then proceeded to train and compare different models using the dataset. The [training](#Training-and-Evaluation) process and model comparisons were documented and tracked using MLFlow. Once we selected the best model, we deployed it to the cloud for scalability and accessibility. The deployment process is described under [deployment to cloud](#deployment-to-cloud). Finally, we developed a [frontend](#frontend) application to provide an intuitive user interface for interacting with the ML system.

## Data Extraction 


## Exploratory Data Analysis
An exploratory data analysis was conducted using `ydata-profiling` to generate insights about the distribution of the extracted dataset. The analysis focused on the flat rent data set, and the results can be accessed [here](https://michaelseitz98.github.io/enterprise-ai-project/eda-wue-rent-all.html).

The purpose of the exploratory data analysis was to gain an overview of the variables, identify missing values, assess class imbalance, and explore correlations among different variables. Statistical measures and visualizations were employed to understand the dataset's structure, uncover patterns, and identify potential issues. The analysis serves as a crucial step in the data exploration process, providing a foundation for informed decisions related to feature engineering and modeling.


## Model Training and Evaluation


## Deployment to Cloud


## Frontend Application


## Outlook & Discussion
