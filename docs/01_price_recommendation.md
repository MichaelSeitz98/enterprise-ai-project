# Flat Price Assistant 

Accurate price predictions for flat rents and house buys in Würzburg are crucial. They enable informed decision making, aid budget planning, support market analysis, ensure fair transactions, and reduce information asymmetry. Price predictions play a crucial role in facilitating well-informed choices and efficient transactions in Würzburg's real estate market.

To develop an ML system to enable this, we followed a structured process.  First, we scraped our own dataset by utilizing web scraping techniques. The details of the data extraction process can be found under the [data extraction](#data-extraction) section. Next, we conducted [exploratory data analysis](#exploratory-data-analysis) and performed [feature engineering](#feature-engineering) to prepare the dataset for modeling. We then proceeded to train and compare different models using the dataset. The [training](#model-training-and-evaluation) procemodel comparisons were documented and tracked using MLFlow. Once we selected the best model, we deployed it to the cloud for scalability and accessibility. The deployment process is described under [Architecture & Model Deployment](#architecture-and-model-deployment). Finally, we developed a [frontend](#frontend-application) application to provide an intuitive user interface for interacting with the ML system.

## Data Extraction & Preprocessing


### Exploratory Data Analysis
An exploratory data analysis was conducted using `ydata-profiling` to generate insights about the distribution of the extracted dataset. The analysis focused on the flat rent data set, and the results can be accessed publically via the following links. 

[EDA: basic dataset of flats to rent in Würzburg](https://michaelseitz98.github.io/enterprise-ai-project/eda-wue-rent-all.html)

The purpose of the exploratory data analysis was to gain an overview of the variables, identify missing values, assess class imbalance, and explore correlations among different variables. Statistical measures and visualizations were employed to understand the dataset's structure, uncover patterns, and identify potential issues. The analysis serves as a crucial step in the data exploration process, providing a foundation for informed decisions related to feature engineering and modeling. As the system can easily be adapted to the use case of predicting house purchase prices, we also extracted house data from Würzburg. A detailed insight into this can be seen [here](https://michaelseitz98.github.io/enterprise-ai-project/eda-wue-houses.html).

### Features Engineering

After the exploratory data analysis, we chose our final features to avoid overfitting. First, we looked at which features had a **high correlation** with the object feature. Then we looked at which features were **inbalanced**. For example, there is only one house with a bidet with a high price. Now the bidet is an indicator that only expensive houses have a bidet. So we remove this feature because only one apartment has it. We also looked at which feature the user can enter in our web application. The finale feature can be seen in this [notebook](https://github.com/MichaelSeitz98/enterprise-ai-project/blob/main/immowelt_price_guide/frontend/app.py).

### Preprocessing 

First we preprocessed all features that were **binary** with a binariser. For example, if an apartment has a specific feature such as a garden, the binariser sets the value of that feature to one. If the feature is not present in the apartment, it will get a zero as an indicator.
In the next step we will precode all **categorical** features with a one-hot-encoder, because we do not have ordinary features. In this [notebook](https://github.com/MichaelSeitz98/enterprise-ai-project/blob/main/immowelt_price_guide/train_and_eval_models.ipynb) and this [dictionary](https://github.com/MichaelSeitz98/enterprise-ai-project/tree/main/immowelt_price_guide/scrape_and_preprocess) you can find the necessary steps we took to prepare our features.

## Model Training and Evaluation

For this regression task, different models were trained, tuned and compared. The related code to the model  run can be be found in the notebook `train_and_eval_models.ipynb` and `model_functions.py`. To ensure reproducibility and comparability between models within differen setups, model training was performed as a pipeline. For experimentation, such as finding suitable features, data set, data augmentation methods, model architecture, all runs are logged using MLFlow. 

- Linear regression 
- Lasso regression 
- Ridge regression 
- Elasticnet regression 
- Random Forest Regression 
- XGBoost Regressor
- Our own (dynamically updated) Benchmark

All of these models are benchmarked against a **benchmark model**. This baseline model predicts prices using only the living room information and the current average rental/purchase price per square metre in Würzburg. The benchmark automatically scrapes the current price from [wohnungsboerse.net/mietspiegel-Wuerzburg](https://www.wohnungsboerse.net/mietspiegel-Wuerzburg/2772), where it is updated every month, so the benchmark is always up to date. In the same way, for buying a house the dynamic benchmark is using the average purchase price per square meter of Würzburg, scraped from [wohnungsboerse.net/immobilienpreise-Wuerzburg](https://www.wohnungsboerse.net/immobilienpreise-Wuerzburg/2772).


### Logging and Storing via MLFlow

Every different set-up of used features, used models and differently used hyperparameter was logged and compared to each other via `MLFlow`. All different runs aka experiment where tracked and evaluated there, see like a example model comparison. So, the best suitable model could be chose.  

![experiments](resources/mlflow_experiment_view_table.png)

If a model is chosen to be deployed for our productive systems, it can be registered to `model registry`. This s a centralized repository for managing and versioning machine learning models. We utilized it to track and store different versions of our models, enabling easy comparison and deployment. It streamlined our model management process and enables collaboration amongteam members . The Model Registry integrated seamlessly with our deployment pipeline, ensuring that the selected models can be deployed to our "Würzburger Mietpreis-Checker" application, by setting the stage to "production" and load it via API from the application. This allowed us to easily incorporate the latest models into our production application for rent price analysis in Würzburg.

![model_registry](resources/mlflow_model_registry.png)

### Data Augmentation

To expand our limited dataset, we employed data augmentation techniques. We utilized a Generative Adversarial Network (GAN) specifically designed for tabular data called `CTGAN`. This GAN can be trained on an existing dataset and generate new data instances that possess similar characteristics, effectively increasing the size of the dataset. 
To facilitate this process, we developed a comprehensive pipeline that integrates the training and evaluation procedures with augmented data. The implementation is available in the following Jupyter Notebook: [train_and_eval_modules.ipynb](https://github.com/MichaelSeitz98/enterprise-ai-project/blob/main/immowelt_price_guide/train_and_eval_models.ipynb).

Please note that the tabular GAN was exclusively trained using the training data. No information from the validation or test dataset was utilized for generating the augmented samples.
Although there is potential for data augmentation using `CTGAN`, our experiments clearly demonstrated that the mean absolute error (MAE) did not improve across any of the models. We generated additional rows ranging from 0 to 1000 for training purposes and evaluated their performance on "untouched" data. Obviously, the benchmark from 286 remained consistent in all experimental setups. For this reason, data augementation with CTGAN was not applied in the final system. 

![plot2](resources/syntetic_data_for_train_impact.png)


### Continous Learning / Retraining

We implemented a dynamic learning pipeline where the training base can be updated with the latest scraped data from Würzburg.
The complete retraining pipeline is also developed in the `train_and_eval.ipynb` notebook and schematically follows the process shown.

 ![retrain_process](resources/dynamic_retrain.png)

The newly trained models are evaluated on the same validation as before, so it is clear whether the new data improved the model or not. This method is also useful for extending the dataset over time, as the dataset is continuously extended. 


## Frontend Application
### User Frontend
We use Gradio as our frontend framework. `Gradio` is particularly good at applying models. To be able to predict a property price, we need a dataset that has the same requirements as our training, validation and test dataset. Therefore, the user has to enter his property characteristics in the front end. The next step is to generate the dataframe from this. After this step, we load our state of the art model to predict the rental price for the user. In this gif you can see how the user has to use our frontend application to get the predicted rental price for the apartment. Here you can watch how our [frontend](https://www.youtube.com/watch?v=4TPxXEXrMQA) is working



### Admin Frontend
In this frontend application, the admins of our website can scrap new data and automatically retrain the machine learning models. First the admin has to choose which models to retrain. Then the user can click on the button. Now our backend scraps new data and combines it with our old dataset. Our models can now be retrained. When the retraining process is finished, we can decide which models have improved and which model is now the best model to predict the price. This application is separate from our user frontend, it's just for us to retrain and visualise the performance of our models.

# Architecture and Model Deployment

This technical documentation provides an overview of the architecture and model deployment process for our ML application. The application leverages MLFlow for model training, management, and versioning, while the frontend is built using Gradio. The backend application is developed with FastAPI and hosted on Heroku, allowing for easy deployment of new models to the cloud.

## Architecture Overview

![Architecture](resources/architecture-flat-price-app.png)

The architecture of our ML application can be described as follows:

1. **MLFlow**: MLFlow is used for model training, tracking, and versioning. It runs on a local server, and all artifacts are stored in our repository. This enables every team member with access to the repository to create models, train them, and store different versions using MLFlow.

2. **Frontend (Gradio)**: The frontend of our application is built using Gradio. It collects the necessary data for the model to create predictions. Once the data is collected, the Gradio app sends a POST request to our FastAPI backend application.

3. **Backend (FastAPI)**: The FastAPI backend application provides various endpoints.. When a request is received at the `/predict` endpoint, the desired model is loaded and ready to make predictions. The dedicted folder is located under `/immowelt_price_guide/backend`. [Also find here](https://github.com/MichaelSeitz98/enterprise-ai-project/tree/main/immowelt_price_guide/backend). 

The `main.py` includes the FastAPI app and the following endpoints:
   -  `/predict` endpoint: This endpoint is used to make predictions. It receives a POST request from the frontend application, loads the desired model, and returns the prediction.
   - `/model-info` endpoint: This endpoint is used to retrieve information about the model. It receives a GET request from the frontend application and returns the model's input and output parameters as well as the name and some infos about the version. 
   - `/` root: This endpoint is used to check the health of the application. It receives a GET request from the frontend application and returns a status code of 200 if the application is running.


4. **Cloud Deployment (Heroku)**: The backend application is hosted on Heroku, making the endpoint permanently publicly available for everyone. The frontend and backend are designed to run independently of each other.

Find hosted available api documentation here: [https://flat-price-assistant-wue-9d5350c50d5c.herokuapp.com/docs](https://flat-price-assistant-wue-9d5350c50d5c.herokuapp.com/docs).

## Model Deployment Process

Deploying a new model to the cloud follows the following steps:

1. Set the new model in production using MLFlow. This ensures that the model is available and can be accessed by name.

2. Prepare the deployment cell: We have a preconfigured cell that performs the necessary steps for deployment. By simply setting a new name, the following actions are automated:

   1. Load the model by its name from the MLFlow server.
   2. Gather all the necessary information about the model, including input and output parameters, required Python version, and the required packages with their specific versions.
   3. Set the model, its details, and requirements to the backend folder of our repository. This folder serves as the foundation for the deployed model in the cloud.

3. Commit the changes to the main branch of our repository. Since Heroku is set up with our Git repository, pushing the changes triggers a new deployment on the cloud. The command `git push heroku main` is used for this purpose.

4. Heroku deployment: The deployment process is possible because we have a `heroku.yml` file placed in our root directory, which directs to the Dockerfile located in our backend folder. The Dockerfile defines the image to be built for the model in production. Once the container is built, it runs on the Heroku cloud.

By following these steps, a new model can be easily deployed to the cloud, ensuring that the latest version is available for use as well as all the required dependencies. This makes sure to not run into version or dependency issues when deploying a new model.

## Conclusion

The architecture and model deployment process described in this documentation provide a clear and readable overview of how our ML application is structured and how new models can be deployed to the cloud using MLFlow, Gradio, FastAPI, and Heroku.

## Outlook & Discussion

* **Explainable AI** - We have not yet implemented explainable AI in our frontend. In this step, we want to be able to explain to the user why the model predicted this price. This is something we will implement in the future. In this picture you can see which features are decisive for your individual price prediction. We use the ``shape waterfall`` method to explain the prediction.
![shap waterfall](resources/shap_waterfall_example.png)


