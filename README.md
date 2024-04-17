# Flight Price Predictor 

We aim to develop a machine learning application that predicts the price of a flight ticket given the starting and end destinations and other factors. 

This will aid ticket purchasers in determining whether their ticket is overpriced, as well as giving an indicator of the most ideal time window for purchase.

# Upload Data to Google Cloud

Due to the large size of the data, we upload the data directly from Kaggle to GCS without installing locally. This is done through running the following script 

[Upload Data to GCS from Kaggle](https://www.kaggle.com/code/combustingrats/copy-kaggle-data-to-google-cloud-services)

Here, we must first sync to our gcloud account through the "Add-Ons" tab, as well as add our flights data as an input through the "+Add Input" option on the right toolbar

# Cluster Setup and Specifications

Below are the main cluster specifications we utilized to complete our project

- Single node (1 master, 0 workers)
- 2.0 (Ubuntu 18.04 LTS, Hadoop 3.2, Spark 3.1)
- Remember to enable optional components of Anaconda and Jupyter Notebook to run scripts
- n2-standard-4 for both manager and worker nodes. 2 worker nodes.
- cloud storage bucket set to the bucket created in the above step

# Sequence of Code Execution

After the data is on GCS, pull the code from this repo and you may now run the scripts in the following order

1. __EDA__ : eda_with_spark.ipynb. This runs the EDA script with visualizations used in our report/presentation
2. __Preprocess Variables__: preprocess_variables.ipynb. This will save a new parquet file which is the final dataset you will load in subsequent scripts.
3. __Baseline Model__: baseline_model.ipynb. This is the baseline model benchmark before further fine tuning/optimization
4. __Tuned Models__: hyperparameter_tuning.ipynb (Random Forest), gbt_model_tuning_ATL_.ipynb (GBT), linear_model_tuning-2.ipynb (Linear Regression), xgboost_model_tuning.ipynb (XGBoost)  These are our fine tuned models and their metrics found through grid search.
5. __User Interface__: stream_lit directory. This is a directory of scripts to open a streamlit based web application for our project. streamlit_demo.py is the file to run to create the interface.

# Resources 

Link to data
https://www.kaggle.com/datasets/dilwong/flightprices
