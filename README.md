# Traffic Prediction AI

## Project Overview
This project aims to build an end-to-end AI-powered traffic prediction system using the Kaggle Traffic Prediction dataset. The system will provide accurate travel time predictions and route optimization for citizens, efficient routing for delivery services, and traffic pattern analysis for city officials.

## Dataset
The dataset used for this project is sourced from Kaggle and contains traffic data that will be utilized for training and evaluating various machine learning models.

## Project Structure
```
traffic-prediction-ai
├── data
│   └── raw
├── notebooks
│   └── exploratory_analysis.ipynb
├── src
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction_service.py
├── app
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow
1. **Data Loading**: Load the raw traffic dataset from the `data/raw` directory.
2. **Data Cleaning**: Clean the dataset by handling missing values and preparing it for analysis.
3. **Exploratory Data Analysis (EDA)**: Perform EDA to visualize traffic trends and patterns.
4. **Feature Engineering**: Extract relevant features from the dataset for model training.
5. **Model Training**: Train multiple models including Linear Regression, Random Forest, XGBoost, and LSTM.
6. **Model Evaluation**: Evaluate the trained models using metrics such as RMSE, MAE, and R².
7. **Prediction Service**: Create a service to load the best model and make predictions based on user input.
8. **Streamlit App**: Develop a web application to allow users to input parameters and visualize predictions.

## Application Relevance
This traffic prediction system is particularly relevant for urban areas like Indore, where efficient traffic management can significantly improve travel times and reduce congestion.

## Installation
To install the required packages, run:
```
pip install -r requirements.txt
```
📍 Application Relevance to Indore

Indore (population 3.2M) faces heavy congestion during peak hours (7–10 AM, 5–8 PM).
Our solution can directly address these issues:

👥 Citizens

Time loss: 45 min/day → AI reduces commute by 20%

Fuel costs: Save ₹800/month per commuter

Stress: Less congestion = fewer road rage incidents

🚚 Delivery Services

Late deliveries: Currently 22% delayed

Costs: Extra ₹15/order fuel wasted

Satisfaction: On-time performance boosts ratings

🏛️ City Officials

Emergency delay: Ambulance avg 18 mins → target <10 mins

Revenue loss: ₹25 crore/year due to traffic delays

Infrastructure strain: 12% roads over capacity

## flow chart link
https://app.eraser.io/workspace/UiG2lRowZqvXxKtLeETw?origin=share

## Usage
To run the Streamlit web application, navigate to the `app` directory and execute:
```
streamlit run streamlit_app.py


```
## demo
https://docs.google.com/videos/d/1pVEJLCnNM_Bu7QyAn5Res5HnyVdWSV4JEqZdQIjUtfM/edit?usp=sharing

## License
This project is licensed under the MIT License.
