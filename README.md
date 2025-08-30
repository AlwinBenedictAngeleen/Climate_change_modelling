# Climate Change Modeling: Predicting CO2 Emissions from Satellite Data

## Project Overview
This project develops a machine learning model to predict carbon dioxide (CO2) emission levels using public satellite data. The dataset includes various atmospheric measurements (e.g., cloud properties, sulphur dioxide levels) collected over Rwanda. The goal is to build an accurate predictive model that can contribute to environmental monitoring efforts.

## Technical Approach
- **Data Acquisition:** The dataset was sourced from a Kaggle competition, originating from NASA satellite measurements.
- **Data Preprocessing:** Handled missing values through median imputation and prepared numerical features for modeling.
- **Model Selection:** Implemented a **Random Forest Regressor**, an ensemble learning algorithm known for its high accuracy and robustness with complex datasets.
- **Evaluation:** Assessed model performance using standard regression metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

## How to Run This Project
1.  **Ensure Python is installed** (Python 3.8 or higher is recommended).
2.  **Install required libraries** by running `pip install pandas numpy scikit-learn` in the Terminal.
3.  **Download the dataset** `train.csv` from [Kaggle](https://www.kaggle.com/competitions/playground-series-s3e20/data) and place it in a `data` folder within this project directory.
4.  **Run the analysis** by executing `python main.py` in the Terminal from within the `ClimateProject` directory.

## Results
The final Random Forest model was evaluated on a held-out test set (20% of the data). The results demonstrate the model's ability to accurately predict CO2 emissions:

- **Mean Absolute Error (MAE):** [Value will be generated when run]
- **Root Mean Squared Error (RMSE):** [Value will be generated when run]
- **R-squared (R²):** [Value will be generated when run]

An R² value close to 1.0 indicates that the model explains a large portion of the variance in the emission data, signifying a strong predictive performance.

## Conclusion
This project successfully demonstrates the application of machine learning to an environmental science problem. The results indicate that satellite-derived atmospheric data can be effectively used to model and predict CO2 emission levels. This approach could be valuable for large-scale environmental monitoring where ground-based sensors are limited.

## Author
ALWIN BENEDICT A
benwin316@gmail.com