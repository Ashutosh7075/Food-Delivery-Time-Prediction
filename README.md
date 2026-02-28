# Food Delivery Time Prediction

## Project Overview
This project predicts food delivery time using machine learning techniques. It also classifies deliveries as fast or delayed based on operational and environmental factors such as distance, weather, traffic, and delivery personnel experience.

The project follows a complete machine learning workflow including:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Linear Regression
- Logistic Regression
- Model evaluation
- Business insights

---

## Dataset Description
The dataset contains 200 delivery records with the following features:

- Distance
- Weather Conditions
- Traffic Conditions
- Delivery Person Experience
- Order Priority
- Order Time
- Vehicle Type
- Restaurant Rating
- Customer Rating
- Order Cost
- Tip Amount
- Delivery Time (Target Variable)

---

## Data Preprocessing
- Removed irrelevant columns
- Handled missing values
- Applied one-hot encoding for categorical variables
- Standardized numerical features using StandardScaler

---

## Exploratory Data Analysis
- Generated descriptive statistics
- Created correlation heatmap
- Detected outliers using boxplot

Saved visualizations:
- heatmap.png
- boxplot.png

---

## Linear Regression Results
- MSE: 1021.93
- MAE: 27.18
- R²: -0.10

The regression model showed limited predictive performance.

---

## Logistic Regression Results
- Accuracy: 52.5%
- Precision: 60.6%
- Recall: 76.9%
- F1 Score: 67.8%

The classification model performed moderately well in detecting delayed deliveries.

---

## Business Insights
- Distance significantly impacts delivery time.
- High traffic increases delay probability.
- Delivery experience improves performance.
- Peak hours require better staffing strategies.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Future Improvements
- Use Random Forest or Gradient Boosting
- Incorporate real-time traffic data
- Increase dataset size
