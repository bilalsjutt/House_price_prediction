# House Price Prediction Model

A comprehensive machine learning project for predicting house prices using various regression algorithms and advanced feature engineering techniques.

## Overview

This project implements multiple machine learning models to predict house prices based on property characteristics, location data, and temporal features. The pipeline includes extensive data preprocessing, feature engineering, and model comparison to achieve optimal prediction accuracy.

## Dataset

The project uses a house price dataset (`data.csv`) containing the following features:
- **Property Details**: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition
- **Location Data**: street, city, state, zip, country
- **Temporal Data**: date (sale date)
- **Historical Data**: yr_built, yr_renovated
- **Target Variable**: price

### Dataset Statistics
- **Initial Shape**: 4,551 records after removing zero-price entries
- **Features**: 18 original columns
- **Target**: House price in USD

## Features

### Data Preprocessing
- **Missing Value Handling**: Drops rows with missing values
- **Outlier Detection**: Uses IQR method to identify and remove price outliers
- **Data Type Conversion**: Converts date column to datetime format

### Feature Engineering

#### Temporal Features
- Sale year, month, quarter, day of year, weekday
- Seasonal indicators (spring, summer, fall, winter)
- Peak season identification (April-June)

#### Property Features
- Total rooms (bedrooms + bathrooms)
- Bedroom-to-bathroom ratio
- Luxury property indicators based on price and room count

#### Location-Based Features
- City average price and property count statistics
- Country average price and property count statistics
- Street frequency encoding for high-cardinality categorical data

#### Advanced Features
- **Polynomial Features**: Interaction terms for key numerical variables
- **Target Encoding**: Location-based price statistics
- **Feature Selection**: SelectPercentile with f_regression (top 75% features)

### Models Implemented

1. **Random Forest Regressor**
   - 300 estimators, max_depth=20
   - Optimized hyperparameters for best performance

2. **Gradient Boosting Regressor**
   - 200 estimators, learning_rate=0.1
   - Subsample=0.8 for regularization

3. **Extra Trees Regressor**
   - 300 estimators with randomized splitting
   - Fast training with good generalization

4. **Ridge Regression**
   - Alpha=100 for regularization
   - Uses scaled and enhanced features

5. **Lasso Regression**
   - Alpha=1000 for feature selection
   - Automatic feature selection through L1 regularization

## Model Performance

Based on the test results, here are the key performance metrics:

### Lasso Regression (Example Results)
- **R² Score**: 0.6619 (66.19% variance explained)
- **RMSE**: $125,889.58
- **MAE**: $91,948.89

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Data Loading**:
   ```python
   dataset = pd.read_csv("path/to/data.csv")
   ```

2. **Run the Complete Pipeline**:
   ```python
   # The script handles all preprocessing, feature engineering, and model training
   python house_price_prediction.py
   ```

3. **Model Comparison**:
   The script automatically trains multiple models and stores results in `models_results` dictionary for easy comparison.

## Key Components

### Data Preprocessing Pipeline
- Zero price filtering
- Missing value handling
- Outlier detection and removal using IQR method
- Date feature extraction

### Feature Engineering Pipeline
- Temporal feature creation
- Location-based statistical features
- Polynomial feature generation
- Feature selection using statistical methods

### Model Training Pipeline
- Train-test split with proper data leakage prevention
- Separate preprocessing for train and test sets
- Cross-validation ready implementation
- Comprehensive evaluation metrics

## File Structure

```
project/
├── data.csv                 # Dataset file
├── house_price_prediction.py # Main script
└── README.md               # This file
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- warnings (built-in)

## Performance Considerations

- **Feature Scaling**: StandardScaler applied to numerical features
- **Memory Optimization**: Efficient data handling for large datasets
- **Parallel Processing**: n_jobs=-1 for ensemble methods
- **Cross-validation**: Ready for k-fold validation implementation

## Future Enhancements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Additional feature engineering (geographic features, neighborhood analysis)
- Deep learning models (Neural Networks)
- Model ensemble techniques
- Feature importance analysis and visualization

## Notes

- The script includes comprehensive error handling and progress tracking
- All models are stored in `models_results` dictionary for easy access
- Feature engineering is designed to prevent data leakage
- The pipeline is modular and can be easily extended with new models

## Author

This project demonstrates advanced machine learning techniques for real estate price prediction, combining traditional statistical methods with modern ensemble approaches.
