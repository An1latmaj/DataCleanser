# EDA and Preprocessing Guide

This guide outlines best practices for exploratory data analysis (EDA) and preprocessing using the DataCleanser utility.

## Table of Contents
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Complete Pipeline Example](#complete-pipeline-example)

## Exploratory Data Analysis

EDA is a critical first step in any data science project. The DataCleanser utility streamlines this process with the following methods:

### Basic Information

```python
# Get overview of the dataset
cleanser.get_basic_info()
```

This displays:
- DataFrame shape (rows and columns)
- Data types for each column
- Missing value count
- Basic descriptive statistics

### Visualizing Distributions

```python
# Visualize numeric feature distributions
cleanser.visualize_distributions()
```

This generates:
- Histograms for numeric features
- Box plots to identify potential outliers
- Density plots to understand distribution shapes

### Correlation Analysis

```python
# Examine relationships between variables
cleanser.plot_correlations()
```

This creates:
- Correlation matrix heatmap
- Pairwise scatter plots for numeric features

## Data Preprocessing

After exploring your data, use these preprocessing techniques:

### Missing Value Handling

```python
# Handle missing values
cleanser.handle_missing_values(strategy='auto')
```

Available strategies:
- `'mean'`, `'median'`, `'mode'`: Impute with respective statistic
- `'constant'`: Replace with a specified value
- `'auto'`: Automatically select appropriate strategy based on data type
- `'drop'`: Remove rows with missing values

### Outlier Treatment

```python
# Handle outliers
cleanser.handle_outliers(method='iqr', threshold=1.5)
```

Available methods:
- `'iqr'`: Interquartile Range method (default threshold=1.5)
- `'zscore'`: Z-score method (default threshold=3)
- `'percentile'`: Remove values outside specified percentile range
- `'cap'`: Cap values at specified percentiles

### Date Feature Engineering

```python
# Extract useful features from date columns
cleanser.create_date_features('date_column')
```

This extracts:
- Year, month, day
- Day of week, day of year
- Quarter, week of year
- Is weekend/weekday

### Categorical Encoding

```python
# Encode categorical variables
cleanser.encode_categorical(method='onehot')
```

Available encoding methods:
- `'onehot'`: One-hot encoding (creates binary columns)
- `'label'`: Label encoding (converts to numeric values)
- `'ordinal'`: Ordinal encoding (for categories with inherent order)
- `'target'`: Target encoding (for supervised learning)

### Feature Scaling

```python
# Scale numerical features
cleanser.scale_features(method='standard')
```

Available scaling methods:
- `'standard'`: Standardization (zero mean, unit variance)
- `'minmax'`: Min-Max normalization (scale to 0-1 range)
- `'robust'`: Robust scaling using median and IQR
- `'log'`: Log transformation (for skewed data)

## Complete Pipeline Example

The DataCleanser allows method chaining for a cleaner workflow:

```python
processed_df = cleanser.handle_missing_values(strategy='median') \
                      .handle_outliers(method='iqr', threshold=1.5) \
                      .create_date_features('signup_date') \
                      .encode_categorical(method='onehot') \
                      .scale_features(method='standard') \
                      .get_data()
```

## Best Practices

1. **Always explore before preprocessing**: Use the built-in visualization methods to understand your data first.
2. **Consider domain knowledge**: Use domain expertise to guide preprocessing decisions.
3. **Try different approaches**: Use the `reset_to_original()` method to try different preprocessing pipelines.
4. **Document your steps**: Record which preprocessing steps you applied for reproducibility.
5. **Check after preprocessing**: Verify that your preprocessing worked as expected.

For more detailed examples, see the Jupyter notebook: `example_usage.ipynb`