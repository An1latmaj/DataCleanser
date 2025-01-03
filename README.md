# DataCleanser

A comprehensive Python utility for data exploration, cleaning, and preprocessing, designed to streamline EDA and prepare data for machine learning pipelines.

## Overview

DataCleanser simplifies the data preparation workflow by providing a unified interface for common data cleaning and preprocessing tasks. Based on best practices from Kaggle competitions and data science literature, this utility helps data scientists and analysts spend less time on boilerplate code and more time on analysis and modeling.

## Features

- **Exploratory Data Analysis**
  - Automated data profiling and visualization
  - Distribution analysis
  - Correlation exploration
  - Missing value assessment

- **Data Cleaning**
  - Multiple strategies for handling missing values
  - Outlier detection and treatment
  - Data type conversion and validation

- **Feature Engineering**
  - Date feature extraction
  - Categorical encoding
  - Feature scaling
  - Feature transformation

- **Usability**
  - Method chaining for clean pipelines
  - Automatic detection of appropriate strategies
  - Preservation of original data
  - Detailed logging of transformations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DataCleanser.git
cd DataCleanser

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from data_cleanser import DataCleanser
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create DataCleanser instance
cleanser = DataCleanser(df)

# Explore data
cleanser.get_basic_info()
cleanser.visualize_distributions()
cleanser.plot_correlations()

# Process data with method chaining
processed_df = cleanser.handle_missing_values() \
                     .handle_outliers() \
                     .encode_categorical() \
                     .scale_features() \
                     .get_data()
```

## Documentation

For detailed usage instructions, see:

- [EDA and Preprocessing Guide](eda_preprocessing_guide.md): Complete guide to data exploration and preprocessing techniques
- [Example Notebook](example_usage.ipynb): Jupyter notebook with practical examples

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Use Cases

- **Machine Learning Preparation**: Clean and preprocess data for model training
- **Data Quality Assessment**: Quickly evaluate dataset quality and identify issues
- **Feature Engineering**: Generate new features and transform existing ones
- **Data Exploration**: Rapidly visualize and understand dataset characteristics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by best practices from Kaggle courses and competitions
- Thanks to the pandas, numpy, matplotlib, and scikit-learn communities for their excellent tools