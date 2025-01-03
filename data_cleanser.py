"""
DataCleanser: A utility for performing EDA and data preprocessing 
based on techniques from kaggle-courses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

class DataCleanser:
    """
    A class to handle common data exploration and preprocessing tasks
    based on best practices from kaggle-courses.
    """
    
    def __init__(self, df=None, file_path=None):
        """
        Initialize DataCleanser with either a pandas DataFrame or a file path.
        
        Args:
            df (pandas.DataFrame, optional): Input DataFrame to process
            file_path (str, optional): Path to a CSV file to load
        """
        if df is not None:
            self.df = df.copy()
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            self.df = None
            
        self.original_df = self.df.copy() if self.df is not None else None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
    def load_data(self, file_path, **kwargs):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv()
        """
        self.df = pd.read_csv(file_path, **kwargs)
        self.original_df = self.df.copy()
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        return self
        
    def get_basic_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            print("No data loaded yet.")
            return
            
        print(f"Dataset Shape: {self.df.shape}")
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nSummary Statistics:")
        print(self.df.describe())
        
        # Missing values summary
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Values:")
            missing_percent = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Count': missing,
                'Percentage': missing_percent
            }).sort_values('Count', ascending=False)
            print(missing_df[missing_df['Count'] > 0])
        else:
            print("\nNo missing values found!")
            
        # Identify column types
        self._identify_column_types()
        
        return self
        
    def _identify_column_types(self):
        """
        Identify and store types of columns (numeric, categorical, datetime).
        """
        if self.df is None:
            return
            
        # Reset lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in self.df.columns:
            # Try to convert to datetime
            try:
                pd.to_datetime(self.df[col])
                self.datetime_columns.append(col)
                continue
            except:
                pass
                
            # Check data type
            if self.df[col].dtype in ['int64', 'float64']:
                if self.df[col].nunique() < 10 and self.df[col].nunique() / len(self.df) < 0.05:
                    self.categorical_columns.append(col)
                else:
                    self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)
                
        print(f"\nColumn types identified:")
        print(f"- Numeric columns: {len(self.numeric_columns)}")
        print(f"- Categorical columns: {len(self.categorical_columns)}")
        print(f"- DateTime columns: {len(self.datetime_columns)}")
        
    def visualize_distributions(self, columns=None, figsize=(15, 12)):
        """
        Visualize distributions of numeric features.
        
        Args:
            columns (list, optional): List of columns to visualize. If None, uses all numeric columns.
            figsize (tuple, optional): Figure size
        """
        if self.df is None:
            print("No data loaded yet.")
            return
            
        if columns is None:
            if not self.numeric_columns:
                self._identify_column_types()
            columns = self.numeric_columns[:10]  # Limit to 10 to avoid cluttered plots
            
        cols = min(3, len(columns))
        rows = (len(columns) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            if i < len(axes):
                try:
                    sns.histplot(self.df[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                except:
                    axes[i].text(0.5, 0.5, f"Could not plot {col}", ha='center')
                    
        plt.tight_layout()
        plt.show()
        return self
        
    def plot_correlations(self, columns=None, figsize=(12, 10)):
        """
        Plot correlation matrix for numeric features.
        
        Args:
            columns (list, optional): List of columns to include. If None, uses all numeric columns.
            figsize (tuple, optional): Figure size
        """
        if self.df is None:
            print("No data loaded yet.")
            return
            
        if columns is None:
            if not self.numeric_columns:
                self._identify_column_types()
            columns = self.numeric_columns
            
        if len(columns) > 15:
            print(f"Warning: Large correlation matrix with {len(columns)} columns may be hard to read.")
            print("Limiting to top 15 columns.")
            columns = columns[:15]
            
        corr = self.df[columns].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return self
        
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str or dict): 
                - 'auto': Automatically choose strategy based on column type
                - 'drop': Drop rows with missing values
                - 'mean', 'median', 'mode': Fill with mean/median/mode
                - dict: Map column names to specific strategies
                
        Returns:
            self: The DataCleanser object
        """
        if self.df is None:
            print("No data loaded yet.")
            return self
            
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values to handle.")
            return self
            
        print(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            original_shape = self.df.shape
            self.df = self.df.dropna()
            print(f"Dropped {original_shape[0] - self.df.shape[0]} rows with missing values.")
            
        elif strategy == 'auto':
            # Automatically handle based on column type
            if not self.numeric_columns or not self.categorical_columns:
                self._identify_column_types()
                
            # For numeric columns: Use median
            for col in self.numeric_columns:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"Column '{col}': Filled {missing[col]} missing values with median ({median_val:.2f})")
                    
            # For categorical columns: Use mode (most frequent)
            for col in self.categorical_columns:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"Column '{col}': Filled {missing[col]} missing values with mode ({mode_val})")
                    
        elif strategy in ['mean', 'median', 'mode']:
            # Apply the same strategy to all columns
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        # If mean/median requested for non-numeric, fallback to mode
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                        
        elif isinstance(strategy, dict):
            # Apply specific strategy to each column as specified in the dict
            for col, method in strategy.items():
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    if method == 'drop':
                        self.df = self.df.dropna(subset=[col])
                    elif method == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif method == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif method == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    elif isinstance(method, (int, float, str)):
                        self.df[col].fillna(method, inplace=True)
                        
        print("Missing values handled successfully!")
        return self
        
    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Detect and handle outliers in numeric columns.
        
        Args:
            columns (list, optional): List of columns to process. If None, uses all numeric columns.
            method (str): Method to use ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection (1.5 for IQR, 3 for z-score)
            
        Returns:
            self: The DataCleanser object
        """
        if self.df is None:
            print("No data loaded yet.")
            return self
            
        if columns is None:
            if not self.numeric_columns:
                self._identify_column_types()
            columns = self.numeric_columns
            
        print(f"Handling outliers with method: {method}")
        
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in ['int64', 'float64']:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Count outliers
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    print(f"Column '{col}': {outliers} outliers detected ({outliers/len(self.df)*100:.1f}%)")
                    # Cap the outliers
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"  - Capped values to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
            elif method == 'zscore':
                from scipy import stats
                z_scores = stats.zscore(self.df[col], nan_policy='omit')
                outliers = (abs(z_scores) > threshold).sum()
                
                if outliers > 0:
                    print(f"Column '{col}': {outliers} outliers detected ({outliers/len(self.df)*100:.1f}%)")
                    # Cap the outliers
                    self.df.loc[abs(z_scores) > threshold, col] = np.sign(
                        self.df.loc[abs(z_scores) > threshold, col]) * threshold * self.df[col].std() + self.df[col].mean()
                    print(f"  - Capped outliers using z-score threshold of {threshold}")
                    
        return self
        
    def encode_categorical(self, columns=None, method='onehot'):
        """
        Encode categorical variables.
        
        Args:
            columns (list, optional): List of columns to encode. If None, uses all categorical columns.
            method (str): Encoding method ('onehot', 'label', 'ordinal')
            
        Returns:
            self: The DataCleanser object
        """
        if self.df is None:
            print("No data loaded yet.")
            return self
            
        if columns is None:
            if not self.categorical_columns:
                self._identify_column_types()
            columns = self.categorical_columns
            
        print(f"Encoding categorical variables with method: {method}")
        
        if method == 'onehot':
            # One-hot encode categorical variables
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
            print(f"One-hot encoded {len(columns)} columns. New shape: {self.df.shape}")
            
        elif method == 'label':
            # Label encode categorical variables
            le = LabelEncoder()
            for col in columns:
                if col in self.df.columns:
                    self.df[f"{col}_encoded"] = le.fit_transform(self.df[col].astype(str))
            print(f"Label encoded {len(columns)} columns.")
            
        return self
        
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numeric features.
        
        Args:
            columns (list, optional): List of columns to scale. If None, uses all numeric columns.
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            self: The DataCleanser object
        """
        if self.df is None:
            print("No data loaded yet.")
            return self
            
        if columns is None:
            if not self.numeric_columns:
                self._identify_column_types()
            columns = self.numeric_columns
            
        print(f"Scaling features with method: {method}")
        
        if method == 'standard':
            # Standardization (z-score normalization)
            scaler = StandardScaler()
            self.df[columns] = scaler.fit_transform(self.df[columns])
            
        elif method == 'minmax':
            # Min-Max scaling
            scaler = MinMaxScaler()
            self.df[columns] = scaler.fit_transform(self.df[columns])
            
        elif method == 'robust':
            # Robust scaling
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            self.df[columns] = scaler.fit_transform(self.df[columns])
            
        print(f"Scaled {len(columns)} features.")
        return self
        
    def create_date_features(self, column, drop_original=False):
        """
        Extract features from a date column.
        
        Args:
            column (str): Name of the date column
            drop_original (bool): Whether to drop the original column
            
        Returns:
            self: The DataCleanser object
        """
        if self.df is None or column not in self.df.columns:
            print(f"Column {column} not found.")
            return self
            
        try:
            # Convert to datetime if not already
            self.df[column] = pd.to_datetime(self.df[column])
            
            # Extract date components
            self.df[f'{column}_year'] = self.df[column].dt.year
            self.df[f'{column}_month'] = self.df[column].dt.month
            self.df[f'{column}_day'] = self.df[column].dt.day
            self.df[f'{column}_dayofweek'] = self.df[column].dt.dayofweek
            self.df[f'{column}_quarter'] = self.df[column].dt.quarter
            
            # Drop original if requested
            if drop_original:
                self.df.drop(column, axis=1, inplace=True)
                
            print(f"Created date features from column '{column}'.")
            
        except Exception as e:
            print(f"Error creating date features: {e}")
            
        return self
    
    def reset_to_original(self):
        """Reset the dataframe to its original state."""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            print("Reset to original data.")
        return self
    
    def get_data(self):
        """Return the processed dataframe."""
        return self.df
