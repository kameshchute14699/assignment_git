import pandas as pd
import numpy as np


file_path = "E:\Users\hp\Downloads\products (1).csv"
data = pd.read_csv(file_path)



missing_summary = data.isnull().sum()  


data = data.drop_duplicates()  

numerical_cols = ['Price', 'StockQuantity']
z_scores = np.abs((data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std())
data_no_outliers_z = data[(z_scores < 3).all(axis=1)]


Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
data_no_outliers_iqr = data[~((data[numerical_cols] < (Q1 - 1.5 * IQR)) | 
                              (data[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]



profit_margins = {'Electronics': 0.40, 'Clothing': 0.50, 'Furniture': 0.30, 'Books': 0.20}
data['ProfitMargin'] = data['Category'].map(profit_margins)
data['Profit'] = data['Price'] * data['ProfitMargin']


if 'OrderDate' not in data.columns:
    data['OrderDate'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D') 

data['Month'] = data['OrderDate'].dt.month
data['Weekday'] = data['OrderDate'].dt.day_name()

data['Sales'] = data['Price'] * data['StockQuantity']  
data['CumulativeSales'] = data.groupby('Category')['Sales'].cumsum()


data['SalesGrowth'] = data.groupby('ProductID')['Sales'].pct_change().fillna(0) * 100


conditions = [
    (data['SalesGrowth'] > 0),
    (data['SalesGrowth'] == 0),
    (data['SalesGrowth'] < 0)
]
choices = ['Growing', 'Stable', 'Declining']
data['Trend'] = np.select(conditions, choices, default='Unknown')


processed_file_path = '/mnt/data/processed_products.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to: {processed_file_path}")
