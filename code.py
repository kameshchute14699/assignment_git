import pandas as pd
import numpy as np


file_path = "E:\Users\hp\Downloads\sales.csv"
data = pd.read_csv(file_path)


missing_summary = data.isnull().sum()  

data = data.drop_duplicates() 


numerical_cols = ['Price', 'StockQuantity']
z_scores = np.abs((data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std())
data_no_outliers_z = data[(z_scores < 3).all(axis=1)]


Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
data_no_outliers_iqr = data[~((data[numerical_cols] < (Q1 - 1.5 * IQR))  
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




mean_sales = sales_data['TotalAmount'].mean()
median_sales = sales_data['TotalAmount'].median()
variance_sales = sales_data['TotalAmount'].var()
std_dev_sales = sales_data['TotalAmount'].std()


correlation_quantity_sales = sales_data[['Quantity', 'TotalAmount']].corr()


print("Descriptive Statistics for TotalAmount:")
print(f"Mean: {mean_sales}")
print(f"Median: {median_sales}")
print(f"Variance: {variance_sales}")
print(f"Standard Deviation: {std_dev_sales}")

print("\nCorrelation Coefficients between Quantity and TotalAmount:")
print(correlation_quantity_sales)


processed_file_path = '/mnt/data/processed_products.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to: {processed_file_path}")



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


file_path = '/mnt/data/products (1).csv'
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


sales_file_path = '/mnt/data/sales.csv'
sales_data = pd.read_csv(sales_file_path)


sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data = sales_data.sort_values(by='Date')


X = sales_data[['Quantity']]
y = sales_data['TotalAmount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_pred = lr_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)


monthly_sales = sales_data.resample('M', on='Date').sum()['TotalAmount']


arima_model = ARIMA(monthly_sales, order=(1, 1, 1))
arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(steps=6)

print("Linear Regression Metrics:")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

print("\nARIMA Forecast (Next 6 Months):")
print(arima_forecast)


processed_file_path = '/mnt/data/processed_products.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to: {processed_file_path}")
