import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import pickle

# Load your dataset
data = pd.read_csv( r'C:\Users\HP\OneDrive\Desktop\Sandip_IITI\Programming Lab\weather\Weather_Prediction\test.csv')
data.ffill()
data=data.dropna()
data['Date'] = pd.to_datetime(data['Date'],format='mixed')
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['day_of_week'] = data['Date'].dt.dayofweek
X = data[['City', 'year', 'month', 'day', 'day_of_week']]
y = data[['tavg', 'tmin', 'tmax', 'prcp']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['City']),
            ('num', 'passthrough', ['year', 'month', 'day', 'day_of_week'])
        ])),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)



# Save the model
pickle.dump(pipeline, open('weather_model.pkl', 'wb'))
# Load the model
model = pickle.load(open('weather_model.pkl', 'rb'))