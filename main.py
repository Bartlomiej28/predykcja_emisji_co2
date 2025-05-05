import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/CO2 Emissions_Canada.csv', sep=',')

columns_to_encode = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

encoded_data = data.copy()

for col in columns_to_encode:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col])

X = encoded_data.drop(['CO2 Emissions(g/km)'], axis=1)
y = encoded_data['CO2 Emissions(g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(125, activation='swish', input_shape=[11]),
    keras.layers.Dense(128, activation='swish'),
    keras.layers.Dense(64, activation='swish'),
    keras.layers.Dense(32, activation='swish'),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=[keras.metrics.RootMeanSquaredError()])

model.fit(X_train_scaled, y_train, batch_size=32, epochs=30)

train_error = model.evaluate(X_train_scaled, y_train)
test_error = model.evaluate(X_test_scaled, y_test)

print(train_error, test_error)

prediction = model.predict(X_test_scaled[1].reshape(1, -1))
print('Przewidziane: ', prediction, ' Prawdziwe: ', y_test.iloc[1])