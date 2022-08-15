import json
import os
import numpy as np
import pandas as pd
import boto3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

s3 = boto3.client('s3')
s3.download_file("model-storage-lambda", "dataset.csv", '/tmp/dataset.csv')

data = pd.read_csv('/tmp/dataset.csv', index_col = False)

train_dataset = data.sample(frac=0.9, random_state=15)
test_dataset = data.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('price')
test_labels = test_features.pop('price')

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.1,
    verbose=0, epochs=20)

data = pd.read_csv('coded_cars.csv')
manufacturerDict = {} #direct manufacturer to model dictionary
modelCodeDict = {} #model to model code dictionary
manufacturerCodeDict = {} #make to make code dictionary
transmissionCodeDict = {
    'Automatic': 0,
    'Manual': 1   
}
conditionCodeDict = {
    'Excellent': 0,
    'Fair': 1,
    'Good': 2,
    'Like New': 3,
    'New': 4
}
driveCodeDict = {
    'AWD': 0,
    'FWD': 1,
    'RWD': 2
}
titleStatusCodeDict = {
    'Clean': 0,
    'Lien': 1,
    'Missing': 2,
    'Rebuilt': 4,
    'Salvage': 5
}
stateCodeDict = {}
paintCodeDict = {}

i = 0
for row in data.itertuples():
    if row.paint_color not in paintCodeDict:
        paintCodeDict[row.paint_color] = row.paint_color_Codes
        
    if row.manufacturer not in manufacturerDict:
       manufacturerDict[row.manufacturer] = [row.model]
       modelCodeDict[row.model] = row.model_Codes
       manufacturerCodeDict[row.manufacturer] = row.manufacturer_Codes
    else:
        if row.model not in manufacturerDict[row.manufacturer]:
            manufacturerDict[row.manufacturer].append(row.model) 
            modelCodeDict[row.model] = row.model_Codes

def lambda_handler(event, context):
    
    year = int(event['queryStringParameters']['year'])
    manufac = str(event['queryStringParameters']['manufacturer'])
    model = str(event['queryStringParameters']['model'])
    transmission = str(event['queryStringParameters']['transmission'])
    condition = str(event['queryStringParameters']['condition'])
    odometer = int(event['queryStringParameters']['odometer'])
    paint = str(event['queryStringParameters']['color'])
    title = str(event['queryStringParameters']['title'])
    state = 1

    d = np.array([[year, manufac, model, condition, odometer, title, transmission, paint, state]])

    test_predictions = dnn_model.predict(d)
    result = {}
    result['predicted_price'] = test_predictions[0][0]
    
    response_obj = {}
    response_obj['statusCode'] = 200
    response_obj['headers'] = {}
    response_obj['headers']['Content-Type'] = 'application/json'
    response_obj['body'] = json.dumps(result)
    
    # TODO implement
    return response_obj
