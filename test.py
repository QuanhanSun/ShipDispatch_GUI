import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

global ship1
global ship2


def LoadData(path, kind):
    global ship1
    global ship2

    if kind == 1:
        ship1 = pd.read_csv(path)
        ship1 = ship1.set_index('timestamp')
    else:
        ship2 = pd.read_csv(path)
        ship2 = ship2.set_index('timestamp')


def new_data(data1, data2):
    frame = [data1, data2]
    new_data = pd.concat(frame, sort=False, ignore_index=True)
    from sklearn.utils import shuffle
    new_data = shuffle(new_data)
    return new_data


# def column_selection(data):
#     X = data[['engine_power', 'Rot', 'Rotation', 'RudderAngle', 'SPEED_KNOTS', 'WindDiration', 'WindSpeed', 'draft',
#               'reWind', 'trim']]
#     Y = data['ship_FuelEfficiency']
#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=50,
#                                   max_features="auto", min_samples_leaf=100)
#     model.fit(X, Y)
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     feat_labels = X.columns
#     a = []
#     for f in range(X.shape[1]):
#         temp = feat_labels[indices[f]]
#         a.append(temp)
#     select_columns = ['draft', 'engine_power', 'SPEED_KNOTS', 'Rotation', 'WindSpeed']
#     return select_columns
path = '/Users/sunquanhan/Desktop/GUI_Demo/ship1_total.csv'
LoadData(path, 1)
LoadData(path, 2)

data = new_data(ship1, ship2)

X = data[['engine_power', 'Rot', 'Rotation', 'RudderAngle', 'SPEED_KNOTS', 'WindDiration', 'WindSpeed', 'draft',
          'reWind', 'trim']]
Y = data['ship_FuelEfficiency']

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=50,
                              max_features="auto", min_samples_leaf=100)
model.fit(X,Y)