import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

warnings.filterwarnings('ignore')
global ship1
global ship2


# # read csv files
# result=pd.read_csv('ship1_total.csv')
# result=result.set_index('timestamp')
#
# #query condition data
# full_data=result.query("condition==1")
# empty_data=result.query("condition==0")

# fuction 1 clean dataset
def LoadData(path, kind):
    global ship1
    global ship2

    if kind == 1:
        ship1 = pd.read_csv(path)
        ship1 = ship1.set_index('timestamp')
    else:
        ship2 = pd.read_csv(path)
        ship2 = ship2.set_index('timestamp')


def clean(data):
    names = data.columns
    IOR = []
    UpLimit1 = []
    DownLimit1 = []
    up = []
    for i, col in enumerate(names):
        s = data[col]
        IQR = s.quantile(0.75) - s.quantile(0.25)
        IOR.append(IQR)
        UpLimit = s.quantile(0.75) + IQR * 1.5
        DownLimit = s.quantile(0.75) - IQR * 1.5
        UpLimit1.append(UpLimit)
        DownLimit1.append(DownLimit)
    df_des = data.describe()
    df_des.loc['UpLimit'] = UpLimit1
    df_des.loc['DownLimit'] = DownLimit1
    new_data = data[(data['ship_FuelEfficiency'] < df_des['ship_FuelEfficiency']['UpLimit']) & (
            data['ship_FuelEfficiency'] > df_des['ship_FuelEfficiency']['DownLimit'])]
    return new_data


def new_data(data1, data2):
    frame = [data1, data2]
    new_data = pd.concat(frame, sort=False, ignore_index=True)
    from sklearn.utils import shuffle
    new_data = shuffle(new_data)
    return new_data


# using random forest algorithm to selected five important features
def column_selection(data):
    # X = data[['engine_power', 'Rot', 'Rotation', 'RudderAngle', 'SPEED_KNOTS', 'WindDiration', 'WindSpeed', 'draft',
    #           'reWind', 'trim']]
    # Y = data['ship_FuelEfficiency']
    # from sklearn.ensemble import RandomForestRegressor
    # model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=50,
    #                               max_features="auto", min_samples_leaf=100)
    # model.fit(X, Y)
    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # feat_labels = X.columns
    # a = []
    # for f in range(X.shape[1]):
    #     temp = feat_labels[indices[f]]
    #     a.append(temp)
    select_columns = ['draft', 'engine_power', 'SPEED_KNOTS', 'Rotation', 'WindSpeed']
    return select_columns


def data_transform_X(data, select_columns):
    model_data = data[select_columns]
    from sklearn.preprocessing import StandardScaler  # 实现数据标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(model_data)
    model_data = pd.DataFrame(data, columns=model_data.columns)
    from sklearn.preprocessing import PolynomialFeatures  # 将标准化数据进行多项式转化，最高次项为2次。
    poly = PolynomialFeatures(2)
    X_poly = poly.fit(model_data)
    features = pd.DataFrame(poly.transform(model_data), columns=poly.get_feature_names(model_data.columns))
    X_final_data = features
    return X_final_data


def Y_value(data):
    # 将模块目标参数：吨海里油耗量进行中心化处理
    a = (data['ship_FuelEfficiency'] - data['ship_FuelEfficiency'].min()) / (
            data['ship_FuelEfficiency'].max() - data['ship_FuelEfficiency'].min())
    return a


def ship1_lasso_colums(data, Y):
    names = data.columns
    from sklearn.linear_model import LassoCV
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(data, Y)
    from sklearn.feature_selection import RFE
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(model_lasso, n_features_to_select=1)
    rfe.fit(data, Y)
    rank = rfe.ranking_
    indices = np.argsort(rank)[::1]
    feat_labels = data.columns
    a = []
    for f in range(data.shape[1]):
        temp = feat_labels[indices[f]]
        a.append(temp)
    lasso_select_columns = a[0:8]
    return lasso_select_columns


def lasso_colums_ship2(data, Y):
    names = data.columns
    from sklearn.linear_model import LassoCV
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(data, Y)
    from sklearn.feature_selection import RFE
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(model_lasso, n_features_to_select=1)
    rfe.fit(data, Y)
    rank = rfe.ranking_
    indices = np.argsort(rank)[::1]
    feat_labels = data.columns
    a = []
    for f in range(data.shape[1]):
        temp = feat_labels[indices[f]]
        a.append(temp)
    lasso_select_columns = a[0:12]
    return lasso_select_columns


def lasso_coef_full(data, Y, lasso_columns):
    train_full_1 = data[lasso_columns]
    names = train_full_1.columns
    from sklearn.linear_model import LassoCV
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(train_full_1, Y)
    coef = pd.Series(model_lasso.coef_, index=names)
    coef_full = coef.round(3)
    return coef_full


def his_engine_data(data):
    raw_data = data.describe()
    em_power = raw_data['engine_power']['50%'].round(2)
    em_rotation = raw_data['Rotation']['50%'].round(2)
    return em_power, em_rotation


def ship1_opt_speed_full(data, coef_full, emp_full, emr_full):
    raw_full = clean(data).describe()
    from scipy.optimize import minimize_scalar
    def f(x):
        return coef_full['SPEED_KNOTS^2'] * x ** 2 + coef_full['SPEED_KNOTS'] * x + coef_full[
            'engine_power SPEED_KNOTS'] * (emp_full) * x + coef_full['engine_power'] * (emp_full) + coef_full[
                   'engine_power^2'] * (emp_full ** 2) + coef_full['Rotation'] * (emr_full) + coef_full[
                   'SPEED_KNOTS Rotation'] * (emr_full) * x

    res = minimize_scalar(f, method='brent')
    opt_speed = res.x * raw_full['SPEED_KNOTS']['std'] + raw_full['SPEED_KNOTS']['mean']
    return opt_speed


def lasso_coef_empty(data, Y, lasso_columns):
    train_full_1 = data[lasso_columns]
    names = train_full_1.columns
    from sklearn.linear_model import LassoCV
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(train_full_1, Y)
    coef = pd.Series(model_lasso.coef_, index=names)
    coef_empty = coef.round(3)
    return coef_empty


def shop1_opt_speed_empty(data, coef_empty, emp_empty, emr_empty):
    raw_empty = clean(data).describe()
    from scipy.optimize import minimize_scalar
    def f(x):
        return coef_empty['SPEED_KNOTS^2'] * x ** 2 + coef_empty['SPEED_KNOTS'] * x + coef_empty[
            'engine_power SPEED_KNOTS'] * (emp_empty) * x + coef_empty['engine_power'] * (emp_empty) + coef_empty[
                   'engine_power^2'] * (emp_empty ** 2) + coef_empty['Rotation'] * (emr_empty)

    res = minimize_scalar(f, method='brent')
    opt_speed = res.x * raw_empty['SPEED_KNOTS']['std'] + raw_empty['SPEED_KNOTS']['mean']
    return opt_speed


def ship2_opt_speed(data, coef_, emp_full, emr_full):
    raw_full = clean(data).describe()
    from scipy.optimize import minimize_scalar
    def f(x):
        return coef_['SPEED_KNOTS^2'] * x ** 2 + coef_['SPEED_KNOTS'] * x + coef_['engine_power SPEED_KNOTS'] * (
            emp_full) * x + coef_['engine_power'] * (emp_full) + coef_['engine_power^2'] * (emp_full ** 2) + coef_[
                   'Rotation'] * (emr_full) + coef_['SPEED_KNOTS Rotation'] * (emr_full) * x + coef_['Rotation^2'] * (
                       emr_full ** 2)

    res = minimize_scalar(f, method='brent')
    opt_speed = res.x * raw_full['SPEED_KNOTS']['std'] + raw_full['SPEED_KNOTS']['mean']
    return opt_speed


def get_ship1_speed(data, LoadCondition):
    full_data = data[data['condition'] == 1]
    clean_full = clean(full_data)
    empty_data = data[data['condition'] == 0]
    clean_empty = clean(empty_data)
    whole_data = new_data(clean_full, clean_empty)
    select_columns = column_selection(whole_data)
    if LoadCondition == 1:
        X_full = data_transform_X(clean_full, select_columns)
        Y_full = Y_value(clean_full)
        full_lasso_columns = ship1_lasso_colums(X_full, Y_full)
        coef_full = lasso_coef_full(X_full, Y_full, full_lasso_columns)
        emp_full, emr_full = his_engine_data(X_full)
        full_opt_speed = ship1_opt_speed_full(full_data, coef_full, emp_full, emr_full)
        return full_opt_speed
    if LoadCondition == 0:
        X_empty = data_transform_X(clean_empty, select_columns)
        Y_empty = Y_value(clean_empty)
        lasso_columns_empty = ship1_lasso_colums(X_empty, Y_empty)
        coef_empty = lasso_coef_empty(X_empty, Y_empty, lasso_columns_empty)
        emp_empty, emr_empty = his_engine_data(X_empty)
        empty_opt_speed = shop1_opt_speed_empty(empty_data, coef_empty, emp_empty, emr_empty)
        return empty_opt_speed


def get_ship2_speed(data, LoadCondition):
    global ship2
    global ship2optspeed
    full_data = data[data['condition'] == 1]
    clean_full = clean(full_data)
    empty_data = data[data['condition'] == 0]
    clean_empty = clean(empty_data)
    whole_data = new_data(clean_full, clean_empty)
    select_columns = column_selection(whole_data)
    if LoadCondition == 1:
        X_full = data_transform_X(clean_full, select_columns)
        Y_full = Y_value(clean_full)
        lasso_columns_full = lasso_colums_ship2(X_full, Y_full)
        coef_full = lasso_coef_full(X_full, Y_full, lasso_columns_full)
        emp_full_2, emr_full_2 = his_engine_data(X_full)
        ship2_full_opt = ship2_opt_speed(full_data, coef_full, emp_full_2, emr_full_2)
        return ship2_full_opt
    if LoadCondition == 0:
        X_empty = data_transform_X(clean_empty, select_columns)
        Y_empty = Y_value(clean_empty)
        lasso_columns_empty = lasso_colums_ship2(X_empty, Y_empty)
        coef_empty = lasso_coef_empty(X_empty, Y_empty, lasso_columns_empty)
        emp_empty_2, emr_empty_2 = his_engine_data(X_empty)
        ship2_empty_opt = ship2_opt_speed(empty_data, coef_empty, emp_empty_2, emr_empty_2)
        return ship2_empty_opt


def get_speed(x1, x2, x3, x4):
    if x3 == 1 and x4 == 1:
        x1 = (x1 - 9573.89) / 1363.21
        x2 = (x2 - 44.52) / 1.95
        from scipy.optimize import minimize_scalar
        def f(x):
            return 0.022 * x ** 2 - 0.191 * x - 0.017 * (x1) * x + 0.181 * (x1) + 0.006 * (x1 ** 2) + 0.028 * (
                x2) - 0.005 * (x2) * x

        res = minimize_scalar(f, method='brent')
        opt_speed = res.x * 1.12 + 10.32
        return opt_speed
    if x3 == 1 and x4 == 0:
        x1 = (x1 - 7993) / 2579.95
        x2 = (x2 - 43.88) / 4.92
        from scipy.optimize import minimize_scalar
        def f(x):
            return 0.027 * x ** 2 - 0.167 * x - 0.042 * (x1) * x + 0.008 * (x1 ** 2) + 0.025 * (x2) + 0.087 * (x2)

        res = minimize_scalar(f, method='brent')
        opt_speed = res.x * 1.66 + 11.67
        return opt_speed
    if x3 == 2 and x4 == 1:
        x1 = (x1 - 9798.23) / 2344.71
        x2 = (x2 - 44.63) / 3.51
        from scipy.optimize import minimize_scalar
        def f(x):
            return 0.024 * x ** 2 - 0.226 * x + 0.012 * (x1) * x + 0.145 * (x1) + 0.007 * (x1 ** 2) + 0.028 * (
                x2) + 0.065 * (x2) - 0.015 * ((x2) ** 2)

        res = minimize_scalar(f, method='brent')
        opt_speed = res.x * 1.38 + 10.30
        return opt_speed
    if x3 == 2 and x4 == 0:
        x1 = (x1 - 8638.723) / 2035.38
        x2 = (x2 - 43.44) / 3.39
        from scipy.optimize import minimize_scalar
        def f(x):
            return 0.007 * x ** 2 - 0.122 * x - 0.026 * (x1) * x + 0.166 * (x1) + 0.015 * (x1 ** 2) + 0.028 * (
                x2) * x + 0.065 * (x2) - 0.015 * ((x2) ** 2)

        res = minimize_scalar(f, method='brent')
        opt_speed = res.x * 1.40 + 11.29
        return opt_speed
