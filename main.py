from pmdarima.arima import auto_arima
import os
import pandas as pd
from data_process import get_mape, get_rmse
import csv

ratio = 0.8
output = []
steps = 4

# read data
path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
df = pd.read_csv(path, encoding='gbk')
columns = df.columns

for j in range(1, len(columns)):
    # data preprocess
    # j = 1
    print(columns[j])
    raw_series = df[columns[j]]
    raw_series = raw_series.tolist()

    train_len = int(len(raw_series) * ratio)
    test_len = len(raw_series) - train_len

    train_set = raw_series[:train_len]
    test_set = raw_series[train_len:]

    # train model
    model = auto_arima(train_set, start_p=0, start_q=0)
    train_fitted = model.arima_res_.fittedvalues
    # print(model.summary())
    # model.plot_diagnostics()
    train_MAPE = get_mape(train_set, train_fitted)
    train_RMSE = get_rmse(train_set, train_fitted)
    # print(train_set, train_fitted, train_MAPE, train_RMSE)

    forecast, conf_int = model.predict(n_periods=4, return_conf_int=True)
    test_MAPE = get_mape(test_set, forecast)
    test_RMSE = get_rmse(test_set, forecast)
    # print(test_set, forecast, test_MAPE, test_RMSE)

    _output = [steps, columns[j], train_fitted, train_set, train_MAPE, train_RMSE, forecast, test_set, test_MAPE, test_RMSE]
    output.append(_output)


# save result
f = open('output.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
header = ('steps', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
csv_writer.writerow(header)
for data in output:
    csv_writer.writerow(data)
f.close()
