import numpy as np
import matplotlib.pyplot as plt
import robin_stocks.robinhood as r
import quandl
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Days forecasted currently hard coded, should be refactored as an
# argument for the prepare data function.
days_to_forecast = int(30)


# Initialize Quandl, listed api key is currently active as of 6 May 21
def initialize_quandl():
    quandl.ApiConfig.api_key = "io8gV4t4aqiCZnX4ybMR"
    quandl.ApiConfig.use_retries = False


# Returns a data frame consisting of the features listed in "qopts"
# Can be modified to return other data points
def quandl_get_table(ticker):
    entry = str(ticker)
    my_table = quandl.get_table('WIKI/PRICES', qopts={'columns': ['date', 'close']},
                                ticker=[entry],
                                date={'gte': '2016-01-01', 'lte': '2018-04-28'})
    x = my_table['date']
    y = my_table['close']
    return x, y


# Returns the full data frame for the selected ticker.
# Dataframe can be slimmed down to only the wanted info
# df = df[['DesiredColumn']]
def quandl_get_full_table(ticker):
    entry = str(ticker)
    df = quandl.get('WIKI/' + entry)
    return df


# Ticker is the string representing the stocks you wish to gather data for.
# Grabs the data frame for the requested ticker by calling the get full table method.
# Sets up frame to predict the number of days in the future per the hard-coded
# "days_to_forecast" variable. This can be modified for a longer or shorter
# timeframe.
# Returns X, y, and X_forecast for use in the perform_linear_regression method
# forecast_frame provides a frame for prediction based on the days_to_forecast variable
# Do not hard code an int in the function for the days to forecast, change the global variable.
def prepare_data(ticker):
    entry = str(ticker)
    df = quandl_get_full_table(entry)
    df = df[['Adj. Close']]
    df['Prediction'] = df[['Adj. Close']].shift(-days_to_forecast)
    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    forecast_frame = X[-days_to_forecast:]
    X = X[:-days_to_forecast]
    y = np.array(df['Prediction'])
    y = y[:-days_to_forecast]
    return X, forecast_frame, y


def graph_data(x, y):
    plt.plot(x, y)
    plt.xticks(rotation='vertical')
    plt.show()


def perform_linear_regression(X, y, X_prediction):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("confidence score for Linear Regression: ", confidence)
    forecast_prediction = clf.predict(X_prediction)
    return confidence, forecast_prediction


# Support Vector Regression as a test to see which performs better
def perform_svr(X, y, X_prediction):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    clf = SVR()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("confidence score for Support Vector Regression:", confidence)
    forecast_prediction = clf.predict(X_prediction)
    return confidence, forecast_prediction


# Login for the robinstocks api, requires current account and 2FA
def login_robinhood():
    user_name = input("Enter User Name: ")
    password = input("Enter Password: ")
    totp = input("Enter Google Authentication Key.")
    login = r.login(user_name, password, mfa_code=totp)


# Polls robinstocks api for current holdings
def get_holdings():
    my_stocks = r.build_holdings()
    for key, value in my_stocks.items():
        print(key, value)


# ticker = stock ticker symbol
# expDate = expiry date for desired option list
# type = 'call' or 'put'
# Returns the options data for the selected date, example object
# included in the body for reference
# Must be a valid date for option listings or it will return nothing
# For most stocks the third friday of the month.
def get_options_data(ticker, exp_date, option_type):
    option_data = r.find_options_by_expiration(
        ticker, expirationDate=exp_date, optionType=option_type
    )
    # Example object returned from inquiry (label: value)
    # {'chain_id': '8b070ab5-16f5-417c-8abc-0a1b81cd3349',
    # 'chain_symbol': 'LI',
    # 'created_at': '2021-01-28T03:11:19.311835Z',
    # 'expiration_date': '2021-03-12',
    # 'id': '01f0ae8b-3cfb-4099-87cc-2b87d4f390fc',
    # 'issue_date': '2020-08-11',
    # 'min_ticks': {'above_tick': '0.05', 'below_tick': '0.01', 'cutoff_price': '3.00'},
    # 'rhs_tradability': 'untradable',
    # 'state': 'active',
    # 'strike_price': '32.5000',
    # 'tradability': 'tradable',
    # 'type': 'call',
    # 'updated_at': '2021-01-28T03:11:19.311841Z',
    # 'url': 'https://api.robinhood.com/options/instruments/01f0ae8b-3cfb-4099-87cc-2b87d4f390fc/',
    # 'sellout_datetime': '2021-03-12T20:00:00+00:00',
    # 'adjusted_mark_price': '0.010000', 'ask_price': '0.100000',
    # 'ask_size': 68,
    # 'bid_price': '0.000000',
    # 'bid_size': 0,
    # 'break_even_price': '32.510000',
    # 'high_price': '0.080000',
    # 'instrument': 'https://api.robinhood.com/options/instruments/01f0ae8b-3cfb-4099-87cc-2b87d4f390fc/',
    # 'instrument_id': '01f0ae8b-3cfb-4099-87cc-2b87d4f390fc', 'last_trade_price': '0.050000',
    # 'last_trade_size': 5,
    # 'low_price': '0.050000',
    # 'mark_price': '0.050000',
    # 'open_interest': 621,
    # 'previous_close_date': '2021-03-04',
    # 'previous_close_price': '0.030000',
    # 'volume': 9,
    # 'symbol': 'LI',
    # 'occ_symbol': 'LI    210312C00032500',
    # 'chance_of_profit_long': '0.005984',
    # 'chance_of_profit_short': '0.994016', 'delta': '0.009154',
    # 'gamma': '0.007239',
    # 'implied_volatility': '1.095589',
    # 'rho': '0.000038',
    # 'theta': '-0.006006',
    # 'vega': '0.000767',
    # 'high_fill_rate_buy_price': '0.080000',
    # 'high_fill_rate_sell_price': '0.000000',
    # 'low_fill_rate_buy_price': '0.030000',
    # 'low_fill_rate_sell_price': '0.050000'}

    for item in option_data:
        print(' price -', item['strike_price'], ' exp - ', item['expiration_date'],
              '˓ symbol - ', item['chain_symbol'], ' delta - ', item['delta'],
              ' theta - ', item['theta'])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initialize_quandl()
    # login_robinhood()
    date, price = quandl_get_table('WMT')
    graph_data(date, price)
    X, X_forecast, y = prepare_data('WMT')
    lr_confidence, forecast_lr = perform_linear_regression(X, y, X_forecast)
    svr_confidence, forecast_svr = perform_svr(X, y, X_forecast)
    x_list = list(range(0, days_to_forecast))
    plt.plot(x_list, forecast_lr)
    plt.plot(x_list, forecast_svr)
    plt.xticks(rotation='vertical')
    plt.show()
    # get_options_data('AMZN', '2021-06-18', 'call')

