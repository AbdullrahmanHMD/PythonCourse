import numpy as np 
import datetime
import pandas as pd
import os

from dateutil.relativedelta import relativedelta
from forex_python.converter import CurrencyRates


def getData(numberOfPoints, jump):

    c = CurrencyRates()
    date = datetime.date.today()

    dates = []
    tl_rates = []

    for i in range(numberOfPoints):
        date += datetime.timedelta(days=-jump)
        dates.append(date)
        tl_rates.append(c.get_rate('USD', 'TRY', date))
    
    return np.array(dates), np.array(tl_rates)

number_of_days = 10
dates, rates = getData(number_of_days, 100)

data = np.stack((dates, rates), axis=1)

path = os.path.abspath(os.getcwd()) + "/file.csv"

pd.DataFrame(data, columns=['Date', 'TL value in USD']).to_csv(path)
