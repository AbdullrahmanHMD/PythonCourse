import numpy as np 
import pandas as pd
import datetime
import os

from dateutil.relativedelta import relativedelta
# The API to retirieve the data from.
from forex_python.converter import CurrencyRates

def getData(numberOfPoints, jump):

    c = CurrencyRates()
    date = datetime.date.today()

    dates = []
    tl_rates = []
    # Doing the same query numberOfPoints times since
    # this API does not support getting the data in
    # one query.
    # Note: this may take about 20 - 25 seconds to get
    # the data.
    for i in range(numberOfPoints):
        date += datetime.timedelta(days=-jump)
        dates.append(date)
        tl_rates.append(c.get_rate('USD', 'TRY', date))
    
    return np.array(dates), np.array(tl_rates)


number_of_days = 100
jump = 1

dates, rates = getData(number_of_days, jump)
day_numbering = [(number_of_days - i) * jump for i in range(1, number_of_days + 1)]

data = np.stack((dates, day_numbering, rates), axis=1)

path = os.path.abspath(os.getcwd()) + "/data.csv"

pd.DataFrame(data, columns=['Date', 'Day #', 'TL value in USD']).to_csv(path)
