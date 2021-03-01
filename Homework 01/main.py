from portfolio import Portofolio
from stock import Stock
from mutualFund import MutualFund

p = Portofolio()

print(p.__str__)


s = Stock(15, "HFM")
m = MutualFund("LMA")

print(m.symbol)
print(m.getPrice())
