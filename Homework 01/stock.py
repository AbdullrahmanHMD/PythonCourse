from purchasable import Purchasable
from random import uniform

class Stock(Purchasable):

    def __init__(self, price, symbol):
        super().__init__(price, symbol)

    def getPrice(self):
        return uniform(0.5 * self.price, 1.5 * self.price)