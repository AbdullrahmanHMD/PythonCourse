from purchasable import Purchasable
import random as rd
class Stock(Purchasable):

    def __init__(self, price, symbol):
        super().__init__(price, symbol)

    def getPrice(self):
        return rd.uniform(0.5 * self.price, 1.5 * self.price)