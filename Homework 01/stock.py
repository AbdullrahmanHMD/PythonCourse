from purchasable import Purchasable
from random import uniform

class Stock(Purchasable):
    
    def __init__(self, price, symbol):
        super().__init__(uniform(0.5 * price, 1.5 * price), symbol)
