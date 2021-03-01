from purchasable import Purchasable
from random import uniform

UPPER_BOUND, LOWER_BOUND = 1.2, 0.9  

class MutualFund(Purchasable):

    def __init__(self, symbol):
        super().__init__(uniform(LOWER_BOUND, UPPER_BOUND), symbol)
