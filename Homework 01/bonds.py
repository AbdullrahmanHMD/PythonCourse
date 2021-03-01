from purchasable import Purchasable

class Bonds(Purchasable):

    def __init__(self, price, symbol):
        super().__init__(price, symbol)