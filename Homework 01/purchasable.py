class Purchasable():

    def __init__(self, price, symbol):
        self.price = price 
        self.symbol = symbol

    def getPrice(self):
        raise NotImplementedError("Sublass must implement getPrice()")

    def __str__(self):
        return "Price: {}\nTicker Symbol: {}".format(self.price, self.symbol)

    def __repr__(self):
        return "Price: {}\nTicker Symbol: {}".format(self.price, self.symbol)