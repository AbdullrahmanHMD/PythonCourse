class Purchasable():

    def __init__(self, price, symbol):
        self.price = price 
        self.symbol = symbol


    def __eq__(self, other):
        return self.symbol == other.symbol

    def __str__(self):
        return "Price: {} | Ticker Symbol: {}".format(self.price, self.symbol)

    def __repr__(self):
        return self.__str__()