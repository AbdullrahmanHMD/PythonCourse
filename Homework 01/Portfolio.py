class Portofolio():

    def __init__(self):
        self.cash = 0.0
        self.stocks = []
        self.mutual_funds = []

    def addCash(self, cash_amount):
        self.cash += cash_amount

    def buyStock(self, amount, stock):
        amount_to_pay = amount * stock.price

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay
            self.stocks.append(stock)

            print("Stock/s {} successfully purchased!".format(stock))
        else : 
            print("Transaction failed: not enough cash")

    def buyMutualFunds(self, amount, mutual_fund):
        amount_to_pay = amount * mutual_fund.price

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay
            self.mutual_funds.append(mutual_fund)

            print("Mutual fund/s {} successfully purchased!".format(mutual_fund))
        else : 
            print("Transaction failed: not enough cash")

    def __str__(self):
        return """Balance: {}\n
                Purchased Stocks: {}\n
                Purchased Mutual Funds{}\n""".format(self.cash, self.stocks, self.mutual_funds)
                
    def __repr__(self):
        return """Balance: {}\n
                Purchased Stocks: {}\n
                Purchased Mutual Funds{}\n""".format(self.cash, self.stocks, self.mutual_funds)