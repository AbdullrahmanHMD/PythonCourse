class Portfolio():

    def __init__(self):
        self.cash = 0.0

        self.stocks = []
        self.mutual_funds = []

        self.transaction_history = []
        self.transaction_number = 0

    def addCash(self, cash_amount):
        self.cash += cash_amount
        print("${} added successfully. Current balance: {}".format(cash_amount, self.cash))
        self.addTransaction("Cash added", cash_amount)

    def withdrawCash(self, amount):                                 
        if amount <= self.cash:
            self.cash -= amount 
            print("${} withdrawn successfully. Current balance: {}".format(amount, self.cash))
            self.addTransaction("Cash withdrawn", - amount)
        else :
            print("Cash withdrawal failed: not enough cash")

    def buyStock(self, amount, stock):
        amount_to_pay = amount * stock.price

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay
            self.stocks.append(stock)

            print("Stock/s {} successfully purchased!".format(stock))
            self.addTransaction("Stock {} bought".format(stock), amount_to_pay)
        else : 
            print("Transaction failed: not enough cash")

    def buyMutualFund(self, share, mutual_fund):
        amount_to_pay = 1 / share

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay
            self.mutual_funds.append(mutual_fund)

            print("Mutual fund/s {} successfully purchased!".format(mutual_fund))
            self.addTransaction("Mutual fund {} bought".format(mutual_fund), amount_to_pay)
        else : 
            print("Transaction failed: not enough cash")

    def sellStock(self, amount, stock):
        amount_to_gain = amount * stock.price

        if self.cash >= amount_to_gain:
            self.cash += amount_to_gain
            self.stocks.append(stock)

            print("Stock/s {} successfully sold!".format(stock))
            self.addTransaction("Stock {} sold".format(stock), amount_to_gain)
        else : 
            print("Transaction failed: not enough cash")

    def addTransaction(self, message, amount):
        self.transaction_history.append((self.transaction_number, "{} | {}".format(message, amount)))

    def history(self):
        print(sorted(self.transaction_history, key=lambda x : x[0]))

    def __str__(self):
        return "\nBalance: {}\nPurchased Stocks: {}\nPurchased Mutual Funds: {}\n".format(self.cash, self.stocks, self.mutual_funds)

    def __repr__(self):
        return self.__str__()