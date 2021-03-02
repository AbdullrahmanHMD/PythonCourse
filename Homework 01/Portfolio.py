class Portfolio():

    def __init__(self):
        self.cash = 0.0

        self.stocks = {} 
        self.mutual_funds = []

        self.transaction_history = []
        self.transaction_number = 0
#-------------------------------------------------------------------------------------------------

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
            
#-------------------------------------------------------------------------------------------------

    def buyStock(self, amount, stock):
        amount_to_pay = amount * stock.price

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay

            if stock.symbol in self.stocks.keys():
                self.stocks[stock.symbol] = self.stocks[stock.symbol] + amount
            else :
                self.stocks[stock.symbol] = amount

            print("Stock/s successfully purchased!")

            self.addTransaction("{} {} Stock/s bought".format(amount,stock.symbol), amount_to_pay)
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

    def sellStock(self, symbol, amount):
        if amount > self.stocks[symbol]:
            print("No enough stocks to be sold!")
        else:
            if symbol in self.stocks.keys():
                self.stocks[symbol] = self.stocks[symbol] - amount
                self.cash += 1
            else :
                print("No such stock available")
                

    def addTransaction(self, message, amount):
        self.transaction_history.append((self.transaction_number, "{} for ${}".format(message, "{:.2f}".format(amount))))
        self.transaction_number += 1

    def history(self):
        transaction_hist = sorted(self.transaction_history, key=lambda x : x[0])
        for t in transaction_hist:
            print(t)
    
    def __str__(self):
        return "\nBalance: {}\nPurchased Stocks: {}\nPurchased Mutual Funds: {}\n".format(self.cash, self.stocks, self.mutual_funds)

    def __repr__(self):
        return self.__str__()