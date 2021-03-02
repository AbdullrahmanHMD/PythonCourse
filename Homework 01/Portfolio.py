class Portfolio():

    def __init__(self):
        self.cash = 0.0

        self.stocks = {} 
        self.mutual_funds = {}

        self.stocks_prices = {}
        self.mutual_funds_prices = {}

        self.transaction_history = []
        self.transaction_number = 0
#-------------------------------------------------------------------------------------------------

    def addCash(self, cash_amount):
        self.cash += cash_amount
        print("${} added successfully. Current balance: {}".format(cash_amount, "{:.2f}".format(self.cash)))
        self.addTransaction("Cash added", cash_amount)

    def withdrawCash(self, amount):                                 
        if amount <= self.cash:
            self.cash -= amount 
            print("${} withdrawn successfully. Current balance: {}".format(amount, "{:.2f}".format(self.cash)))
            self.addTransaction("Cash withdrawn", - amount)
        else :
            print("Cash withdrawal failed: not enough cash")
            
#-------------------------------------------------------------------------------------------------

    def buyStock(self, amount, stock):
        amount_to_pay = amount * stock.price

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay

            self.stocks_prices[stock.symbol] = stock.price

            if stock.symbol in self.stocks.keys():
                self.stocks[stock.symbol] = self.stocks[stock.symbol] + amount
            else :
                self.stocks[stock.symbol] = amount

            print("Stock/s successfully purchased!")

            self.addTransaction("{} {} Stock/s bought".format(amount,stock.symbol), amount_to_pay)
        else : 
            print("Transaction failed: not enough cash")

    def buyMutualFund(self, shares, mutual_fund):
        amount_to_pay = 1 / shares

        if self.cash >= amount_to_pay:
            self.cash -= amount_to_pay

            self.mutual_funds_prices[mutual_fund.symbol] = mutual_fund.price

            if mutual_fund.symbol in self.mutual_funds.keys():
                self.mutual_funds[mutual_fund.symbol] = self.mutual_funds[mutual_fund.symbol] + shares
            else :
                self.mutual_funds[mutual_fund.symbol] = shares

            print("Mutual fund/s successfully purchased!")
            self.addTransaction("{} {} Mutual funds bought".format(shares, mutual_fund.symbol), amount_to_pay)
        else : 
            print("Transaction failed: not enough cash")

    def sellStock(self, symbol, amount):
        if amount > self.stocks[symbol]:
            print("No enough stocks to be sold!")
        else:
            if symbol in self.stocks.keys():
                self.stocks[symbol] = self.stocks[symbol] - amount
                self.cash += self.stocks_prices[symbol]
            else :
                print("No such stock available")
    
    def sellMutualFund(self, symbol, shares):
        if shares > self.mutual_funds[symbol]:
            print("No enough shares to be sold!")
        else:
            if symbol in self.mutual_funds.keys():
                self.mutual_funds[symbol] = self.mutual_funds[symbol] - shares
                self.cash += self.mutual_funds_prices[symbol]
            else :
                print("No such mutual funds available")    
                

    def addTransaction(self, message, amount):
        self.transaction_history.append((self.transaction_number, "{} | ${}".format(message, "{:.2f}".format(abs(amount)))))
        self.transaction_number += 1

    def history(self):
        transaction_hist = sorted(self.transaction_history, key=lambda x : x[0])

        print("\n----Transaction History----\n")
        for t in transaction_hist:
            print(t)

        print("\n--------------------------\n")
    
    def __str__(self):
        return "\nBalance: {}\nPurchased Stocks: {}\nPurchased Mutual Funds: {}\n".format("{:.2f}".format(self.cash), self.stocks, self.mutual_funds)

    def __repr__(self):
        return self.__str__()