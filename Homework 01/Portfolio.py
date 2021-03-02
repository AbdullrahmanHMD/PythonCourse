class Portfolio():

    def __init__(self):
        self.cash = 0.0

        self.stocks = {} # A dictionary that keeps stocks' ticker symbols as keys and the amount of stocks as values, i.e. a stocks histogram.
        self.mutual_funds = {} # A dictionary that keeps mutual funds' ticker symbols as keys and the amount of shares as values, i.e. a mutual funds histogram.

        self.stocks_prices = {} # A dictionary to keep track of the stocks' prices.
        self.mutual_funds_prices = {} # A dictionary to keep track of the mutual funds' prices.

        self.transaction_history = [] # An array to store the transactions.
        self.transaction_number = 0 # A counter that is used as a purchase history indicator.

    def addCash(self, cash_amount):
        self.cash += cash_amount
        print("${} added successfully. Current balance: {}".format(cash_amount, "{:.2f}".format(self.cash)))
        self.addTransaction("Cash added", cash_amount)

    def withdrawCash(self, amount):                                 
        if amount <= self.cash: # Checks if there is enough cash to withdraw from.
            self.cash -= amount 
            print("${} withdrawn successfully. Current balance: {}".format(amount, "{:.2f}".format(self.cash)))
            self.addTransaction("Cash withdrawn", - amount)
        else :
            print("Cash withdrawal failed: not enough cash")

    def buyStock(self, amount, stock):
        amount_to_pay = amount * stock.price

        if self.cash >= amount_to_pay: # Checks if there is enough cash to buy the given stocks
            self.cash -= amount_to_pay

            self.stocks_prices[stock.symbol] = stock.price

            if stock.symbol in self.stocks.keys():  # Checks if the type of the given stock was purchased before.
                self.stocks[stock.symbol] = self.stocks[stock.symbol] + amount
            else :
                self.stocks[stock.symbol] = amount

            print("Stock/s successfully purchased!")

            self.addTransaction("{} {} Stock/s bought".format(amount,stock.symbol), amount_to_pay) # Adds the transaction to the transaction history.
        else : 
            print("Transaction failed: not enough cash")

    def buyMutualFund(self, shares, mutual_fund):
        amount_to_pay = 1 / shares

        if self.cash >= amount_to_pay: # Checks if there is enough cash to buy the given shares
            self.cash -= amount_to_pay

            self.mutual_funds_prices[mutual_fund.symbol] = mutual_fund.price

            if mutual_fund.symbol in self.mutual_funds.keys(): # Checks if the type of the given shares was purchased before.
                self.mutual_funds[mutual_fund.symbol] = self.mutual_funds[mutual_fund.symbol] + shares
            else :
                self.mutual_funds[mutual_fund.symbol] = shares

            print("Mutual fund/s successfully purchased!")
            self.addTransaction("{} {} Mutual funds bought".format(shares, mutual_fund.symbol), amount_to_pay) # Adds the transaction to the transaction history.
        else : 
            print("Transaction failed: not enough cash")

    def sellStock(self, symbol, amount):
        if symbol in self.stocks.keys(): # Checks if the requested stock is available.
            if amount <= self.stocks[symbol]: # Checks if there is enough stocks of the requested type.
                gain_amount = self.stocks_prices[symbol] * amount # Calculates the amount of gain.

                self.stocks[symbol] = self.stocks[symbol] - amount
                self.cash += gain_amount

                self.addTransaction("{} {} stock/s sold".format(amount, symbol), gain_amount) # Adds the transaction to the transaction history.
            else:
                print("No enough stocks to be sold!")
        else :
            print("No such stock available")

    
    def sellMutualFund(self, symbol, shares):

        if symbol in self.mutual_funds.keys(): # Checks if the requested mutual funds are available.
            if shares <= self.mutual_funds[symbol]: # Checks if there is enough shares of the requested type.
                gain_amount = self.mutual_funds_prices[symbol] * shares # Calculates the amount of gain.

                self.mutual_funds[symbol] = self.mutual_funds[symbol] - shares
                self.cash += gain_amount

                self.addTransaction("{} {} shares sold".format(shares, symbol), gain_amount) # Adds the transaction to the transaction history.
            else:
                print("No enough shares to be sold!")
        else :
            print("No such mutual funds available") 

    # addTransaction: A helper function to add a certain transaction to the transaction_history.
    def addTransaction(self, message, amount):
        # Adds a tuple that consists of the transaction number (That simulates the history of adding a transaction)
        # and information about the transaction
        self.transaction_history.append((self.transaction_number, "{} | ${}".format(message, "{:.2f}".format(abs(amount)))))
        self.transaction_number += 1

    def history(self):
        # Sorts the transactions according to the order they were added in.
        transaction_hist = sorted(self.transaction_history, key=lambda x : x[0])

        print("\n----Transaction History----\n")
        for t in transaction_hist:
            print(t)
        print("\n--------------------------\n")
    
    def __str__(self):
        return "\nBalance: {}\nPurchased Stocks: {}\nPurchased Mutual Funds: {}\n".format("{:.2f}".format(self.cash), self.stocks, self.mutual_funds)

    def __repr__(self):
        return self.__str__()