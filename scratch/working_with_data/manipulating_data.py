from typing import DefaultDict, Dict, List, NamedTuple
from datetime import date
from collections import defaultdict

from scratch.working_with_data.cleaning_and_munging import data, StockPrice


# Suppose we want to know the hightest-ever closing price for AAPL.
max_appl_price = max([sp.closing_price for sp in data if sp.symbol == "AAPL"])

# We might want to know the highest-ever closing price for each stock in our dataset.
max_prices: DefaultDict[str, float] = defaultdict(lambda: float("-inf"))
for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

# What are largest and smallest one-day percent chagnes in our dataset.
prices: Dict[str, List[StockPrice]] = defaultdict(list)
for sp in data:
    prices[sp.symbol].append(sp)

prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: date
    pct_change: float

def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """
    Assume prices are for one stock and are in order.
    """

    return [DailyChange(today.symbol, today.date, pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]

all_changes = [change
               for symbol_prices in prices.values()
               for change in day_over_day_changes(symbol_prices)]

# We can now use this new all_changes dataset to find which month is the best to 
# invest in tech stocks. We'll just at the average daily change by month:
changes_by_month: Dict[str, List[DailyChange]] = {month: [] for month in range(1, 13)}

for change in all_changes:
    changes_by_month[change.date.month].append(change.pct_change)

avg_daily_change = {
    month: sum([change.pct_change for change in changes]) / len(changes)
    for month, changes in changes_by_month.items()
}