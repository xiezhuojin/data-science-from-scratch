from typing import NamedTuple
import datetime
from collections import namedtuple


# Python includes a namedtuple class, which is like a tuple but with named slots.

StockPrice = namedtuple("StockPrice", ["symbol", "data", "closing_price"])
price = StockPrice("MSFT", datetime.date(2010, 12, 14), 106.83)

assert price.symbol == "MSFT"
assert price.closing_price == 106.83

# Like regular tuples, namedtuples are immutable, which means that you can't modiy 
# their values once they're created. Occasionally this will get in our way, but 
# mostly that's a good thing..

# You'll notice that we still haven't solved the type annotatin issue. We do that 
# by using the variant, NamedTuple

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """
        It's a class, so we can add methods too.
        """

        return self.symbol in ["MSFT", "GOOG", "FB", "AMZN", "AAPL"]

price = StockPrice("MSFT", datetime.date(2018, 12, 14), 106.83)
assert price.symbol == "MSFT"
assert price.closing_price == 106.83
assert price.is_high_tech()