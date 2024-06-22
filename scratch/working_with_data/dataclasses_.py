import datetime
from dataclasses import dataclass


# Dataclasses are (sort of) a mutable version of NamedTuple. (I say "sort of" 
# because NamedTuples represent their data compactly as tuples, whereas dataclasses 
# are regular Python classes that simply generate methods for you automatically.)

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """
        It's a class, so we can add methods too.
        """

        return self.symbol in ["MSFT", "GOOG", "FB", "AMZN", "AAPL"]

price2 = StockPrice2("MSFT", datetime.date(2018, 12, 14), 106.83)
assert price2.symbol == "MSFT"
assert price2.closing_price == 106.83
assert price2.is_high_tech()

# As mentioned, the big difference is that we can modify a dataclass instance's 
# values:

price2.closing_price /= 2
assert price2.closing_price == 53.415