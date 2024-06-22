from typing import List, Optional
import datetime
import re
import csv
import io

from dateutil.parser import parse

from scratch.working_with_data.using_namedtuples import StockPrice


def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol,
                      parse(date).date(),
                      float(closing_price))

# Now test our function
stock = parse_row(["MSFT", "2018-12-14", "106.03"])
assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock symbol should all be capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

# sholud return None for errors
assert try_parse_row(["MSTF0", "2018-12-14", "106.03"]) == None
assert try_parse_row(["MSTF", "2018-12--14", "106.03"]) == None
assert try_parse_row(["MSTF", "2018-12-14", "x"]) == None

# For example, if we have comma-delimited stock prices with bad data:
stock_prices = \
"""AAPL,6/20/2014,90.91
MSFT,6/20/2014,41.68
FB,6/20/3014,64.5
AAPL,6/19/2014,91.86
MSFT,6/19/2014,n/a
FB,6/19/2014,64.34
"""
mock_stock_prices = io.StringIO(stock_prices)

# we can now read and return only the valid rows:
data: List[StockPrice] = []
reader = csv.reader(mock_stock_prices)
for row in reader:
    maybe_stock = try_parse_row(row)
    if not maybe_stock:
        print(f"Skipping invalid row: {row}")
    else:
        data.append(maybe_stock)
