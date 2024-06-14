import re

import csv


# The Basic of Text Files

# 'r' means read-only, it's assumed if you leave it out
file_for_reading = open("reading_file.txt", "r")
file_for_reading_2 = open("reading_file.txt")

# 'w' is write -- will destroy the file if it already exists!
file_for_writing = open("writing_file.txt", "w")

# don't forget to close your files when you're done
file_for_writing.close()

# Because it is easy to forget to close your files, you shold always use them in
# a ```with``` block, at the end of which they will be closed automatically:
def function_that_gets_data_from(f):
    pass

filename = ""
with open(filename) as f:
    data = function_that_gets_data_from(f)

# If you need to read a whole text file, you can just iterate over the lines of
# the file using for:
starts_with_hash = 0
with open("input.txt") as f:
    for line in f:
        if re.match("^#", line):
            starts_with_hash += 1

def get_domain(email_address: str) -> str:
    """
    Split on '@' and return the last piece.
    """
    
    return email_address.lower().split("@")[-1]


# Delimited Files

# For example, if we had a tab-delimited file of stock prices:

# 6/20/2014   AAPL    90.91
# 6/20/2014   MSFT    41.68
# 6/20/2014   FB  64.5
# 6/19/2014   AAPL    91.86
# 6/19/2014   MSFT    41.51
# 6/19/2014   FB  64.34

# we could process them with:
with open("tab_delimited_stock_prices.txt") as f:
    tab_reader = csv.reader(f, delimiter="\t")
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])

# If your file has headers:

# date:symbol:closing_price
# 6/20/2014:AAPL:90.91
# 6/20/2014:MSFT:41.68
# 6/20/2014:FB:64.5

# you can either skip the header row with an initial call toe reader.next or get
# each row at a dict (with the header as keys) by using csv.DictReader:
with open("colon_delimited_stock_prices.txt") as f:
    colon_reader = csv.DictReader(f)
    for dict_row in colon_reader:
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = dict_row["closing_price"]

# Even if your file doesn't have headers, you can still use ```DictReader``` by
# passing it the keys as ```fieldnames``` parameter.
todays_prices = {"AAPL": 90.91, "MSFT": 41.68, "FB": 64.5}
with open("comma_delimited_stock_prices.txt", "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    for stock, price in todays_prices:
        csv_writer.writerow([stock, price])

# csv.writer will do the right thing if your fields themselves have commas in
# them. Your own hand-rolled writer probably won't.