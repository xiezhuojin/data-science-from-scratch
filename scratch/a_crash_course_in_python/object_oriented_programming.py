class CountingClicker:
    """
    A class can/should have a docstirng, just like a function.
    """

    def __init__(self, count=0) -> None:
        self.count = count

    def __repr__(self) -> str:
        return f"CountingClcker(count={self.count})"

    def click(self, num_times=1):
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0


clicker1 = CountingClicker()
clicker2 = CountingClicker(100)
clicker3 = CountingClicker(count=100)

clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker shold be back to 0"


class NoResetCLicker(CountingClicker):
    """
    This class has all the same methods as CountingCLicker.
    """
    
    def reset(self):
        pass