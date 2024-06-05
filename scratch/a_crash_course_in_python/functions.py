def double(x):
    """
    This is where you put an optional docstring that explains what the function 
    does. For example, tis function multiple its input by 2.
    """
    return x * 2

def apply_to_one(f):
    """
    Calls the function f with 1 as its argument.
    """
    return f(1)

my_double = apply_to_one(double)

y = apply_to_one(lambda x: 2 * x)

def my_print(message="my default message"):
    print(message)

my_print("double")
my_print()

def full_name(first="What's-his-name", last="Something"):
    return first + " " + last

full_name("Joel", "Grus")
full_name("Joel")
full_name(last="Grus")