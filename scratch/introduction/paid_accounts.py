years_of_experience_and_paid_accounts = [
    (0.7, "paid"),
    (1.9, "unpaid"),
    (2.5, "paid"),
    (4.2, "unpaid"),
    (6.0, "unpaid"),
    (6.5, "unpaid"),
    (7.5, "unpaid"),
    (8.1, "unpaid"),
    (8.7, "paid"),
    (10.0, "paid"),
]

def predict_paid_or_unpaid(year_experience) -> str:
    if year_experience < 3.0:
        return "paid"
    elif year_experience < 8.5:
        return "unpaid"
    else:
        return "paid"