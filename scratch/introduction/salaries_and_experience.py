from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt


salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]
salaries_and_tenures.sort(key=lambda point: point[1])

salaries_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salaries_by_tenure[tenure].append(salary)

averge_salary_by_tenure = {tenure: sum(salaries) / len(salaries) for tenure, salaries in salaries_by_tenure.items()}

# pprint(averge_salary_by_tenure)

def tenure_bucket(tenure: float) -> str:
    if tenure < 2:
        return "less than two"
    elif tenure <= 5:
        return "between two and five"
    else:
        return "more than five"

salary_by_tenure_bucket = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure_bucket[tenure_bucket(tenure)].append(salary)

averge_salary_by_tenure_bucket = {bucket: sum(salaries) / len(salaries) for bucket, salaries in salary_by_tenure_bucket.items()}

# pprint(averge_salary_by_tenure_bucket)

plt.subplot(2, 1, 1)
plt.plot([tenure for _, tenure in salaries_and_tenures], [salary for salary, _ in salaries_and_tenures])
plt.plot(averge_salary_by_tenure.keys(), averge_salary_by_tenure.values())
plt.subplot(2, 1, 2)
plt.plot(averge_salary_by_tenure_bucket.keys(), averge_salary_by_tenure_bucket.values())
plt.show()