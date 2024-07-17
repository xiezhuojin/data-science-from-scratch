import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import dot
from scratch.logistic_regression.the_logistic_function import logistic
from scratch.logistic_regression.applying_the_model import beta_unscaled, x_test, y_test

# We haven’t yet used the test data that we held out.  Let’s see what happens 
# if we predict paid account whenever the probability exceeds 0.5:

true_postives = false_positives = true_negatives = false_nagatives = 0

for x_i, y_i in zip(x_test, y_test):
    prediction = logistic(dot(beta_unscaled, x_i))

    if y_i == 1 and prediction >= 0.5:
        true_postives += 1
    elif y_i == 1:
        false_nagatives += 1
    elif prediction >= 0.5:
        false_positives += 1
    else:
        true_negatives += 1

precision = true_postives / (true_postives + false_positives)
recall = true_postives / (true_postives + false_nagatives)

print(precision)
print(recall)

predictions = [logistic(dot(beta_unscaled, x_i)) for x_i in x_test]
plt.scatter(predictions, y_test, marker="+")
plt.xlabel("predicated probability")
plt.ylabel("actual outcome")
plt.title("Logistic Regression Predicated vs. Actual")
plt.show()
