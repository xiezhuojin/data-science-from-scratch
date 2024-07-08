from glob import glob
from collections import Counter
import random
from typing import List

from scratch.naive_bayes.implementation import Message, NaiveBayesClassifier
from scratch.machine_learning.overfitting_and_underfitting import split_data


path = "spam_data/*/*"

data: List[Message] = []
for filename in glob(path):
    is_spam = "ham" not in filename

    # There are some garbage characters in the emails; the errors='ignore'
    # skips them instead of raising an exception.
    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break  # done with this file

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

predictions = [(message, model.predict(message.text))
               for message in test_messages]

# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)

print(confusion_matrix)

# We can also inspect the model's innards to see which words are least and most 
# indicative of spam:

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    # We probably shouldn't call priviate methods, but it's for a good cause.
    prob_if_spam, prob_if_ham = model._probabilities(token)
    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])