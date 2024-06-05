from collections import Counter

from matplotlib import pyplot as plt


movies = ["Annie Hall", "Ben-Hur", "casablanca", "Gandhi", "Wst Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# plot bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
plt.bar(movies, num_oscars)

plt.title("My Favourte Movies")
plt.ylabel("# of Academy Awards")

plt.show()

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# bucket grades by decile, but put 100 in with the 90s
histogram = Counter([min(grade // 10 * 10, 90) for grade in grades])

plt.bar([x + 5 for x in histogram.keys()],
        histogram.values(),
        10, edgecolor=(0, 0, 0))
plt.axis([-5, 105, 0, 5])

plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

# When creating bar charts it is considered especially bad form for your y-axis 
# not to start at 0, since this is an easy way to mislead people.
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

plt.ticklabel_format(useOffset=False)
# misleading y-axis only shows the part above 500
plt.axis((2016.5, 2018.5, 499, 506))
plt.title("Look at the 'Huge' Increase!")
plt.show()

plt.bar(years, mentions)
plt.axis((2016.5, 2018.5, 0, 550))
plt.title("Not so HUge Anymore")
plt.show()