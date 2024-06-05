import math
from collections import Counter

from scipy.stats import beta, bernoulli
import numpy as np
import matplotlib.pyplot as plt

# 与之前的推断不同，贝叶斯推断把未知参数看成随机变量，然后以一个先验带入，之后使用观察值与贝叶斯
# 理论更新后验分布。

# 例如，当一个位置变量是一个概率（以抛硬币微粒子），我们总是用一个beta分布作为先验（主要用于一
# 个变量在(0,1)区间内的分布情况）。

def B(alpha: float, beta: float) -> float:
    """
    A normalizing constant so that the total probability is 1.
    """

    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

# same as above
beta

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

# same as above
beta.pdf

xs = np.arange(0, 1, 0.01)
a1_b1_ys = [beta.pdf(x, 1, 1) for x in xs]
a10_b10_ys = [beta.pdf(x, 10, 10) for x in xs]
a4_b16_ys = [beta.pdf(x, 4, 16) for x in xs]
a16_b4_ys = [beta.pdf(x, 16, 4) for x in xs]

plt.plot(xs, a1_b1_ys, label="Beta(1, 1)")
plt.plot(xs, a10_b10_ys, label="Beta(10, 10)")
plt.plot(xs, a4_b16_ys, label="Beta(4, 16)")
plt.plot(xs, a16_b4_ys, label="Beta(16, 4)")
plt.legend()
plt.show()

# 假设对p有一个先验分布，我们对硬币的公平性不站队，我们就假设其Beta分布的alpha、beta为1。
# 之前我们抛硬币，贝叶斯理论告诉我们p的后验分布是Beta分布，其alpha = pri_alpha + h,
# beta = pri_beta + t

# 一直做update
guest_alpha = 20
guest_beta = 20
for _ in range(1000):
    flips = bernoulli.rvs(p=0.6, size=1000)
    counter = Counter(flips)
    heads = counter[1]
    tails = counter[0]
    guest_alpha = guest_alpha + heads
    guest_beta = guest_beta + tails
beta_mean = beta.mean(guest_alpha, guest_beta)
print(beta_mean)

xs = np.arange(0, 1, 0.01)
ys = [beta.pdf(x, guest_alpha, guest_beta) for x in xs]
plt.plot(xs, ys)
plt.show()