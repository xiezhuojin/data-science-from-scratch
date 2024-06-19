from typing import Tuple
import math

from scratch.hypothesis_and_inference.p_value import two_sided_p_value

# 假设我们有两份广告词：A("tastes great!"), B("less bias!")，我们对其做A/B test，
# 若A的点击为990/1000，B的点击为10/1000，我们可以自信地任务广告词A更好，但是如果A、B的点击
# 对比不明显怎么办？这就需要用到统计推论

# 假设A的点击为nA/NA，只要NA足够大，我们就认为nA/NA是近似一个正态随机变量，
# 其中pA ～ B(1, p)的二项分布，其均值B_mu = nA / NA，标准差B_sigma = math.sqrt(1 * p * (1 - p))，
# 其对应的正态分布均值N_mu = B_mu，标准差N_sigma = math.sqrt(B_sigma ** 2 / NA)
# 同样的B点击也适用

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

# 假设A、B对应的状态分布独立，它们的差也同样为正态分布，其均值mu = mu_B - mu_A，标准差
# sigma = math.sqrt(sigma_A ^ 2 + sigma_B ^ 2)，其z

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B) -> float:
    """
    返回z score，Z = (X - mu) / sigma
    """
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    # 观测值
    x = p_B - p_A
    # 假设pA = pB
    mu = 0
    sigma = math.sqrt(sigma_A ** 2 + sigma_B ** 2)
    z_score = (x - mu) / sigma
    return z_score

z = a_b_test_statistic(1000, 200, 1000, 180)
p = two_sided_p_value(z)
print(p)

z = a_b_test_statistic(1000, 200, 1000, 150)
p = two_sided_p_value(z)
print(p)