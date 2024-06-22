from typing import List
import time

import tqdm


# Frequently we' ll end up doing computations that take a long time. When you're 
# doing such work, you'd like to know that you're making progress and how long 
# you shold expect to wait.

for i in tqdm.tqdm(range(100)):
    time.sleep(0.1)

def primes_up_to(n: float) -> List[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
            t.set_description(f"{len(primes)} primes")

    return primes

my_primes = primes_up_to(100_000)