import numpy as np

a = [2, 2, 1]

ans = max(set(a), key=a.count)
print(ans)
