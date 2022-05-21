from math import sin
from random import randint

import matplotlib.pyplot as plt

DX = 0.1
x = [DX * i for i in range(20)]
y = [sin(x_) for x_ in x]

# generowanie próbek
samples = []
outputs = []
for i in range(100):
    xx = DX * randint(0,3600)
    samples.append([sin(xx + i * DX) for i in range(3)])
    outputs.append(sin(xx + 3 * DX))

for s, o in zip(samples, outputs):
    print(s, o)




plt.scatter(x,y)
plt.show()