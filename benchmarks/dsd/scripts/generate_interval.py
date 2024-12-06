import numpy as np  
import math

imin = 6.0
imax = 6.0
isum = 120.0

def func(x):
    # step function4
    if x < 1.0:
        return 1.0
    elif x < 5.0:
        return 4.0
    else:
        return 8.0

# func = lambda x: math.exp(x)

sstep = 0.001
smax = 13
samples = np.array([func(sstep * i) for i in range(1, int(smax/sstep + 0.5) + 1, 1)])
samples = samples - min(samples) + 0.0000001
samples = samples / max(samples) * (imax - imin) + imin
samples = 1/samples
x_scale = isum / sum(samples)
new_len = int(len(samples) * x_scale + 0.5) 
# Interpolate
new_samples = np.interp(np.linspace(0, len(samples), new_len), np.arange(len(samples)), samples)

print(new_samples)
print(1/new_samples)
print(len(new_samples))
print(sum(new_samples))

with open('interval.txt', 'w') as f:
    for i in new_samples:
        f.write(str(i) + '\n')