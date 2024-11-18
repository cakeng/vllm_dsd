import numpy as np  
import math

imin = 0.5
imax = 15.0
isum = 120.0

func = lambda x: x
# func = lambda x: math.exp(x)
sstep = 0.001
smax = 1.0
samples = np.array([func(sstep * i) for i in range(1, int(smax/sstep + 0.5) + 1, 1)])
samples = samples - min(samples)
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