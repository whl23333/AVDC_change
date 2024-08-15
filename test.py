import pandas as pd
import os
import torch
import numpy as np
from datasets import accumulate
from einops import rearrange
import matplotlib.pyplot as plt
data = pd.read_pickle("/media/disk3/WHL/flowdiffusion/datasets/metaworld/metaworld_dataset/hammer/corner/000/action.pkl")
d = accumulate(data[0])
x = [i[0] for i in d]
y = [i[1] for i in d]
z = [i[2] for i in d]
grab = [i[3] for i in d]

t = range(len(x))
# plt.figure()
# plt.plot(t, x)
# plt.savefig("/home/hlwang/AVDC_change/flowdiffusion/x_plot.png")
# plt.figure()
# plt.plot(t, y)
# plt.savefig("/home/hlwang/AVDC_change/flowdiffusion/y_plot.png")
# plt.figure()
# plt.plot(t, z)
# plt.savefig("/home/hlwang/AVDC_change/flowdiffusion/z_plot.png")
# plt.figure()
# plt.plot(t, grab)
# plt.savefig("/home/hlwang/AVDC_change/flowdiffusion/grab_plot.png")
# print("hi")

t = [1,2,3,4]
a, b, *c = t
print(a)
print(b)