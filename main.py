import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

lm = LinearRegression()

data = pd.read_csv('afm_01.txt', header=None, sep='\t')
data = data * 10 ** 9
dataLen = len(data)
print(data)
plt.subplots_adjust(hspace=0.4)
plt.subplot(211)
plt.plot(data.loc[:, 1])
plt.title("Y direction")
plt.subplot(212)
plt.plot(data.loc[1, :])
plt.title("X direction")
plt.show()

Z = np.array([])
for y in range(dataLen):
    Z = np.append(Z, np.array(data.loc[y]))

z_notAlign = Z.reshape(dataLen, dataLen)


def find_slope(data, xdir):
    dataLen = len(data)
    trainX = np.arange(dataLen).reshape(-1, 1)
    intercept = np.array([])
    coef = np.array([])
    for line in range(dataLen):
        if xdir == 1:
            trainY = np.array(data.loc[line])
        else:
            trainY = np.array(data.loc[:, line])
        lm.fit(trainX, trainY)
        intercept = np.append(intercept, lm.intercept_)
        coef = np.append(coef, lm.coef_)
    return intercept, coef


intercept, coef = find_slope(data, 0)

print("Y intercept of surface is", round(np.mean(intercept), 2),
      "  deviation of Y intercept is", round(np.std(intercept), 2))
print("Y slope of surface is", round(np.mean(coef), 2),
      "  deviation of Y slope of is", round(np.std(coef), 2))

Ay = np.mean(coef)
By = np.mean(intercept)

intercept, coef = find_slope(data, 1)

print("X intercept is", round(np.mean(intercept), 2),
      "  deviation of X intercept is", round(np.std(intercept), 2))
print("X slope of aligned surface is", round(np.mean(coef), 2),
      "  deviation of X slope is", round(np.std(coef), 2))

Ax = np.mean(coef)
Bx = np.mean(intercept)


def remove_slope(arr, slope_coef, slope_intercept):
    arr2 = np.array([])
    for t in range(len(arr)):
        arr2 = np.append(arr2, arr[t] - (t * slope_coef) - slope_intercept)
    return arr2


Z_align = np.array([])
for y in range(dataLen):
    Z_align = np.append(Z_align, remove_slope(data.loc[y], Ax, Bx))
Z_align = Z_align.reshape(dataLen, dataLen)
for t in range(len(Z_align)):
    Z_align[t] = Z_align[t] - Ay * t - By

print("RMS of not aligned surface is", round(np.std(np.ravel(Z)), 2))
print("RMS of aligned surface is", round(np.std(np.ravel(Z_align)), 2))
x = np.arange(dataLen)
y = np.arange(dataLen)
x, y = np.meshgrid(range(dataLen), range(dataLen))

Z_align = Z_align - np.mean(np.ravel(Z_align))
z_notAlign = z_notAlign - np.mean(np.ravel(z_notAlign))

plt.hist(np.ravel(Z_align), 50, histtype='stepfilled',
         facecolor='g', alpha=0.75)

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(y, x, z_notAlign)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(y, x, Z_align, cmap='jet')

plt.show()
