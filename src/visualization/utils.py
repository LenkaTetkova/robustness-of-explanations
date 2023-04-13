import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap


def make_cmap():
    top = mpl.colormaps['Reds_r'].resampled(128)
    bottom = mpl.colormaps['Greens'].resampled(128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='GreenRed')
    return newcmp
