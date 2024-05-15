import numpy as np
import matplotlib.pyplot as plt
from config import COLOR_FILE, DEPTH_FILE

COLS = 4

rgb_np = np.load(COLOR_FILE)
img_n = rgb_np.shape[0]

rows = img_n // COLS if img_n % COLS == 0 else img_n // COLS + 1
fig = plt.figure(figsize=(20, 16))
for i in range(rgb_np.shape[0]):
    ax = fig.add_subplot(rows, COLS, i + 1)
    ax.set_title(str(i))
    plt.imshow(rgb_np[i])

if DEPTH_FILE is not None:
    depth_np = np.load(DEPTH_FILE)
    img_n = depth_np.shape[0]
    rows = img_n // COLS if img_n % COLS == 0 else img_n // COLS + 1
    fig = plt.figure(figsize=(20, 16))
    CMAP= plt.cm.gray_r
    # CMAP= plt.cm.gnuplot
    for i in range(depth_np.shape[0]):
        ax = fig.add_subplot(rows, COLS, i + 1)
        ax.set_title(str(i))
        shw = ax.imshow(depth_np[i], cmap=CMAP)
    bar = plt.colorbar(shw) 

   
plt.show()