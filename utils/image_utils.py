import numpy as np
import matplotlib.pyplot as plt

def np_rescale(array: np.ndarray, old_low, old_high, low, high):
    array = (array - old_low) * (high - low) / (old_high - old_low) + low
    return array

def show_diff(original, edited, save_name=None):
    diff = np.abs(original - edited)
    diff = np_rescale(diff, 0, 1, 0, 255)
    diff = diff.astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.subplot(1, 3, 2)
    plt.imshow(diff)
    plt.subplot(1, 3, 3)
    plt.imshow(edited)
    if save_name is not None:
        plt.savefig(f'{save_name}.png')
    plt.show()
    plt.close()

