import numpy as np
from typing import Tuple
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import time

def classify_two_gauss_data(
        num_samples:int, 
        noise:float, 
        radius:int = 6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points with two gaussian centers
    """
    variance = radius * noise + 0.5
    n = num_samples // 2

    def gen_gauss(cx, cy, label):
        x = cx + np.sqrt(variance) * np.random.randn(n)
        y = cy + np.sqrt(variance) * np.random.randn(n)
        label = np.ones(n) * label
        return x, y, label

    x_pos, y_pos, label_pos = gen_gauss(2, 2, 1)
    x_neg, y_neg, label_neg = gen_gauss(-2, -2, -1)

    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    label = np.concatenate([label_pos, label_neg])

    return x, y, label

class Animator:
    def __init__(self):
        """
        Make your boring print() into a beautiful animation

        Usage
        -----
        >>> from animator import Animator
        >>> import numpy as np
        >>> ani = Animator()
        >>> num_iterations = 20
        >>> for i in range(num_iterations):
        >>>     x = np.random.rand(10)
        >>>     y = np.random.rand(10)
        >>>     ani.ax.scatter(x, y)
        >>>     ani.render(0.05) 
        >>> ani.close()
        """
        self.fig, self.ax = plt.subplots()  
    
    def render(self, delay=0.05):
        clear_output(wait=True) # Clear output for dynamic display
        display(self.fig)       # Reset display
        time.sleep(delay)       

    def clear(self):
        self.ax.cla()

    def close(self):
        plt.close(self.fig)