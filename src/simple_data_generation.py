import numpy as np
import os
from pathlib import Path

N = 10
PARABOLAS_PATH = Path("data/simple/parabolas")
SEED = 42

def get_parabola(range=1, steps=1000, steepness=1, shift_x=0, shift_y=0):
    parabola = np.linspace(-range, range, steps)
    parabola = np.pow(parabola-shift_x, 2) * steepness
    parabola = parabola + shift_y
    return parabola

if __name__ == "__main__":
    os.makedirs(PARABOLAS_PATH, exist_ok=True)

    for i in range(N):
        steepness = (np.random.random() - 0.5) * 2
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        parabola = get_parabola(steepness=steepness, shift_x=shift_x, shift_y=shift_y)
        np.save(PARABOLAS_PATH.joinpath(str(i)), parabola)