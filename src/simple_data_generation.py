import numpy as np
import os
from pathlib import Path

N = 10
PARABOLAS_PATH = Path("data/simple/parabolas")

def get_parabola(start=1, stop=1, steps=1000, steepness=1, shift_x=0, shift_y=0):
    parabola = np.linspace(start, stop, steps)
    parabola = np.pow(parabola-shift_x, 2)
    parabola = parabola + shift_y
    return parabola

if __name__ == "__main__":
    os.makedirs(PARABOLAS_PATH, exist_ok=True)

    parabola = get_parabola()
    np.save(PARABOLAS_PATH.joinpath("1"), parabola)