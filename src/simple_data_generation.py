import numpy as np
import os
from pathlib import Path

N = 1
DESTINATION_PATH = Path("data/simple")
SEED = 42
RANDOM_CHANGES_NUMBER = 4


def get_parabola(range=1, steps=1000, steepness=1, shift_x=0, shift_y=0):
    parabola = np.linspace(-range, range, steps)
    parabola = np.pow(parabola - shift_x, 2) * steepness
    parabola = parabola + shift_y
    return parabola


def get_sine_wave(range=1, steps=1000, frequency=1, amplitude=1, shift_x=0, shift_y=0):
    x = np.linspace(-range, range, steps)
    sine_wave = np.sin((x - shift_x) * frequency) * amplitude + shift_y
    return sine_wave


def get_linear(range=1, steps=1000, slope=1, shift_x=0, shift_y=0):
    linear = np.linspace(-range, range, steps)
    linear = linear * slope
    linear = linear + shift_y
    return linear


def get_constant(range=1, steps=1000, value=0):
    constant = np.full(steps, value)
    return constant


def get_exponential(range=1, steps=1000, base=np.e, shift_x=0, shift_y=0):
    x = np.linspace(-range, range, steps)
    exponential = np.power(base, x - shift_x) + shift_y
    return exponential


def get_logarithmic(range=1, steps=1000, base=np.e, shift_x=1, shift_y=0):
    x = np.linspace(0.01, range, steps)  # Avoid log(0)
    logarithmic = np.log(x + shift_x) / np.log(base) + shift_y
    return logarithmic


def get_simple_composition(steps=1000, components=5):
    x = np.zeros(steps)
    for _ in range(components):
        frequency = (np.random.random() - 0.5) * 10
        amplitude = (np.random.random() - 0.5) * 2
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        x += get_sine_wave(
            frequency=frequency, amplitude=amplitude, shift_x=shift_x, shift_y=shift_y
        )
    return x


def shift_time_series(ts, shift):
    if shift >= 0:
        return ts[shift:]
    return ts[:shift]


def shrunk_time_series(ts, factor):
    new_length = int(len(ts) / factor)
    return np.interp(np.linspace(0, len(ts), new_length), np.arange(len(ts)), ts)


def save_deformed(ts, path, random_changes):
    for i, change in enumerate(random_changes):
        deformed_ts = shift_time_series(ts, int(change[0] * 100))
        deformed_ts = shrunk_time_series(deformed_ts, 1 + change[1] * 0.5)
        np.save(path / str(i), deformed_ts)


if __name__ == "__main__":
    np.random.seed(SEED)

    os.makedirs(DESTINATION_PATH, exist_ok=True)

    random_changes = (np.random.random((RANDOM_CHANGES_NUMBER, 2)) - 0.5) * 2
    random_changes = np.concat([np.zeros((1, 2)), random_changes])

    for i in range(N):
        PARABOLAS_PATH = DESTINATION_PATH.joinpath(f"parabolas_{i}")
        os.makedirs(PARABOLAS_PATH, exist_ok=True)
        steepness = (np.random.random() - 0.5) * 2
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        parabola = get_parabola(steepness=steepness, shift_x=shift_x, shift_y=shift_y)
        save_deformed(parabola, PARABOLAS_PATH, random_changes)

    for i in range(N):
        SINE_WAVES_PATH = DESTINATION_PATH.joinpath(f"sine_waves_{i}")
        os.makedirs(SINE_WAVES_PATH, exist_ok=True)
        frequency = (np.random.random() - 0.5) * 10
        amplitude = (np.random.random() - 0.5) * 2
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        sine_wave = get_sine_wave(
            frequency=frequency, amplitude=amplitude, shift_x=shift_x, shift_y=shift_y
        )
        save_deformed(sine_wave, SINE_WAVES_PATH, random_changes)

    for i in range(N):
        LINEARS_PATH = DESTINATION_PATH.joinpath(f"linears_{i}")
        os.makedirs(LINEARS_PATH, exist_ok=True)
        slope = (np.random.random() - 0.5) * 2
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        linear = get_linear(slope=slope, shift_x=shift_x, shift_y=shift_y)
        save_deformed(linear, LINEARS_PATH, random_changes)

    for i in range(N):
        CONSTANTS_PATH = DESTINATION_PATH.joinpath(f"constants_{i}")
        os.makedirs(CONSTANTS_PATH, exist_ok=True)
        value = (np.random.random() - 0.5) * 2
        constant = get_constant(value=value)
        save_deformed(constant, CONSTANTS_PATH, random_changes)

    for i in range(N):
        EXPONENTIALS_PATH = DESTINATION_PATH.joinpath(f"exponentials_{i}")
        os.makedirs(EXPONENTIALS_PATH, exist_ok=True)
        base = np.random.random() * 5 + 0.1
        shift_x = (np.random.random() - 0.5) * 2
        shift_y = (np.random.random() - 0.5) * 2
        exponential = get_exponential(base=base, shift_x=shift_x, shift_y=shift_y)
        save_deformed(exponential, EXPONENTIALS_PATH, random_changes)

    for i in range(N):
        LOGARITHMICS_PATH = DESTINATION_PATH.joinpath(f"logarithmics_{i}")
        os.makedirs(LOGARITHMICS_PATH, exist_ok=True)
        base = np.random.random() * 5 + 0.1
        shift_x = np.random.random() * 2 + 0.1  # Ensure shift_x > 0 to avoid log(0)
        shift_y = (np.random.random() - 0.5) * 2
        logarithmic = get_logarithmic(base=base, shift_x=shift_x, shift_y=shift_y)
        save_deformed(logarithmic, LOGARITHMICS_PATH, random_changes)

    for i in range(N):
        SIMPLE_COMPOSITIONS_PATH = DESTINATION_PATH.joinpath(f"simple_compositions_{i}")
        os.makedirs(SIMPLE_COMPOSITIONS_PATH, exist_ok=True)
        components = np.random.randint(3, 10)
        simple_composition = get_simple_composition(components=components)
        save_deformed(simple_composition, SIMPLE_COMPOSITIONS_PATH, random_changes)
