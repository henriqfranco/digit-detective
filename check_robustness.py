import numpy as np

from experiment import SEED
from robustness import DIRECTIONS, add_noise, shift_one_pixel


def check_robustness():
    images = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
    original = images.copy()
    expected_first = {
        "left": [[2, 3, 0], [5, 6, 0], [8, 9, 0]],
        "right": [[0, 1, 2], [0, 4, 5], [0, 7, 8]],
        "up": [[4, 5, 6], [7, 8, 9], [0, 0, 0]],
        "down": [[0, 0, 0], [1, 2, 3], [4, 5, 6]],
    }
    for direction in DIRECTIONS:
        shifted = shift_one_pixel(images, direction)
        np.testing.assert_array_equal(shifted[0], expected_first[direction])
        np.testing.assert_array_equal(shifted[1], shift_one_pixel(images[1:2], direction)[0])
        assert shifted.shape == images.shape and not np.shares_memory(shifted, images)
        assert np.isfinite(shifted).all() and 0 <= shifted.min() <= shifted.max() <= 16
    try:
        shift_one_pixel(images, "diagonal")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown direction must be rejected")

    pixels = np.array([[[0, 8, 16]]], dtype=int)
    noise = np.array([[[-2.0, 0.5, 2.0]]])
    np.testing.assert_array_equal(add_noise(pixels, noise, 2), [[[0, 9, 16]]])
    np.testing.assert_array_equal(add_noise(pixels, noise, 0), pixels)
    np.testing.assert_array_equal(add_noise(pixels, np.zeros_like(noise), 4), pixels)
    np.testing.assert_array_equal(add_noise(pixels, noise, 1), [[[0, 8.5, 16]]])
    np.testing.assert_array_equal(pixels, [[[0, 8, 16]]])
    np.testing.assert_array_equal(noise, [[[-2.0, 0.5, 2.0]]])
    random_noise = np.random.default_rng(SEED).normal(size=images.shape)
    repeated_noise = np.random.default_rng(SEED).normal(size=images.shape)
    np.testing.assert_array_equal(random_noise, repeated_noise)
    for sigma in (1, 2, 4):
        corrupted = add_noise(images, random_noise, sigma)
        np.testing.assert_array_equal(corrupted, add_noise(images, repeated_noise, sigma))
        assert corrupted.shape == images.shape and np.issubdtype(corrupted.dtype, np.floating)
        assert np.isfinite(corrupted).all() and 0 <= corrupted.min() <= corrupted.max() <= 16
        assert not np.shares_memory(corrupted, images)
    np.testing.assert_array_equal(images, original)
    print("Robustness checks passed: exact shifts, zero fill, no wrapping/mutation, clipping, float noise, and repeatability.")


if __name__ == "__main__":
    check_robustness()
