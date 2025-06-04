import numpy as np


def test_histogram_counts_and_centers():
    x2 = np.array([2.6, 2.5, 0.8, 2.8, 2.4, 2.1, 2.3, 3.4, 1.7, 1.7, 2.2, 1.3, 3.4, 2.3, 1.6, 1.4])
    n_classes = 5
    hist, bin_edges = np.histogram(x2, bins=n_classes)
    class_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    expected_hist = np.array([2, 4, 4, 4, 2])
    expected_centers = np.array([1.06, 1.58, 2.1, 2.62, 3.14])

    assert np.allclose(hist, expected_hist)
    assert np.allclose(class_centers, expected_centers)
