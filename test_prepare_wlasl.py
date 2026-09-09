import numpy as np

from prepare_wlasl import hand_focused_crop, square_crop, transform_hands


def test_square_crop_is_square_and_padded():
    x0, y0, side = square_crop([10, 20, 50, 40], 100, 100, 0.1)
    assert np.isclose(side, 48)
    assert np.isclose(x0, 6)
    assert np.isclose(y0, 6)


def test_mediapipe_hand_order_and_coordinate_transform():
    landmarks = np.zeros((2, 553, 3), dtype=np.float32)
    landmarks[:, :21, :2] = (0.25, 0.50)   # MediaPipe right
    landmarks[:, 21:42, :2] = (0.75, 0.25) # MediaPipe left
    hands, visible = transform_hands(landmarks, np.array([0, 1]), 200, 100,
                                     (0.0, 0.0, 200.0))
    assert hands.shape == (2, 2, 21, 3)
    assert visible.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    # Output is left then right; y is normalized by the square crop, not height.
    assert np.allclose(hands[0, 0, 0, :2], [0.75, 0.125])
    assert np.allclose(hands[0, 1, 0, :2], [0.25, 0.25])


def test_hand_focused_crop_uses_visible_hand_extent():
    landmarks = np.zeros((2, 553, 3), dtype=np.float32)
    landmarks[:, :21, :2] = (0.25, 0.50)
    landmarks[:, 21:42, :2] = (0.75, 0.50)
    x0, y0, side = hand_focused_crop(landmarks, np.array([0, 1]), 200, 100,
                                     [0, 0, 200, 100], 0.1)
    assert np.isclose(side, 120)
    assert np.isclose(x0, 40)
    assert np.isclose(y0, -10)
