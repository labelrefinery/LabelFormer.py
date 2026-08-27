import numpy as np
import pytest

from labelformer import geometry as G


def test_wrap_angle():
    assert G.wrap_angle(np.pi) == pytest.approx(np.pi)
    assert G.wrap_angle(-np.pi) == pytest.approx(np.pi)
    assert G.wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    a = np.random.default_rng(0).uniform(-10, 10, 100)
    w = G.wrap_angle(a)
    assert np.all((w > -np.pi) & (w <= np.pi))
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)


def test_se2_roundtrip():
    rng = np.random.default_rng(1)
    T = G.se2_from_xyt(rng.normal(size=5), rng.normal(size=5), rng.uniform(-np.pi, np.pi, 5))
    eye = G.se2_inv(T) @ T
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), (5, 3, 3)), atol=1e-12)


def test_transform_points_and_boxes():
    T = G.se2_from_xyt(1.0, 2.0, np.pi / 2)
    pt = G.transform_points_2d(T, np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(pt, [[1.0, 3.0]], atol=1e-12)

    box = np.array([1.0, 0.0, 0.0, 4.0, 2.0])
    out = G.transform_boxes_bev(T, box)
    np.testing.assert_allclose(out, [1.0, 3.0, np.pi / 2, 4.0, 2.0], atol=1e-12)


def test_box_corners():
    c = G.box_corners_bev(np.array([0.0, 0.0, 0.0, 4.0, 2.0]))
    expected = {(2.0, 1.0), (-2.0, 1.0), (-2.0, -1.0), (2.0, -1.0)}
    got = {tuple(np.round(p, 9)) for p in c}
    assert got == expected
    # rotation by 90deg swaps extents
    c90 = G.box_corners_bev(np.array([0.0, 0.0, np.pi / 2, 4.0, 2.0]))
    assert np.abs(c90[:, 0]).max() == pytest.approx(1.0)
    assert np.abs(c90[:, 1]).max() == pytest.approx(2.0)


def test_points_in_box_mask():
    box = np.array([10.0, 5.0, np.pi / 4, 4.0, 2.0])
    rng = np.random.default_rng(2)
    # generate points in box frame, map to world, verify mask
    local = rng.uniform([-2, -1], [2, 1], size=(200, 2)) * 0.999
    T = G.se2_from_xyt(box[0], box[1], box[2])
    world = G.transform_points_2d(T, local)
    assert G.points_in_box_mask(world, box).all()
    far = world + 10.0
    assert not G.points_in_box_mask(far, box).any()
    # scale enlarges the region
    edge = G.transform_points_2d(T, np.array([[2.2, 0.0]]))
    assert not G.points_in_box_mask(edge, box)[0]
    assert G.points_in_box_mask(edge, box, scale=1.2)[0]


def test_canonicalize_headings():
    base = 0.3
    h = np.array([base, base + np.pi, base + 0.05, base - 0.02, base + np.pi + 0.01])
    canon, flipped = G.canonicalize_headings(h)
    np.testing.assert_allclose(canon, [base, base, base + 0.05, base - 0.02, base + 0.01], atol=1e-9)
    assert flipped.tolist() == [False, True, False, False, True]

    # majority flipped by pi -> canonical follows the majority
    h2 = np.array([base + np.pi, base + np.pi, base])
    canon2, flipped2 = G.canonicalize_headings(h2)
    np.testing.assert_allclose(canon2[:2], base + np.pi - 2 * np.pi * (base + np.pi > np.pi), atol=1e-9)
    assert flipped2.tolist() == [False, False, True]
