from hunter_kinodynamic_rl.navigation.mapping.raytracing import (
    bresenham_line, clip_ray_to_bounds, trace_clipped,
)


def test_bresenham_line_horizontal():
    cells = bresenham_line(0, 0, 0, 4)
    assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_bresenham_line_vertical():
    cells = bresenham_line(0, 0, 4, 0)
    assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_bresenham_line_diagonal():
    cells = bresenham_line(0, 0, 3, 3)
    assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_bresenham_line_single_cell():
    assert bresenham_line(2, 2, 2, 2) == [(2, 2)]


def test_bresenham_line_endpoints_are_inclusive():
    cells = bresenham_line(1, 1, 5, 9)
    assert cells[0] == (1, 1)
    assert cells[-1] == (5, 9)


def test_bresenham_line_reverse_direction_same_length_and_endpoints():
    forward = bresenham_line(0, 0, 5, 8)
    backward = bresenham_line(5, 8, 0, 0)
    assert len(forward) == len(backward)
    assert backward[0] == forward[-1]
    assert backward[-1] == forward[0]


def test_clip_ray_fully_inside_bounds_is_unchanged():
    clipped = clip_ray_to_bounds(2, 2, 5, 5, height=10, width=10)
    assert clipped == (2, 2, 5, 5)


def test_clip_ray_partially_outside_shortens_to_last_in_bounds_point():
    # Ray from (0,0) toward (0, 20) in a 10-wide grid must clip col to 9.
    clipped = clip_ray_to_bounds(0, 0, 0, 20, height=10, width=10)
    assert clipped is not None
    r0, c0, r1, c1 = clipped
    assert (r0, c0) == (0, 0)
    assert c1 == 9


def test_clip_ray_fully_outside_returns_none():
    clipped = clip_ray_to_bounds(-5, -5, -5, -1, height=10, width=10)
    assert clipped is None


def test_clip_ray_negative_start_in_bounds_end_clips_start():
    clipped = clip_ray_to_bounds(-3, 5, 5, 5, height=10, width=10)
    assert clipped is not None
    r0, c0, r1, c1 = clipped
    assert r0 == 0
    assert (r1, c1) == (5, 5)


def test_trace_clipped_matches_bresenham_when_fully_inside():
    a = trace_clipped(1, 1, 4, 4, height=10, width=10)
    b = bresenham_line(1, 1, 4, 4)
    assert a == b


def test_trace_clipped_out_of_bounds_ray_returns_empty():
    assert trace_clipped(-5, -5, -5, -1, height=10, width=10) == []


def test_trace_clipped_never_yields_out_of_bounds_indices():
    cells = trace_clipped(5, 5, 5, 100, height=10, width=10)
    for r, c in cells:
        assert 0 <= r < 10
        assert 0 <= c < 10
    assert cells[-1] == (5, 9)
