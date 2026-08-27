import numpy as np
import pytest

from hunter_kinodynamic_rl.navigation.localization.interface import (
    INVALID_POSE, LocalizationBackend, PoseEstimate, is_pose_finite, is_pose_usable,
)
from hunter_kinodynamic_rl.navigation.localization.odom_backend import OdomLocalizationBackend


def test_odom_backend_satisfies_localization_backend_protocol():
    """Code review fix (round 2, item 5): pose_at() is part of the
    LocalizationBackend contract, not a concrete-class-only extra."""
    assert isinstance(OdomLocalizationBackend(), LocalizationBackend)


def _pose(**overrides):
    base = dict(x=1.0, y=2.0, yaw=0.5, stamp_sec=100.0, confidence=1.0, valid=True)
    base.update(overrides)
    return PoseEstimate(**base)


def test_invalid_pose_is_never_usable():
    assert is_pose_usable(_pose(valid=False), now_sec=100.0, timeout_sec=1.0, min_confidence=0.0) is False


def test_low_confidence_pose_is_rejected():
    assert is_pose_usable(_pose(confidence=0.2), now_sec=100.0, timeout_sec=1.0, min_confidence=0.5) is False


def test_fresh_high_confidence_pose_is_usable():
    assert is_pose_usable(_pose(), now_sec=100.2, timeout_sec=1.0, min_confidence=0.5) is True


def test_stale_pose_is_rejected():
    assert is_pose_usable(_pose(stamp_sec=90.0), now_sec=100.0, timeout_sec=1.0, min_confidence=0.5) is False


def test_pose_from_the_future_is_rejected():
    assert is_pose_usable(_pose(stamp_sec=110.0), now_sec=100.0, timeout_sec=1.0, min_confidence=0.5) is False


def test_pose_slightly_after_now_sec_within_timeout_is_usable():
    """The age check is symmetric (abs(age), not just age > timeout) --
    routine when now_sec is a SCAN timestamp being synchronized against an
    interpolated/nearest odometry sample rather than genuine wall/sim
    "now", where the pose can legitimately land a hair after now_sec."""
    assert is_pose_usable(_pose(stamp_sec=100.3), now_sec=100.0, timeout_sec=1.0, min_confidence=0.5) is True


def test_default_invalid_pose_constant_is_unusable():
    assert is_pose_usable(INVALID_POSE, now_sec=0.0, timeout_sec=100.0, min_confidence=0.0) is False


def test_is_pose_finite_true_for_ordinary_pose():
    assert is_pose_finite(_pose()) is True


@pytest.mark.parametrize("field", ["x", "y", "yaw", "stamp_sec"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_is_pose_finite_false_for_non_finite_scalar_fields(field, bad_value):
    assert is_pose_finite(_pose(**{field: bad_value})) is False


def test_is_pose_finite_false_for_non_finite_covariance():
    cov = np.eye(3)
    cov[1, 1] = float("nan")
    assert is_pose_finite(_pose(covariance=cov)) is False


def test_is_pose_usable_rejects_non_finite_pose_even_if_marked_valid():
    """Code review fix (item 3): a NaN pose must never pass as usable
    purely because `valid=True`/confidence look fine."""
    bad = _pose(x=float("nan"))
    assert bad.valid is True  # the dataclass itself doesn't enforce this
    assert is_pose_usable(bad, now_sec=100.0, timeout_sec=1.0, min_confidence=0.0) is False


@pytest.mark.parametrize("bad_confidence", [float("nan"), float("inf"), float("-inf")])
def test_is_pose_finite_false_for_non_finite_confidence(bad_confidence):
    assert is_pose_finite(_pose(confidence=bad_confidence)) is False


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.5])
def test_is_pose_finite_false_for_out_of_range_confidence(bad_confidence):
    assert is_pose_finite(_pose(confidence=bad_confidence)) is False


def test_is_pose_usable_rejects_nan_confidence_even_with_min_confidence_zero():
    """Code review fix (round 2, item 2): `NaN < min_confidence` is False
    in Python, so the old `pose.confidence < min_confidence` check alone
    silently let a NaN confidence through even when min_confidence=0.0
    (the most permissive possible setting) -- is_pose_finite must catch it
    independently of that comparison."""
    bad = _pose(confidence=float("nan"))
    assert is_pose_usable(bad, now_sec=100.0, timeout_sec=1.0, min_confidence=0.0) is False


def test_odom_backend_starts_invalid():
    backend = OdomLocalizationBackend()
    assert backend.latest_pose().valid is False


def test_odom_backend_update_returns_latest_pose():
    backend = OdomLocalizationBackend()
    backend.update(x=1.0, y=2.0, yaw=0.3, stamp_sec=5.0, confidence=0.9)
    pose = backend.latest_pose()
    assert pose.valid is True
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(2.0)
    assert pose.yaw == pytest.approx(0.3)
    assert pose.confidence == pytest.approx(0.9)


def test_odom_backend_update_default_covariance_shape():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    assert backend.latest_pose().covariance.shape == (3, 3)


def test_odom_backend_invalidate_clears_pose():
    backend = OdomLocalizationBackend()
    backend.update(x=1.0, y=1.0, yaw=0.0, stamp_sec=1.0)
    assert backend.latest_pose().valid is True
    backend.invalidate()
    assert backend.latest_pose().valid is False


@pytest.mark.parametrize("field", ["x", "y", "yaw", "stamp_sec"])
def test_odom_backend_update_forces_invalid_on_non_finite_field(field):
    """Code review fix (item 3): even if the caller passes valid=True, a
    NaN/Inf reading must come out invalid -- never trusted downstream just
    because the caller claimed it was fine."""
    kwargs = dict(x=1.0, y=2.0, yaw=0.3, stamp_sec=5.0, valid=True)
    kwargs[field] = float("nan")
    pose = OdomLocalizationBackend().update(**kwargs)
    assert pose.valid is False


def test_odom_backend_update_forces_invalid_on_non_finite_covariance():
    cov = np.zeros((3, 3))
    cov[0, 0] = float("inf")
    pose = OdomLocalizationBackend().update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0, covariance=cov, valid=True)
    assert pose.valid is False


@pytest.mark.parametrize("bad_confidence", [float("nan"), float("inf"), float("-inf"), -0.1, 1.5])
def test_odom_backend_update_forces_invalid_on_bad_confidence(bad_confidence):
    """Code review fix (round 2, item 2): confidence is now part of the
    same finite/range check every other pose field gets."""
    pose = OdomLocalizationBackend().update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0, confidence=bad_confidence, valid=True)
    assert pose.valid is False


def test_odom_backend_bad_confidence_update_is_excluded_from_pose_at_history():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=1.0)
    backend.update(x=99.0, y=0.0, yaw=0.0, stamp_sec=2.0, confidence=float("nan"))
    backend.update(x=2.0, y=0.0, yaw=0.0, stamp_sec=3.0)
    pose = backend.pose_at(2.0, max_dt_sec=5.0)
    assert pose.valid is True
    assert pose.x == pytest.approx(1.0)  # from the two GOOD samples, not the poisoned x=99.0


def test_odom_backend_non_finite_update_is_excluded_from_pose_at_history():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=1.0)
    backend.update(x=float("nan"), y=0.0, yaw=0.0, stamp_sec=2.0)
    backend.update(x=2.0, y=0.0, yaw=0.0, stamp_sec=3.0)
    # pose_at(2.0, ...) must interpolate/nearest from the two GOOD samples
    # (t=1.0, t=3.0), never the poisoned t=2.0 sample.
    pose = backend.pose_at(2.0, max_dt_sec=5.0)
    assert pose.valid is True
    assert pose.x == pytest.approx(1.0)  # linear interpolation midpoint


def test_pose_at_empty_history_returns_invalid():
    backend = OdomLocalizationBackend()
    assert backend.pose_at(0.0, max_dt_sec=1.0).valid is False


def test_pose_at_interpolates_between_bracketing_samples():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    backend.update(x=10.0, y=20.0, yaw=0.0, stamp_sec=10.0)
    pose = backend.pose_at(2.5, max_dt_sec=100.0)
    assert pose.valid is True
    assert pose.x == pytest.approx(2.5)
    assert pose.y == pytest.approx(5.0)
    assert pose.stamp_sec == pytest.approx(2.5)


def test_pose_at_wraps_yaw_interpolation_across_pi_boundary():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=3.0, stamp_sec=0.0)
    backend.update(x=0.0, y=0.0, yaw=-3.0, stamp_sec=10.0)  # shortest path crosses +-pi
    pose = backend.pose_at(5.0, max_dt_sec=100.0)
    # The shortest-path midpoint between 3.0 and -3.0 (going the SHORT way
    # through +-pi) has magnitude close to pi, not 0.0 (the naive linear
    # midpoint of the raw values).
    assert abs(pose.yaw) > 3.0


def test_pose_at_within_history_span_but_no_exact_bracket_uses_nearest_when_only_one_side_exists():
    backend = OdomLocalizationBackend()
    backend.update(x=5.0, y=0.0, yaw=0.0, stamp_sec=10.0)
    # Query strictly before the only sample -- no "before", only "after".
    pose = backend.pose_at(9.0, max_dt_sec=5.0)
    assert pose.valid is True
    assert pose.x == pytest.approx(5.0)


def test_pose_at_beyond_max_dt_sec_is_rejected():
    backend = OdomLocalizationBackend()
    backend.update(x=5.0, y=0.0, yaw=0.0, stamp_sec=10.0)
    pose = backend.pose_at(100.0, max_dt_sec=1.0)
    assert pose.valid is False


def test_pose_at_within_max_dt_sec_of_single_sample_is_usable():
    backend = OdomLocalizationBackend()
    backend.update(x=5.0, y=0.0, yaw=0.0, stamp_sec=10.0)
    pose = backend.pose_at(10.4, max_dt_sec=1.0)
    assert pose.valid is True


def test_pose_at_uses_message_domain_regardless_of_magnitude():
    """Explicitly proves the fix for the sim-time/wall-time mixing bug:
    pose_at works identically whether timestamps are small Gazebo-sim-time
    values or huge wall-clock epoch values -- it never reads any external
    clock itself."""
    for base in (12.345, 1_788_000_000.0):
        backend = OdomLocalizationBackend()
        backend.update(x=1.0, y=1.0, yaw=0.0, stamp_sec=base)
        backend.update(x=3.0, y=1.0, yaw=0.0, stamp_sec=base + 2.0)
        pose = backend.pose_at(base + 1.0, max_dt_sec=5.0)
        assert pose.valid is True
        assert pose.x == pytest.approx(2.0)


def test_pose_at_does_not_interpolate_across_a_long_outage_gap():
    """Code review fix (round 2, item 1): a bracket existing on BOTH sides
    used to be sufficient to interpolate regardless of how far apart the
    two samples were. odom at t=0 (x=0) and t=100 (x=100) bracketing a
    scan at t=50 with a tight max_dt_sec=0.5 must NOT synthesize x=50 --
    that fabricates a straight-line path across what could be a 100s
    localization outage or rosbag stall."""
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    backend.update(x=100.0, y=0.0, yaw=0.0, stamp_sec=100.0)
    pose = backend.pose_at(50.0, max_dt_sec=0.5)
    assert pose.valid is False


def test_pose_at_falls_back_to_nearest_when_only_one_side_of_bracket_is_close():
    """Both a before- AND an after-sample exist (the query lies between
    them), but only ONE is within max_dt_sec -- must fall back to that
    close one (still gated by max_dt_sec), not interpolate across the wide
    gap to the far one."""
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    backend.update(x=100.0, y=0.0, yaw=0.0, stamp_sec=100.0)
    pose = backend.pose_at(0.3, max_dt_sec=0.5)
    assert pose.valid is True
    assert pose.x == pytest.approx(0.0)  # the close (t=0.0) sample, not an interpolation toward x=100.0


def test_pose_at_interpolation_gap_boundary_is_inclusive():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    backend.update(x=1.0, y=0.0, yaw=0.0, stamp_sec=1.0)
    # Exactly max_dt_sec away on each side -- still a valid interpolation.
    pose = backend.pose_at(0.5, max_dt_sec=0.5)
    assert pose.valid is True
    assert pose.x == pytest.approx(0.5)


def test_invalidate_clears_history_so_pose_at_never_bridges_an_outage():
    backend = OdomLocalizationBackend()
    backend.update(x=0.0, y=0.0, yaw=0.0, stamp_sec=0.0)
    backend.update(x=10.0, y=0.0, yaw=0.0, stamp_sec=10.0)
    backend.invalidate()
    pose = backend.pose_at(5.0, max_dt_sec=100.0)
    assert pose.valid is False


# ------------------------------------------------------------------ gazebo_odom_backend (requires rclpy message types)
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav_msgs")

from nav_msgs.msg import Odometry  # noqa: E402

from hunter_kinodynamic_rl.navigation.localization.gazebo_odom_backend import (  # noqa: E402
    GazeboOdomLocalizationBackend, confidence_from_covariance, yaw_from_quaternion,
)


def test_yaw_from_quaternion_identity_is_zero():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_yaw_from_quaternion_ninety_degrees():
    half = np.pi / 4
    assert yaw_from_quaternion(0.0, 0.0, np.sin(half), np.cos(half)) == pytest.approx(np.pi / 2)


def test_confidence_from_covariance_zero_trace_is_full_confidence():
    assert confidence_from_covariance(0.0) == pytest.approx(1.0)


def test_confidence_from_covariance_decreases_with_trace():
    low = confidence_from_covariance(0.01)
    high = confidence_from_covariance(10.0)
    assert 0.0 < high < low <= 1.0


def test_gazebo_backend_parses_odometry_message():
    msg = Odometry()
    msg.pose.pose.position.x = 3.0
    msg.pose.pose.position.y = -1.5
    half = np.pi / 4
    msg.pose.pose.orientation.z = float(np.sin(half))
    msg.pose.pose.orientation.w = float(np.cos(half))
    msg.pose.covariance = [0.0] * 36

    backend = GazeboOdomLocalizationBackend()
    backend.on_odometry_msg(msg, stamp_sec=42.0)
    pose = backend.latest_pose()
    assert pose.valid is True
    assert pose.x == pytest.approx(3.0)
    assert pose.y == pytest.approx(-1.5)
    assert pose.yaw == pytest.approx(np.pi / 2)
    assert pose.stamp_sec == pytest.approx(42.0)
    assert pose.confidence == pytest.approx(1.0)


def test_gazebo_backend_odom_mode_ignores_covariance_confidence():
    """LocalizationConfig.backend='odom' (use_covariance_confidence=False)
    -- confidence pinned to 1.0 regardless of a large covariance."""
    msg = Odometry()
    cov = [0.0] * 36
    cov[0] = 50.0  # large x variance
    msg.pose.covariance = cov

    backend = GazeboOdomLocalizationBackend(use_covariance_confidence=False)
    backend.on_odometry_msg(msg, stamp_sec=1.0)
    assert backend.latest_pose().confidence == pytest.approx(1.0)


def test_gazebo_backend_gazebo_odom_mode_derives_confidence_from_covariance():
    msg = Odometry()
    cov = [0.0] * 36
    cov[0] = 50.0
    cov[7] = 50.0
    cov[35] = 50.0
    msg.pose.covariance = cov

    backend = GazeboOdomLocalizationBackend(use_covariance_confidence=True)
    backend.on_odometry_msg(msg, stamp_sec=1.0)
    assert backend.latest_pose().confidence < 1.0


def test_gazebo_backend_nan_covariance_produces_invalid_pose():
    msg = Odometry()
    cov = [0.0] * 36
    cov[0] = float("nan")
    msg.pose.covariance = cov

    backend = GazeboOdomLocalizationBackend()
    backend.on_odometry_msg(msg, stamp_sec=1.0)
    # confidence_from_covariance maps a non-finite trace to 0.0 (never a
    # misleadingly high value), AND the pose is forced invalid outright by
    # OdomLocalizationBackend.update's own finite check on the covariance.
    assert backend.latest_pose().valid is False


def test_confidence_from_covariance_non_finite_trace_is_zero():
    assert confidence_from_covariance(float("nan")) == pytest.approx(0.0)
    assert confidence_from_covariance(float("inf")) == pytest.approx(0.0)
