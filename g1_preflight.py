"""Fixed, offline G1 kinematic preflight; not an execution/freshness gate.

The frozen source owns IK, joint limits, phase-aware collision filtering and
swept interpolation. This consumer changes only the numerical IK configuration,
not the registered 5 mm / 3 degree boundary or collision policy. Online callers
must independently enforce the existing 100 ms observation-age limit.
"""

from __future__ import annotations

from g1_source import activate_source


def bounded_ik_config(source=None):
    """Return the frozen solver's bounded, tighter-than-registered settings."""
    activate_source(source)
    from g1_fast_sequential_ik import FastIKConfig

    return FastIKConfig(
        damping=0.03,
        iteration_step_s=0.04,
        maximum_iterations=60,
        max_joint_speed_rad_s=2.0,
        position_tolerance_m=0.004,
        orientation_tolerance_rad=0.04363323129985824,  # 2.5 degrees
    )


def create_preflight(model, source=None):
    """Construct the complete frozen swept gate, without configurable relaxations."""
    activate_source(source)
    from g1_fast_swept_path_preflight import G1FastSweptPathPreflight

    return G1FastSweptPathPreflight(
        model,
        ik_config=bounded_ik_config(source),
        maximum_arm_interpolation_step_rad=0.01,
        maximum_finger_interpolation_step_m=0.001,
    )
