# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-gradient regression tests for SolverFeatherstone: the VELOCITY-FEEDBACK path.

Regression guard for a silent contact-backward severing: warp 1.14 drops reverse-mode adjoints
through struct FIELD ASSIGNMENTS (``s.a = x`` propagates no gradient), so building the shared
``ContactKinematics`` struct field-by-field in ``eval_contact_kinematics`` (semi_implicit
kernels_contact.py) zeroed d(contact force)/d(body_q, body_qd) entirely. Forward values were
untouched; every no-contact gradient stayed exact -- but the velocity->velocity feedback of every
contact vanished from BPTT, so ``d(qd_next)/d(qd_prev)`` through a frictional contact read exactly
IDENTITY. The fix builds the struct with a single constructor call (which warp differentiates
correctly). These tests pin the closed-form velocity-feedback derivatives that the pre-fix code got
wrong (they read 1.0), plus an FD cross-check on a small contact-rich articulated biped.

The existing tests in test_featherstone_implicit_friction.py differentiate w.r.t. an applied FORCE
(a path through the solve that never needs the contact kernel's backward), which is why they passed
while the contact backward was dead. Do not weaken these into force-only checks.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

RADIUS = 0.1
DENSITY = 1000.0
DT = 0.02
KF = 1000.0
KD = 2000.0
V0 = 1.0
SPHERE_MASS = DENSITY * (4.0 / 3.0) * np.pi * RADIUS**3


@wp.kernel
def _sum_joint_qd(joint_qd: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, joint_qd[i])


def _build_slider(device, requires_grad, axis, v0, mu=1.0, ke=1.0e7, kf=KF, kd=0.0):
    """1-DOF prismatic sphere centered on the ground plane (penetration = RADIUS)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.density = DENSITY
    body = builder.add_link()
    builder.add_shape_sphere(body, radius=RADIUS)
    j = builder.add_joint_prismatic(
        parent=-1, child=body, axis=axis, armature=0.0, damping=0.0, target_ke=0.0, target_kd=0.0
    )
    builder.add_articulation([j])
    builder.joint_q = [0.0]
    builder.joint_qd = [v0]
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=requires_grad)
    for attr, val in (
        ("shape_material_ke", ke),
        ("shape_material_kf", kf),
        ("shape_material_mu", mu),
        ("shape_material_kd", kd),
    ):
        a = getattr(model, attr).numpy()
        a[:] = val
        getattr(model, attr).assign(wp.array(a, dtype=wp.float32, device=device))
    return model


def _dqd_dqd0(model, device, integrator):
    """One step: d(qd_1)/d(qd_0) differentiated through solver.step (contact velocity feedback)."""
    state0 = model.state(requires_grad=True)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    state1 = model.state(requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=1, integrator=integrator
    )
    control = model.control()
    contacts = model.collide(state0)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        solver.step(state0, state1, control, contacts, DT)
        wp.launch(_sum_joint_qd, dim=1, inputs=[state1.joint_qd], outputs=[loss], device=device)
    tape.backward(loss)
    return float(state0.joint_qd.grad.numpy()[0])


def test_friction_velocity_feedback_grad(test, device):
    """d(v_1)/d(v_0) through the viscous friction contact must be the closed-form map, NOT identity:
    euler 1 - dt*kf/m (~-3.775), implicitfast m/(m+dt*kf) (~0.173). The severed contact backward
    read exactly +1.0 for both."""
    m = SPHERE_MASS
    for integrator, expected in (("euler", 1.0 - DT * KF / m), ("implicitfast", m / (m + DT * KF))):
        model = _build_slider(device, True, newton.Axis.X, V0)
        g = _dqd_dqd0(model, device, integrator)
        test.assertTrue(np.isfinite(g))
        # the regression signature: exactly 1.0 (no contact feedback at all)
        test.assertNotAlmostEqual(g, 1.0, delta=1e-3, msg=f"{integrator}: contact velocity feedback missing")
        test.assertAlmostEqual(g, expected, delta=1e-3 + 1e-3 * abs(expected), msg=f"{integrator}: grad != closed form")


def test_normal_damping_velocity_feedback_grad(test, device):
    """d(v_1)/d(v_0) through the depth-scaled normal dashpot (euler): 1 - dt*kd*depth/m with
    depth = RADIUS. Pre-fix this also read exactly +1.0."""
    m = SPHERE_MASS
    expected = 1.0 - DT * (KD * RADIUS) / m  # ~0.045 for kd=2000, depth=0.1
    model = _build_slider(device, True, newton.Axis.Z, -V0, mu=0.0, ke=0.0, kf=0.0, kd=KD)
    g = _dqd_dqd0(model, device, "euler")
    test.assertTrue(np.isfinite(g))
    test.assertNotAlmostEqual(g, 1.0, delta=1e-3, msg="normal-damping velocity feedback missing")
    test.assertAlmostEqual(g, expected, delta=1e-3 + 1e-3 * abs(expected))


def _build_biped(device, requires_grad):
    """Free-root torso + 2x(hip+knee+ankle revolute) legs + capsule feet standing on the ground.
    Small G1 stand-in: PD position drives, joint limits, G1-like contact materials."""
    import math

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.density = 1000.0
    leg_l = shin_l = 0.25
    foot_r = 0.03
    root_z = leg_l + shin_l + foot_r - 2.0e-4
    torso = builder.add_link()
    builder.add_shape_box(torso, hx=0.15, hy=0.10, hz=0.05)
    joints = [builder.add_joint_free(child=torso, parent_xform=wp.transform((0.0, 0.0, root_z), wp.quat_identity()))]
    rot_z_to_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)
    for y in (0.12, -0.12):
        thigh = builder.add_link()
        builder.add_shape_capsule(
            thigh, radius=0.04, half_height=leg_l / 2 - 0.04,
            xform=wp.transform((0.0, 0.0, -leg_l / 2), wp.quat_identity()),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=torso, child=thigh, axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, y, 0.0), wp.quat_identity()),
                armature=0.01, limit_lower=-1.5, limit_upper=1.5, limit_ke=2500.0, limit_kd=100.0,
                target_ke=200.0, target_kd=10.0,
            )
        )
        shin = builder.add_link()
        builder.add_shape_capsule(
            shin, radius=0.035, half_height=shin_l / 2 - 0.035,
            xform=wp.transform((0.0, 0.0, -shin_l / 2), wp.quat_identity()),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=thigh, child=shin, axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, 0.0, -leg_l), wp.quat_identity()),
                armature=0.01, limit_lower=-1.5, limit_upper=1.5, limit_ke=2500.0, limit_kd=100.0,
                target_ke=200.0, target_kd=10.0,
            )
        )
        foot = builder.add_link()
        builder.add_shape_capsule(
            foot, radius=foot_r, half_height=0.06, xform=wp.transform((0.02, 0.0, 0.0), rot_z_to_x)
        )
        joints.append(
            builder.add_joint_revolute(
                parent=shin, child=foot, axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, 0.0, -shin_l), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, foot_r), wp.quat_identity()),
                armature=0.01, limit_lower=-1.0, limit_upper=1.0, limit_ke=2500.0, limit_kd=100.0,
                target_ke=50.0, target_kd=2.0,
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=requires_grad)
    for attr, val in (
        ("shape_material_ke", 5.0e4),
        ("shape_material_kf", 1000.0),
        ("shape_material_kd", 2000.0),
        ("shape_material_mu", 0.7),
    ):
        a = getattr(model, attr).numpy()
        a[:] = val
        getattr(model, attr).assign(wp.array(a, dtype=wp.float32, device=device))
    return model


# G1-like regime: substep dt = 1/1920 s. At larger substep dt the explicit viscous friction on the
# light feet is unstable (dt*kf/m_foot > 2) -> chaotic forward -> FD is meaningless noise.
_BIPED_FRAMES = 2
_BIPED_SUBSTEPS = 32
_BIPED_DT = 1.0 / 60.0 / _BIPED_SUBSTEPS


def _biped_rollout_loss(model, device, qd0, requires_grad):
    n = _BIPED_FRAMES * _BIPED_SUBSTEPS
    states = [model.state(requires_grad=requires_grad) for _ in range(n + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    states[0].joint_qd.assign(wp.array(qd0, dtype=wp.float32, device=device, requires_grad=requires_grad))
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=1, integrator="euler"
    )
    control = model.control()
    loss = wp.zeros(1, dtype=float, requires_grad=requires_grad, device=device)
    tape = wp.Tape() if requires_grad else None

    def _run():
        contacts = None
        for k in range(n):
            if k % _BIPED_SUBSTEPS == 0:
                contacts = model.collide(states[k])
            solver.step(states[k], states[k + 1], control, contacts, _BIPED_DT)
        wp.launch(_sum_joint_qd, dim=model.joint_dof_count, inputs=[states[n].joint_qd], outputs=[loss],
                  device=device)

    if requires_grad:
        with tape:
            _run()
    else:
        _run()
    return loss, states, tape


def test_biped_contact_bptt_grad_vs_fd(test, device):
    """Multi-step grounded BPTT on the contact-rich biped (euler: C0 forward, FD-trustworthy):
    analytic d(sum qd_final)/d(qd_0) must correlate with central FD. The severed contact backward
    gave near-zero correlation on the contact-dominated dims."""
    model = _build_biped(device, True)
    dof = model.joint_dof_count
    qd0 = np.zeros(dof, dtype=np.float32)

    loss, states, tape = _biped_rollout_loss(model, device, qd0, requires_grad=True)
    tape.backward(loss)
    # states[0].joint_qd is the leaf: .assign() copied qd0 into the SAME array, so its .grad holds
    # the input adjoint.
    analytic = states[0].joint_qd.grad.numpy().copy()

    fd_model = _build_biped(device, False)
    eps = 1e-4
    fd = np.zeros(dof, dtype=np.float64)
    for j in range(dof):
        vals = []
        for sgn in (+1.0, -1.0):
            qd = qd0.copy()
            qd[j] += sgn * eps
            lv, _, _ = _biped_rollout_loss(fd_model, device, qd, requires_grad=False)
            vals.append(float(lv.numpy()[0]))
        fd[j] = (vals[0] - vals[1]) / (2 * eps)

    test.assertTrue(np.all(np.isfinite(analytic)))
    denom = np.linalg.norm(analytic) * np.linalg.norm(fd)
    test.assertGreater(denom, 0.0, "zero gradient -- contact backward severed?")
    cos = float(analytic @ fd / denom)
    test.assertGreater(cos, 0.9, f"analytic-vs-FD cos {cos:.3f} too low: contact BPTT gradient wrong")


class TestFeatherstoneContactGrad(unittest.TestCase):
    pass


devices = get_test_devices()
for device in devices:
    add_function_test(
        TestFeatherstoneContactGrad,
        f"test_friction_velocity_feedback_grad_{device}",
        test_friction_velocity_feedback_grad,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneContactGrad,
        f"test_normal_damping_velocity_feedback_grad_{device}",
        test_normal_damping_velocity_feedback_grad,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneContactGrad,
        f"test_biped_contact_bptt_grad_vs_fd_{device}",
        test_biped_contact_bptt_grad_vs_fd,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
