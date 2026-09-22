# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SolverFeatherstone(integrator="implicitfast")`` -- implicit linear-viscous contact
friction folded into the generalized inertia (``H_eff = H + dt * J_t^T kf J_t``).

A single 1-DOF prismatic body (a sphere sliding tangentially on the ground plane) has a closed-form
solution: the only generalized force on the x DOF is the viscous friction ``ft = kf * v`` (the normal
force, contact stiffness, penetration depth and gravity all act along the constrained, non-DOF axis and
never enter the x equation), so for effective DOF mass ``m`` and one step of size ``dt``::

    explicit  ("euler"):        v_next = v * (1 - dt*kf/m)        # can overshoot; unstable for stiff kf
    implicit  ("implicitfast"): v_next = v * m / (m + dt*kf)      # backward Euler; unconditionally stable

These tests check the forward values, the analytic BPTT gradient vs central finite differences, euler
forward parity (the shared-kinematics refactor must not change euler), that the Coulomb-capped branch is
NOT folded into the inertia, that the cached base inertia never accumulates the per-step damping, and
that the integrator argument is validated.
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
V0 = 1.0
SPHERE_MASS = DENSITY * (4.0 / 3.0) * np.pi * RADIUS**3  # ~= 4.1887902

# closed-form one-step maps
EXPLICIT_RATIO = 1.0 - DT * KF / SPHERE_MASS  # ~= -3.7746483
IMPLICIT_RATIO = SPHERE_MASS / (SPHERE_MASS + DT * KF)  # ~= 0.1731712


@wp.kernel
def _sum_joint_qd(joint_qd: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, joint_qd[i])


def _build_slider(device, requires_grad, mu=1.0, ke=1.0e7):
    """1-DOF prismatic body: a sphere centered on the ground plane, free to slide along +x."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.density = DENSITY
    body = builder.add_link()
    builder.add_shape_sphere(body, radius=RADIUS)
    j = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=newton.Axis.X,
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([j])
    builder.joint_q = [0.0]
    builder.joint_qd = [V0]
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=requires_grad)
    # Force uniform contact materials so the (averaged) sphere/ground contact is exactly (ke, kf, mu).
    # Large mu*ke keeps the contact in the linear viscous branch (kf*|vt| < mu*|fn|) for any penetration;
    # small mu*ke (the capped test) drives it into the Coulomb-capped branch.
    for attr, val in (
        ("shape_material_ke", ke),
        ("shape_material_kf", KF),
        ("shape_material_mu", mu),
        ("shape_material_kd", 0.0),
    ):
        a = getattr(model, attr).numpy()
        a[:] = val
        getattr(model, attr).assign(wp.array(a, dtype=wp.float32, device=device))
    return model


NORMAL_KD = 1000.0  # linear normal-dashpot coefficient for the normal-contact tests


def _build_normal_slider(device, requires_grad, kd=NORMAL_KD):
    """1-DOF prismatic-Z sphere penetrating the ground plane. ke=0 isolates the normal dashpot (no
    spring) and mu=0 removes friction, so under integrator="implicitfast" the ONLY generalized force on
    the z DOF is the non-depth-scaled normal damping C*vn (C=kd) -- the analog of the prismatic-X friction
    slider, one axis over. Unlike that slider (normal Jacobian J_n=0 along x), here J_n != 0, so the
    implicit normal damping actually enters H_eff."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.density = DENSITY
    body = builder.add_link()
    builder.add_shape_sphere(body, radius=RADIUS)
    j = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([j])
    builder.joint_q = [0.0]  # sphere center on the z=0 plane -> penetrates by RADIUS
    builder.joint_qd = [-1.0]  # approaching (vn<0) so the one-sided normal dashpot is active when probed
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=requires_grad)
    for attr, val in (
        ("shape_material_ke", 0.0),  # no normal spring -> isolate the dashpot
        ("shape_material_kf", 0.0),
        ("shape_material_mu", 0.0),
        ("shape_material_kd", kd),
    ):
        a = getattr(model, attr).numpy()
        a[:] = val
        getattr(model, attr).assign(wp.array(a, dtype=wp.float32, device=device))
    return model


# --- hard joint-limit fold: a 1-DOF prismatic-X body driven PAST its limit, no contact ---
# A LIGHT body (so dt^2*limit_ke/m > 2) makes the stiff explicit limit spring an unstable/expansive step
# under euler, which implicitfast's backward-Euler fold turns into a stable contraction -- the joint-space
# analogue of the friction slider. m = body mass (prismatic DOF inertia = mass).
LIM_KE = 2500.0  # hard joint-limit stiffness (G1's MJCF-derived value)
LIM_KD = 100.0  # hard joint-limit damping
LIM_UPPER = 0.1  # symmetric +/- hard limit
Q0_LIMIT = 0.15  # initial position: 0.05 past the upper limit -> limit force active
EFFORT_BIG = 1.0e6  # large effort limit -> the limit force is UNSATURATED -> the fold applies
EFFORT_SMALL = 50.0  # < |limit force| (~225) -> the drive saturates -> the fold must NOT apply
LIM_RADIUS = 0.04  # light sphere: m ~= 0.268, so dt^2*ke/m ~= 3.7 > 2 (euler overshoots, implicit bounded)
LIM_MASS = DENSITY * (4.0 / 3.0) * np.pi * LIM_RADIUS**3  # ~= 0.26808

_LIM_DELTA = LIM_UPPER - Q0_LIMIT  # = -0.05 (restoring direction, q past the upper limit)
# euler integrates the limit force explicitly; implicitfast folds the velocity-linear part (dt^2*ke + dt*kd)
EULER_LIMIT_VNEXT = V0 + DT * (LIM_KE * _LIM_DELTA - LIM_KD * V0) / LIM_MASS  # ~= -15.786 (overshoot)
_LIM_DENOM = LIM_MASS + DT * DT * LIM_KE + DT * LIM_KD
IMPLICIT_LIMIT_VNEXT = (LIM_MASS * V0 + DT * LIM_KE * _LIM_DELTA) / _LIM_DENOM  # ~= -0.683 (bounded)


def _build_limit_slider(device, requires_grad, effort=EFFORT_BIG):
    """1-DOF prismatic-X body sitting past its UPPER hard joint limit, NO contact (no ground, gravity=0):
    the only generalized force on the x DOF is the hard-limit penalty ``limit_ke*(upper-q) - limit_kd*qd``
    (kernels_body.joint_force). With m = body mass the one-step maps are::

        euler:        v_next = v0 + dt*(limit_ke*(upper-q0) - limit_kd*v0)/m         # explicit, expansive
        implicitfast: v_next = (m*v0 + dt*limit_ke*(upper-q0)) / (m + dt^2*limit_ke + dt*limit_kd)  # stable

    a light body makes the euler step overshoot (|v_next| > |v0|) while the implicit fold stays bounded.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.density = DENSITY
    body = builder.add_link()
    builder.add_shape_sphere(body, radius=LIM_RADIUS)
    j = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=newton.Axis.X,
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
        limit_lower=-LIM_UPPER,
        limit_upper=LIM_UPPER,
        limit_ke=LIM_KE,
        limit_kd=LIM_KD,
        effort_limit=effort,
    )
    builder.add_articulation([j])
    builder.joint_q = [Q0_LIMIT]  # past the upper limit -> joint_force applies the stiff limit spring
    builder.joint_qd = [V0]
    # NO ground plane: isolate the joint-limit force (the fold is contact-independent).
    model = builder.finalize(device=device, requires_grad=requires_grad)
    return model


def _rollout(model, device, integrator, n_steps, interval, v0=V0):
    """Run ``n_steps`` with a fresh solver, reusing one contact set (the slider's contact geometry is
    static: prismatic-x keeps z fixed). Returns the final x velocity (float)."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    if v0 != V0:
        state.joint_qd.assign(np.array([v0], dtype=np.float32))
    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=interval, integrator=integrator
    )
    control = model.control()
    contacts = model.collide(state)
    for _ in range(n_steps):
        state_next = model.state()
        solver.step(state, state_next, control, contacts, DT)
        state = state_next
    return float(state.joint_qd.numpy()[0])


def test_implicit_friction_1dof_forward(test, device):
    """Forward parity: euler reproduces the explicit map; implicitfast the backward-Euler map."""
    model = _build_slider(device, requires_grad=False)

    v_euler = _rollout(model, device, "euler", n_steps=1, interval=1)
    v_implicit = _rollout(model, device, "implicitfast", n_steps=1, interval=1)

    # euler matches the explicit closed form (this is also the byte-parity check that the shared
    # eval_contact_kinematics refactor did not perturb the explicit friction force).
    test.assertAlmostEqual(v_euler, V0 * EXPLICIT_RATIO, delta=1e-4 + 1e-4 * abs(EXPLICIT_RATIO))
    # implicitfast matches the unconditionally-stable backward-Euler closed form.
    test.assertAlmostEqual(v_implicit, V0 * IMPLICIT_RATIO, delta=1e-4 + 1e-4 * abs(IMPLICIT_RATIO))

    # sanity vs the hand-computed reference numbers
    test.assertAlmostEqual(v_euler, -3.7746477, delta=2e-3)
    test.assertAlmostEqual(v_implicit, 0.173170745, delta=2e-4)
    # the whole point: implicit does not overshoot (|v| stays bounded), explicit does.
    test.assertLess(abs(v_implicit), V0)
    test.assertGreater(abs(v_euler), V0)


def _grad_qd_wrt_force(device, integrator, build_fn=_build_slider):
    """One step: d(qd_next)/d(applied joint force), differentiated through the solve. Returns
    (analytic, fd). The applied force enters only through qdd = solve(H_eff, tau + f), so this isolates
    the solve adjoint with A = H_eff: analytic should be dt/(m + dt*C) for implicitfast (C = the folded
    linear coefficient: kf for the tangential slider, kd for the normal slider), dt/m for euler.
    (NOTE: differentiating the *initial velocity* through solver.step used to read IDENTITY for euler
    and implicitfast alike -- that was NOT a plumbing limitation but a real severed contact backward:
    warp 1.14 drops adjoints through struct field assignments, and eval_contact_kinematics built its
    result struct field-by-field. Fixed by constructor-building the struct; the velocity-feedback
    closed forms are pinned in test_featherstone_contact_grad.py. This helper still differentiates the
    force, which isolates the solve adjoint with A = H_eff.)"""
    model = build_fn(device, requires_grad=True)
    state0 = model.state(requires_grad=True)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    state1 = model.state(requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=1, integrator=integrator
    )
    control = model.control()
    jf = wp.zeros(model.joint_dof_count, dtype=float, requires_grad=True, device=device)
    control.joint_f = jf
    contacts = model.collide(state0)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        solver.step(state0, state1, control, contacts, DT)
        wp.launch(_sum_joint_qd, dim=1, inputs=[state1.joint_qd], outputs=[loss], device=device)
    tape.backward(loss)
    analytic = float(jf.grad.numpy()[0])

    # central FD of the forward
    fd_model = build_fn(device, requires_grad=False)

    def fwd(f):
        s0 = fd_model.state()
        newton.eval_fk(fd_model, fd_model.joint_q, fd_model.joint_qd, s0)
        c = fd_model.control()
        jfv = np.zeros(fd_model.joint_dof_count, dtype=np.float32)
        jfv[0] = f
        c.joint_f.assign(jfv)
        s1 = fd_model.state()
        sv = newton.solvers.SolverFeatherstone(
            fd_model, angular_damping=0.0, update_mass_matrix_interval=1, integrator=integrator
        )
        sv.step(s0, s1, c, fd_model.collide(s0), DT)
        return float(s1.joint_qd.numpy()[0])

    fd = (fwd(1.0) - fwd(-1.0)) / 2.0
    return analytic, fd


def test_implicit_friction_grad(test, device):
    """The implicit augmentation H_eff = H + dt*D is differentiated correctly: the force->velocity
    response folds dt*kf into the inertia (analytic == FD == dt/(m+dt*kf)), unlike euler (dt/m)."""
    m = SPHERE_MASS
    expected_impl = DT / (m + DT * KF)  # ~= 8.27e-4
    expected_euler = DT / m  # ~= 4.77e-3

    g_impl, fd_impl = _grad_qd_wrt_force(device, "implicitfast")
    g_euler, fd_euler = _grad_qd_wrt_force(device, "euler")

    test.assertTrue(np.isfinite(g_impl) and np.isfinite(g_euler))
    # analytic matches the closed form AND finite differences
    test.assertAlmostEqual(g_impl, expected_impl, delta=1e-6 + 1e-3 * expected_impl)
    test.assertAlmostEqual(g_impl, fd_impl, delta=1e-6 + 1e-2 * abs(fd_impl), msg="implicit grad != FD")
    test.assertAlmostEqual(g_euler, expected_euler, delta=1e-6 + 1e-3 * expected_euler)
    test.assertAlmostEqual(g_euler, fd_euler, delta=1e-6 + 1e-2 * abs(fd_euler), msg="euler grad != FD")
    # the implicit stiffening shrinks the force response by exactly m/(m+dt*kf)
    test.assertAlmostEqual(g_impl / g_euler, IMPLICIT_RATIO, delta=1e-3)


def test_coulomb_branch_not_augmented(test, device):
    """In the Coulomb-capped (sliding) regime the friction force is bounded/nonlinear, so it must NOT be
    folded into the inertia: implicitfast and euler must produce the identical step."""
    # small mu*ke => kf*|vt| (=1000) > mu*|fn| => capped
    model = _build_slider(device, requires_grad=False, mu=0.01, ke=1.0e4)

    v_euler = _rollout(model, device, "euler", n_steps=1, interval=1)
    v_implicit = _rollout(model, device, "implicitfast", n_steps=1, interval=1)

    # capped => D == 0 => H_eff == H_base => identical solve
    test.assertAlmostEqual(v_euler, v_implicit, delta=1e-8)
    # and it really is capped (the force is far weaker than the viscous kf*v=1000 would give, so v stays
    # close to V0 -- it did not overshoot the way the viscous explicit case did)
    test.assertGreater(v_implicit, 0.0)


def test_cached_base_no_accumulate(test, device):
    """The cached base inertia must not accumulate dt*D across steps (H_eff is rebuilt from H_base every
    substep). If it accumulated, the per-step contraction ratio would drift; here it stays constant."""
    model = _build_slider(device, requires_grad=False)
    interval = 32  # cache the base inertia; only step 0 rebuilds it

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=interval, integrator="implicitfast"
    )
    control = model.control()
    contacts = model.collide(state)

    prev = float(state.joint_qd.numpy()[0])
    ratios = []
    for _ in range(4):
        state_next = model.state()
        solver.step(state, state_next, control, contacts, DT)
        state = state_next
        cur = float(state.joint_qd.numpy()[0])
        ratios.append(cur / prev)
        prev = cur

    ratios = np.array(ratios)
    test.assertTrue(np.all(np.isfinite(ratios)))
    # every step contracts by m/(m+dt*kf); a drifting ratio would mean the base accumulated the damping
    np.testing.assert_allclose(ratios, IMPLICIT_RATIO, rtol=2e-3, atol=2e-3)


def test_integrator_validation(test, device):
    """The integrator argument is validated (ValueError, not a stripped assert)."""
    model = _build_slider(device, requires_grad=False)
    with test.assertRaises(ValueError):
        newton.solvers.SolverFeatherstone(model, integrator="backward")
    # the two supported modes construct fine
    newton.solvers.SolverFeatherstone(model, integrator="euler")
    newton.solvers.SolverFeatherstone(model, integrator="implicitfast")


def test_implicit_normal_damping_forward(test, device):
    """implicitfast folds the non-depth-scaled normal contact dashpot kd into the inertia: a penetrating
    prismatic-Z sphere falling at v0 has its approach velocity damped by backward Euler,
    v_next = m/(m+dt*kd)*v0 (the friction map, one axis over). This is the impact-absorbing behavior that
    replaces the explicit penalty's elastic rebound."""
    v0 = -1.0
    expected = v0 * SPHERE_MASS / (SPHERE_MASS + DT * NORMAL_KD)  # ~ -0.17317
    model = _build_normal_slider(device, requires_grad=False)
    v = _rollout(model, device, "implicitfast", n_steps=1, interval=1, v0=v0)
    test.assertTrue(np.isfinite(v))
    test.assertAlmostEqual(v, expected, delta=1e-4 + 1e-3 * abs(expected))
    # the dashpot actually absorbed the approach (|v_next| well below |v0|, not a near-elastic pass-through)
    test.assertLess(abs(v), 0.5 * abs(v0))


def test_implicit_normal_damping_grad(test, device):
    """The implicit normal damping enters H_eff and is differentiated correctly: the force->velocity
    response on the penetrating z DOF folds kd (analytic == FD == dt/(m+dt*kd)), strictly smaller than
    euler's unfolded dt/m -- the normal analog of the friction gradient test."""
    expected_impl = DT / (SPHERE_MASS + DT * NORMAL_KD)
    g_impl, fd_impl = _grad_qd_wrt_force(device, "implicitfast", _build_normal_slider)
    g_euler, _ = _grad_qd_wrt_force(device, "euler", _build_normal_slider)
    test.assertTrue(np.isfinite(g_impl) and np.isfinite(g_euler))
    test.assertAlmostEqual(g_impl, expected_impl, delta=1e-6 + 1e-3 * expected_impl)
    test.assertAlmostEqual(g_impl, fd_impl, delta=1e-6 + 1e-2 * abs(fd_impl), msg="implicit normal grad != FD")
    # folding kd shrinks the force->velocity response relative to euler (which folds nothing)
    test.assertLess(g_impl, g_euler)


def test_implicit_joint_limit_forward(test, device):
    """A DOF past its hard limit: euler integrates the stiff limit spring explicitly and OVERSHOOTS
    (|v_next| > |v0|); implicitfast folds it (backward Euler) and stays bounded. One-step velocities match
    the closed forms; euler must reproduce the pre-fold explicit map (byte-parity)."""
    v_euler = _rollout(_build_limit_slider(device, requires_grad=False), device, "euler", 1, 1)
    v_implicit = _rollout(_build_limit_slider(device, requires_grad=False), device, "implicitfast", 1, 1)
    test.assertAlmostEqual(v_euler, EULER_LIMIT_VNEXT, delta=1e-3 + 1e-3 * abs(EULER_LIMIT_VNEXT))
    test.assertAlmostEqual(v_implicit, IMPLICIT_LIMIT_VNEXT, delta=1e-3 + 1e-3 * abs(IMPLICIT_LIMIT_VNEXT))
    # sanity vs hand-computed refs (m=0.26808, ke=2500, kd=100, delta=-0.05)
    test.assertAlmostEqual(v_euler, -15.7857, delta=1e-2)
    test.assertAlmostEqual(v_implicit, -0.68295, delta=1e-3)
    # the whole point: the explicit limit spring overshoots, the implicit fold stays bounded
    test.assertGreater(abs(v_euler), V0)
    test.assertLess(abs(v_implicit), V0)


def test_implicit_joint_limit_grad(test, device):
    """The folded hard-limit stiffness enters H_eff and is differentiated correctly: the force->velocity
    response folds dt^2*ke + dt*kd (analytic == FD == dt/(m + dt^2*ke + dt*kd)), strictly smaller than
    euler's unfolded dt/m. Same harness as the friction/normal grad tests; the limit is active so H_eff
    carries the fold."""
    expected_impl = DT / _LIM_DENOM
    expected_euler = DT / LIM_MASS
    g_impl, fd_impl = _grad_qd_wrt_force(device, "implicitfast", _build_limit_slider)
    g_euler, fd_euler = _grad_qd_wrt_force(device, "euler", _build_limit_slider)
    test.assertTrue(np.isfinite(g_impl) and np.isfinite(g_euler))
    test.assertAlmostEqual(g_impl, expected_impl, delta=1e-6 + 1e-3 * expected_impl)
    test.assertAlmostEqual(g_impl, fd_impl, delta=1e-6 + 1e-2 * abs(fd_impl), msg="implicit limit grad != FD")
    test.assertAlmostEqual(g_euler, expected_euler, delta=1e-6 + 1e-3 * expected_euler)
    test.assertAlmostEqual(g_euler, fd_euler, delta=1e-6 + 1e-2 * abs(fd_euler), msg="euler limit grad != FD")
    # folding the stiff limit shrinks the force->velocity response (the BPTT-stabilizing contraction)
    test.assertLess(g_impl, g_euler)


def test_joint_limit_saturated_not_folded(test, device):
    """When the limit force is EFFORT-saturated its tangent is zero (the jcalc_tau clamp already bounds it),
    so the fold must NOT apply: implicitfast and euler produce the identical step (the joint-limit analogue
    of the Coulomb-capped-friction-not-folded test)."""
    v_e = _rollout(_build_limit_slider(device, requires_grad=False, effort=EFFORT_SMALL), device, "euler", 1, 1)
    v_i = _rollout(
        _build_limit_slider(device, requires_grad=False, effort=EFFORT_SMALL), device, "implicitfast", 1, 1
    )
    # saturated => no fold => H_eff == H_base => identical solve
    test.assertAlmostEqual(v_e, v_i, delta=1e-7)


devices = get_test_devices()


class TestFeatherstoneImplicitFriction(unittest.TestCase):
    pass


for device in devices:
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_friction_1dof_forward_{device}",
        test_implicit_friction_1dof_forward,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_friction_grad_{device}",
        test_implicit_friction_grad,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_coulomb_branch_not_augmented_{device}",
        test_coulomb_branch_not_augmented,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_cached_base_no_accumulate_{device}",
        test_cached_base_no_accumulate,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_integrator_validation_{device}",
        test_integrator_validation,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_normal_damping_forward_{device}",
        test_implicit_normal_damping_forward,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_normal_damping_grad_{device}",
        test_implicit_normal_damping_grad,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_joint_limit_forward_{device}",
        test_implicit_joint_limit_forward,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_implicit_joint_limit_grad_{device}",
        test_implicit_joint_limit_grad,
        devices=[device],
    )
    add_function_test(
        TestFeatherstoneImplicitFriction,
        f"test_joint_limit_saturated_not_folded_{device}",
        test_joint_limit_saturated_not_folded,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
