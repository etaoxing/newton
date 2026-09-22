# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for differentiable multi-step rollouts with SolverFeatherstone.

The system matrices (M, J, P, H, L) used to be stored on the solver instance, so a
multi-step rollout sharing one solver overwrote them every step. The backward pass of
the linear solve (``eval_dense_solve_batched``) reads the Cholesky factor ``L`` for the
step being differentiated, so all but the last step received the final step's factor and
produced wrong gradients. These matrices are now kept per-:class:`~newton.State`; this
test compares analytic gradients against central finite differences across a multi-step
(substep) tape to guard against a regression.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherstoneGrad(unittest.TestCase):
    pass


@wp.kernel
def _sum_joint_qd(joint_qd: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, joint_qd[i])


def _build_pendulum(device, requires_grad):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 100.0
    l0 = builder.add_link()
    builder.add_shape_box(l0, hx=0.3, hy=0.05, hz=0.05)
    l1 = builder.add_link()
    builder.add_shape_box(l1, hx=0.3, hy=0.05, hz=0.05)
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=l0,
        axis=newton.Axis.Z,
        child_xform=wp.transform((-0.3, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    j1 = builder.add_joint_revolute(
        parent=l0,
        child=l1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((-0.3, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    builder.add_articulation([j0, j1])
    # nonzero initial velocity so the configuration (and thus the mass matrix) changes
    # every step, which is what exposed the cross-step aliasing
    builder.joint_q = [0.5, 0.3]
    builder.joint_qd = [5.0, -6.0]
    return builder.finalize(device=device, requires_grad=requires_grad)


def _forward_loss(model, device, jf_list, interval, dt, n):
    """Run an n-step rollout with a fresh solver, return sum(final joint_qd)."""
    dof = model.joint_dof_count
    states = [model.state() for _ in range(n + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0, update_mass_matrix_interval=interval)
    for t in range(n):
        control = model.control()
        control.joint_f.assign(jf_list[t].astype(np.float32))
        solver.step(states[t], states[t + 1], control, None, dt)
    return float(states[n].joint_qd.numpy().sum()), dof


def test_featherstone_multistep_grad(test, device, interval):
    dt = 0.02
    n = 4
    eps = 0.1

    # --- analytic gradients via a single tape over the whole rollout ---
    model = _build_pendulum(device, requires_grad=True)
    dof = model.joint_dof_count
    states = [model.state(requires_grad=True) for _ in range(n + 1)]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0, update_mass_matrix_interval=interval)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    jf_leaves = []
    controls = []
    for _ in range(n):
        control = model.control()
        jf = wp.zeros(dof, dtype=float, requires_grad=True, device=device)
        control.joint_f = jf  # differentiable leaf
        controls.append(control)
        jf_leaves.append(jf)

    tape = wp.Tape()
    with tape:
        for t in range(n):
            solver.step(states[t], states[t + 1], controls[t], None, dt)
        wp.launch(_sum_joint_qd, dim=dof, inputs=[states[n].joint_qd], outputs=[loss], device=device)
    tape.backward(loss)
    analytic = [jf.grad.numpy().copy() for jf in jf_leaves]

    # --- central finite differences (fresh non-grad model reused across evals) ---
    fd_model = _build_pendulum(device, requires_grad=False)
    jf0 = [np.zeros(dof, dtype=np.float32) for _ in range(n)]
    for t in range(n):
        for k in range(dof):
            plus = [a.copy() for a in jf0]
            minus = [a.copy() for a in jf0]
            plus[t][k] += eps
            minus[t][k] -= eps
            lp, _ = _forward_loss(fd_model, device, plus, interval, dt, n)
            lm, _ = _forward_loss(fd_model, device, minus, interval, dt, n)
            fd_tk = (lp - lm) / (2.0 * eps)
            test.assertAlmostEqual(
                float(analytic[t][k]),
                fd_tk,
                delta=2e-2 + 5e-2 * abs(fd_tk),
                msg=f"grad mismatch at step {t}, dof {k} (interval={interval})",
            )


devices = get_test_devices()
for device in devices:
    for interval in (1, 2):
        add_function_test(
            TestFeatherstoneGrad,
            f"test_featherstone_multistep_grad_interval{interval}_{device}",
            test_featherstone_multistep_grad,
            devices=[device],
            interval=interval,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
