# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ...math import transform_twist, velocity_at_point
from ...sim import BodyFlags, JointType, Model, State
from ...sim.articulation import (
    com_twist_to_point_velocity,
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    origin_twist_to_com_twist,
    transform_2d_rotational_axes,
    transform_3d_rotational_axes,
)
from ..semi_implicit.kernels_body import joint_force
from ..semi_implicit.kernels_contact import FRICTION_CAP_TOTAL, eval_contact_kinematics


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array[wp.vec3],
    # outputs
    body_X_com: wp.array[wp.transform],
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.kernel
def zero_kinematic_body_forces(
    body_flags: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
):
    """Zero accumulated spatial forces for kinematic bodies."""
    tid = wp.tid()
    if (body_flags[tid] & BodyFlags.KINEMATIC) == 0:
        return
    body_f[tid] = wp.spatial_vector()


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t: The rigid-body transform (destination ← source).
        I: The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array[float],
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array[float],
    q_start: int,
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        # Use the FK-transported axes (transform_*_rotational_axes), not the raw joint
        # axes, so velocity and motion subspace stay consistent with FK for multi-angular D6 joints.
        iqd = qd_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[iqd]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[iqd]
            joint_S_s[iqd] = S_s
        if ang_axis_count == 2:
            a0, a1 = transform_2d_rotational_axes(joint_axis[iqd + 0], joint_axis[iqd + 1], joint_q[iq + 0])
            S_0 = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), a0))
            S_1 = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), a1))
            v_j_s += S_0 * joint_qd[iqd + 0] + S_1 * joint_qd[iqd + 1]
            joint_S_s[iqd + 0] = S_0
            joint_S_s[iqd + 1] = S_1
        if ang_axis_count == 3:
            a0, a1, a2 = transform_3d_rotational_axes(
                joint_axis[iqd + 0],
                joint_axis[iqd + 1],
                joint_axis[iqd + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
            )
            S_0 = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), a0))
            S_1 = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), a1))
            S_2 = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), a2))
            v_j_s += S_0 * joint_qd[iqd + 0] + S_1 * joint_qd[iqd + 1] + S_2 * joint_qd[iqd + 2]
            joint_S_s[iqd + 0] = S_0
            joint_S_s[iqd + 1] = S_1
            joint_S_s[iqd + 2] = S_2

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_damping: wp.array[float],
    joint_effort_limit: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    coord_start: int,
    dof_start: int,
    target_q_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array[float],
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]

            q = joint_q[coord_start + i]
            qd = joint_qd[j]

            lower = joint_limit_lower[j]
            upper = joint_limit_upper[j]
            limit_ke = joint_limit_ke[j]
            limit_kd = joint_limit_kd[j]
            target_ke = joint_target_ke[j]
            target_kd = joint_target_kd[j]
            target_pos = joint_target_q[target_q_start + i]
            target_vel = joint_target_qd[j]
            damping = joint_damping[j]

            drive_f = joint_force(
                q, qd, target_pos, target_vel, target_ke, target_kd, lower, upper, limit_ke, limit_kd, damping
            )
            # Actuator saturation: clamp the PD drive torque to the joint's effort limit. Real actuators
            # saturate, and featherstone otherwise IGNORES joint_effort_limit (a correctness gap). This also
            # bounds the gradient -- a saturated drive has zero gradient w.r.t. the position target, so a
            # violent command (huge PD error, e.g. a randomized-init env driving 240 N*m through a 80 N*m
            # ANYmal actuator) can no longer explode the BPTT-through-contact gradient. Default
            # effort_limit=1e6 -> no-op for EFFORT-control envs (their torque is in joint_f, left unclamped).
            eff = joint_effort_limit[j]
            drive_f = wp.clamp(drive_f, -eff, eff)

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + drive_f + joint_f[j]

            tau[j] = t

        return


@wp.func
def jcalc_integrate(
    parent: int,
    joint_X_c: wp.transform,
    body_com_child: wp.vec3,
    type: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        if parent < 0:
            a_parent = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
            alpha = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

            v_parent = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
            omega = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

            p = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])
            r = wp.quat(
                joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
            )

            r_com_joint = wp.transform_point(wp.transform_inverse(joint_X_c), body_com_child)
            x_com = p + wp.quat_rotate(r, r_com_joint)
            v_com = v_parent + wp.cross(omega, x_com)
            a_com = a_parent + wp.cross(alpha, x_com) + wp.cross(omega, v_com)

            omega_new = omega + alpha * dt
            v_com_new = v_com + a_com * dt

            drdt = wp.quat(omega_new, 0.0) * r * 0.5
            r_new = wp.normalize(r + drdt * dt)
            x_com_new = x_com + v_com_new * dt
            p_new = x_com_new - wp.quat_rotate(r_new, r_com_joint)
            v_parent_new = v_com_new - wp.cross(omega_new, x_com_new)

            joint_q_new[coord_start + 0] = p_new[0]
            joint_q_new[coord_start + 1] = p_new[1]
            joint_q_new[coord_start + 2] = p_new[2]

            joint_q_new[coord_start + 3] = r_new[0]
            joint_q_new[coord_start + 4] = r_new[1]
            joint_q_new[coord_start + 5] = r_new[2]
            joint_q_new[coord_start + 6] = r_new[3]

            joint_qd_new[dof_start + 0] = v_parent_new[0]
            joint_qd_new[dof_start + 1] = v_parent_new[1]
            joint_qd_new[dof_start + 2] = v_parent_new[2]
            joint_qd_new[dof_start + 3] = omega_new[0]
            joint_qd_new[dof_start + 4] = omega_new[1]
            joint_qd_new[dof_start + 5] = omega_new[2]
            return

        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # Descendants stay in Featherstone's internal parent-origin coordinates
        # during the integrator step. The public COM convention is restored at
        # the solver boundary once the end-step parent pose is known.
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        dpdt_s = v_s + wp.cross(w_s, p_s)
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_s[0]
        joint_qd_new[dof_start + 1] = v_s[1]
        joint_qd_new[dof_start + 2] = v_s[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # one thread per joint
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_end[index]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    solve_origin: wp.vec3,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_I_m: wp.array[wp.spatial_matrix],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    # outputs
    body_qd: wp.array[wp.spatial_vector],
    joint_S_s: wp.array[wp.spatial_vector],
    body_solve_origin: wp.array[wp.vec3],
    body_I_s: wp.array[wp.spatial_matrix],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    q_start = joint_q_start[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj
    X_wpj_s = wp.transform(wp.transform_get_translation(X_wpj) - solve_origin, wp.transform_get_rotation(X_wpj))

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        joint_q,
        lin_axis_count,
        ang_axis_count,
        X_wpj_s,
        joint_qd,
        q_start,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    x_com_s = wp.transform_get_translation(X_sm) - solve_origin
    body_solve_origin[child] = solve_origin
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[0, 0]

    world_idx = body_world[child]
    world_g = gravity[wp.max(world_idx, 0)]
    f_g = m * world_g
    f_g_s = wp.spatial_vector(f_g, wp.cross(x_com_s, f_g))

    # body forces
    X_sm_s = wp.transform(x_com_s, wp.transform_get_rotation(X_sm))
    I_s = transform_spatial_inertia(X_sm_s, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)
    omega_world = wp.spatial_bottom(v_s)
    v_com_world = wp.spatial_top(v_s) + wp.cross(omega_world, x_com_s)

    body_qd[child] = wp.spatial_vector(v_com_world, omega_world)
    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Convert body forces from COM-frame to world-origin-frame and negate for use in Featherstone dynamics.
@wp.kernel
def convert_body_force_com_to_origin(
    body_q: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    # outputs
    body_f_ext: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    f_ext_com = body_f_ext[tid]

    # skip if force is zero
    if wp.length(f_ext_com) == 0.0:
        return

    body_q_com_val = body_q[tid] * body_X_com[tid]
    r_com = wp.transform_get_translation(body_q_com_val)

    force = wp.spatial_top(f_ext_com)
    torque_com = wp.spatial_bottom(f_ext_com)

    body_f_ext[tid] = -wp.spatial_vector(force, torque_com + wp.cross(r_com, force))


@wp.kernel
def accumulate_free_distance_joint_f_to_body_force(
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_f_public: wp.array[float],
    body_f_ext: wp.array[wp.spatial_vector],
):
    """Accumulate FREE/DISTANCE public COM wrenches into the body-force buffer."""
    joint_id = wp.tid()
    jtype = joint_type[joint_id]
    if jtype != JointType.FREE and jtype != JointType.DISTANCE:
        return

    qd_start = joint_qd_start[joint_id]
    child = joint_child[joint_id]

    force = wp.vec3(
        joint_f_public[qd_start + 0],
        joint_f_public[qd_start + 1],
        joint_f_public[qd_start + 2],
    )
    torque_com = wp.vec3(
        joint_f_public[qd_start + 3],
        joint_f_public[qd_start + 4],
        joint_f_public[qd_start + 5],
    )

    wp.atomic_add(body_f_ext, child, wp.spatial_vector(force, torque_com))


@wp.kernel
def convert_free_distance_joint_qd_public_to_internal(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_qd_public: wp.array[float],
    joint_qd_internal: wp.array[float],
):
    joint_id = wp.tid()
    qd_start = joint_qd_start[joint_id]
    qd_end = joint_qd_start[joint_id + 1]
    jtype = joint_type[joint_id]

    if jtype != JointType.FREE and jtype != JointType.DISTANCE:
        for i in range(qd_start, qd_end):
            joint_qd_internal[i] = joint_qd_public[i]
        return

    parent = joint_parent[joint_id]
    child = joint_child[joint_id]

    X_wpj = joint_X_p[joint_id]
    if parent >= 0:
        X_wpj = body_q[parent] * X_wpj

    q_p = wp.transform_get_rotation(X_wpj)
    x_anchor_world = wp.transform_get_translation(X_wpj)
    x_child_com_world = wp.transform_point(body_q[child], body_com[child])
    r_child_com_parent = wp.quat_rotate_inv(q_p, x_child_com_world - x_anchor_world)

    v_com_parent = wp.vec3(
        joint_qd_public[qd_start + 0],
        joint_qd_public[qd_start + 1],
        joint_qd_public[qd_start + 2],
    )
    omega_parent = wp.vec3(
        joint_qd_public[qd_start + 3],
        joint_qd_public[qd_start + 4],
        joint_qd_public[qd_start + 5],
    )
    v_internal_parent = v_com_parent - wp.cross(omega_parent, r_child_com_parent)

    joint_qd_internal[qd_start + 0] = v_internal_parent[0]
    joint_qd_internal[qd_start + 1] = v_internal_parent[1]
    joint_qd_internal[qd_start + 2] = v_internal_parent[2]
    joint_qd_internal[qd_start + 3] = omega_parent[0]
    joint_qd_internal[qd_start + 4] = omega_parent[1]
    joint_qd_internal[qd_start + 5] = omega_parent[2]


@wp.kernel
def reconstruct_free_distance_joint_q_from_body_pose(
    joint_indices: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    joint_q: wp.array[float],
):
    joint_id = joint_indices[wp.tid()]
    parent = joint_parent[joint_id]
    child = joint_child[joint_id]

    X_wpj = joint_X_p[joint_id]
    if parent >= 0:
        X_wpj = body_q[parent] * X_wpj

    X_wcj = body_q[child] * joint_X_c[joint_id]

    x_err_c = wp.quat_rotate_inv(
        wp.transform_get_rotation(X_wpj),
        wp.transform_get_translation(X_wcj) - wp.transform_get_translation(X_wpj),
    )
    q_pc = wp.quat_inverse(wp.transform_get_rotation(X_wpj)) * wp.transform_get_rotation(X_wcj)

    q_start = joint_q_start[joint_id]
    joint_q[q_start + 0] = x_err_c[0]
    joint_q[q_start + 1] = x_err_c[1]
    joint_q[q_start + 2] = x_err_c[2]
    joint_q[q_start + 3] = q_pc[0]
    joint_q[q_start + 4] = q_pc[1]
    joint_q[q_start + 5] = q_pc[2]
    joint_q[q_start + 6] = q_pc[3]


@wp.kernel
def convert_free_distance_joint_qd_internal_to_public(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_qd_internal: wp.array[float],
    joint_qd_public: wp.array[float],
):
    joint_id = wp.tid()
    qd_start = joint_qd_start[joint_id]
    qd_end = joint_qd_start[joint_id + 1]
    jtype = joint_type[joint_id]

    if jtype != JointType.FREE and jtype != JointType.DISTANCE:
        for i in range(qd_start, qd_end):
            joint_qd_public[i] = joint_qd_internal[i]
        return

    parent = joint_parent[joint_id]
    child = joint_child[joint_id]

    X_wpj = joint_X_p[joint_id]
    if parent >= 0:
        X_wpj = body_q[parent] * X_wpj

    q_p = wp.transform_get_rotation(X_wpj)
    x_anchor_world = wp.transform_get_translation(X_wpj)
    x_child_com_world = wp.transform_point(body_q[child], body_com[child])
    r_child_com_parent = wp.quat_rotate_inv(q_p, x_child_com_world - x_anchor_world)

    v_internal_parent = wp.vec3(
        joint_qd_internal[qd_start + 0],
        joint_qd_internal[qd_start + 1],
        joint_qd_internal[qd_start + 2],
    )
    omega_parent = wp.vec3(
        joint_qd_internal[qd_start + 3],
        joint_qd_internal[qd_start + 4],
        joint_qd_internal[qd_start + 5],
    )
    v_com_parent = v_internal_parent + wp.cross(omega_parent, r_child_com_parent)

    joint_qd_public[qd_start + 0] = v_com_parent[0]
    joint_qd_public[qd_start + 1] = v_com_parent[1]
    joint_qd_public[qd_start + 2] = v_com_parent[2]
    joint_qd_public[qd_start + 3] = omega_parent[0]
    joint_qd_public[qd_start + 4] = omega_parent[1]
    joint_qd_public[qd_start + 5] = omega_parent[2]


@wp.kernel
def convert_free_distance_joint_f_public_to_internal(
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_f_public: wp.array[float],
    joint_f_internal: wp.array[float],
):
    """Copy non-free joint forces; FREE/DISTANCE wrenches are routed via body forces."""
    joint_id = wp.tid()
    qd_start = joint_qd_start[joint_id]
    qd_end = joint_qd_start[joint_id + 1]
    jtype = joint_type[joint_id]

    if jtype != JointType.FREE and jtype != JointType.DISTANCE:
        for i in range(qd_start, qd_end):
            joint_f_internal[i] = joint_f_public[i]
        return

    for i in range(qd_start, qd_end):
        joint_f_internal[i] = 0.0


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_I_m: wp.array[wp.spatial_matrix],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    # outputs
    body_qd: wp.array[wp.spatial_vector],
    joint_S_s: wp.array[wp.spatial_vector],
    body_solve_origin: wp.array[wp.vec3],
    body_I_s: wp.array[wp.spatial_matrix],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_end[index]

    solve_origin = wp.vec3()
    if start < end:
        root = start
        root_type = joint_type[root]
        if root_type == JointType.FREE or root_type == JointType.DISTANCE:
            # Floating roots are the numerically sensitive case: translating
            # the internal frame to the root COM keeps moment arms small while
            # preserving the public COM/world twist and wrench contract.
            solve_origin = wp.transform_get_translation(body_q_com[joint_child[root]])

    # compute link velocities and coriolis forces in the internal solve frame
    for i in range(start, end):
        compute_link_velocity(
            i,
            solve_origin,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            body_world,
            gravity,
            body_qd,
            joint_S_s,
            body_solve_origin,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_damping: wp.array[float],
    joint_effort_limit: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_q_com: wp.array[wp.transform],
    body_solve_origin: wp.array[wp.vec3],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_end[index]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        coord_start = joint_q_start[i]
        target_q_start = joint_target_q_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        # total forces on body
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext_public = body_f_ext[child]
        force = wp.spatial_top(f_ext_public)
        torque_com = wp.spatial_bottom(f_ext_public)
        x_com_s = wp.transform_get_translation(body_q_com[child]) - body_solve_origin[child]
        f_ext = -wp.spatial_vector(force, torque_com + wp.cross(x_com_s, force))
        # GRADIENT BUG (do NOT restore this in-place write): overwriting body_f_ext[child] with the
        # converted/negated wrench makes warp's reverse tape replay the contact adjoint against the
        # OVERWRITTEN value -- `force` in the backward of cross(x_com_s, force) becomes -force,
        # sign-flipping d(wrench)/d(x_com) and corrupting the configuration part of the contact
        # transition Jacobian (anti-correlated friction gradient at depth; grounded BPTT grad vs dflex
        # went cos 0.58 -> 0.99 once this write was removed). f_ext is consumed locally below; body_f_ext
        # is left holding the original COM-frame wrench and compute_body_parent_f re-derives f_ext
        # out-of-place from it. (was: `body_f_ext[child] = f_ext`)
        f_s = f_b_s + f_t_s + f_ext

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            joint_damping,
            joint_effort_limit,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_f,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            target_q_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)


# builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_J_start: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    # outputs
    J: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_end[index]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[index]

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]

                for k in range(6):
                    J[J_offset + dense_index(articulation_dof_count, row_start + k, col)] = S[k]

            j = joint_ancestor[j]


@wp.func
def contact_point_jacobian_column(
    body: int,
    point: wp.vec3,
    column: int,
    dof_count: int,
    J_start: int,
    body_jacobian_row: wp.array[int],
    body_solve_origin: wp.array[wp.vec3],
    J: wp.array[float],
):
    """Return one generalized-velocity column of a body's contact-point Jacobian.

    ``J`` is the articulation spatial Jacobian (rows ``[linear@solve_origin; angular]`` per body, see
    :func:`eval_rigid_jacobian`), and all bodies of an articulation share one ``body_solve_origin``, so
    the world point velocity is ``linear + angular x (point - solve_origin)`` -- the same frame map
    ``eval_rigid_id`` uses for the solve-frame twist.
    """
    row = body_jacobian_row[body] * 6
    linear = wp.vec3(
        J[J_start + dense_index(dof_count, row + 0, column)],
        J[J_start + dense_index(dof_count, row + 1, column)],
        J[J_start + dense_index(dof_count, row + 2, column)],
    )
    angular = wp.vec3(
        J[J_start + dense_index(dof_count, row + 3, column)],
        J[J_start + dense_index(dof_count, row + 4, column)],
        J[J_start + dense_index(dof_count, row + 5, column)],
    )
    return linear + wp.cross(angular, point - body_solve_origin[body])


@wp.func
def relative_contact_jacobian_column(
    column: int,
    body_a: int,
    point_a: wp.vec3,
    body_b: int,
    point_b: wp.vec3,
    dof_count: int,
    J_start: int,
    body_jacobian_row: wp.array[int],
    body_solve_origin: wp.array[wp.vec3],
    J: wp.array[float],
):
    """One column of the relative contact-point velocity Jacobian J_rel (the full 3-vector, NOT yet split
    into normal/tangential). A body index < 0 (world, kinematic, or a different articulation) is skipped,
    so this serves both the coupled same-articulation block (both bodies present) and a single-body
    diagonal block (one body present). The relative sign of body B is irrelevant to J^T J."""
    col = wp.vec3(0.0)
    if body_a >= 0:
        col += contact_point_jacobian_column(
            body_a, point_a, column, dof_count, J_start, body_jacobian_row, body_solve_origin, J
        )
    if body_b >= 0:
        col -= contact_point_jacobian_column(
            body_b, point_b, column, dof_count, J_start, body_jacobian_row, body_solve_origin, J
        )
    return col


@wp.func
def accumulate_contact_block(
    articulation: int,
    body_a: int,
    point_a: wp.vec3,
    body_b: int,
    point_b: wp.vec3,
    n: wp.vec3,
    scale_n: float,
    scale_t: float,
    body_jacobian_row: wp.array[int],
    articulation_J_start: wp.array[int],
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    body_solve_origin: wp.array[wp.vec3],
    J: wp.array[float],
    # output: the effective inertia H_eff (pre-initialized to H_base), accumulated in place
    H_eff: wp.array[float],
):
    """Accumulate ``scale_n * J_n^T J_n + scale_t * J_t^T J_t`` into one articulation's dense inertia
    block, where ``J_n = n^T J_rel`` (normal contact damping) and ``J_t = (I - nn^T) J_rel`` (tangential
    viscous friction) split the relative contact-point Jacobian into its along-normal and off-normal
    parts.

    The FULL (symmetric) block is written, mirroring the dense gemm that fills ``H`` -- the linear
    solve's reverse pass touches both triangles, so filling both makes the gradient (which flows from
    ``H_eff`` back into ``J`` and the contact points) aggregate over both triangles, as for ``H``. Only a
    contact body's ancestor-dof columns are nonzero in ``J`` (the rest are exact zeros from
    :func:`eval_rigid_jacobian`), so off-chain entries contribute nothing.
    """
    dof_count = articulation_H_rows[articulation]
    J_start = articulation_J_start[articulation]
    H_start = articulation_H_start[articulation]
    for i in range(dof_count):
        col_i = relative_contact_jacobian_column(
            i, body_a, point_a, body_b, point_b, dof_count, J_start, body_jacobian_row, body_solve_origin, J
        )
        ndot_i = wp.dot(n, col_i)
        tan_i = col_i - n * ndot_i
        for j in range(dof_count):
            col_j = relative_contact_jacobian_column(
                j, body_a, point_a, body_b, point_b, dof_count, J_start, body_jacobian_row, body_solve_origin, J
            )
            ndot_j = wp.dot(n, col_j)
            tan_j = col_j - n * ndot_j
            wp.atomic_add(
                H_eff,
                H_start + dense_index(dof_count, i, j),
                scale_n * ndot_i * ndot_j + scale_t * wp.dot(tan_i, tan_j),
            )


@wp.kernel
def eval_implicit_contact_friction(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_kf: wp.array[float],
    shape_material_ka: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_articulation: wp.array[int],
    body_jacobian_row: wp.array[int],
    articulation_J_start: wp.array[int],
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    body_solve_origin: wp.array[wp.vec3],
    J: wp.array[float],
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    rigid_contact_stiffness: wp.array[float],
    rigid_contact_damping: wp.array[float],
    rigid_contact_friction_scale: wp.array[float],
    friction_smoothing: float,
    dt: float,
    # output: the effective inertia H_eff, pre-initialized to the cached base inertia H_base before this
    # launch; this kernel accumulates dt * sum (kd*(-d) J_n^T J_n + kf J_t^T J_t) into it in place.
    H_eff: wp.array[float],
):
    """Fold the velocity-linear branches of rigid contact into the generalized inertia (backward Euler),
    ``H_eff = H_base + dt * sum_contacts [ kd*(-d) J_n^T J_n  +  kf J_t^T J_t ]``, then the caller
    factors/solves the symmetric system. Two contributions, both implicit-ONLY (euler never launches this
    kernel, so its behavior is unchanged):

    1. NORMAL damping (``J_n = n^T J_rel``): the backward-Euler form of ``eval_body_contact``'s normal
       damping ``fd = kd*(-d)*min(vn,0)`` (same kd, same depth scaling). Explicitly this term is
       stability-limited (a stiff foot impact needs large kd, which would blow up under euler); folded
       here it is unconditionally stable, so kd can be raised enough to ABSORB landing impacts instead of
       rebounding -- the explicit penalty contact's bounce is what made an open-loop trajectory replay
       slip/topple where a compliant (MuJoCo solref) contact tracks. Added for every penetrating contact
       on approach (vn<0), independent of the friction branch.
    2. TANGENTIAL viscous friction (``J_t = (I - nn^T) J_rel``): the linear ``ft = kf*vt`` branch, folded
       so the per-substep tangent contraction is ``1/(1+dt*lambda)`` (stable) instead of ``1-dt*lambda``
       -- this is what tames the BPTT gradient for G1's many planted feet. Only the foldable viscous
       branch (``kf*|vt| <= -mu*fn``); the Coulomb-capped (sliding) branch stays explicit.

    The matching explicit forces (fn+fd, ft) stay in ``joint_tau`` via ``eval_body_contact``; combined
    with this augmentation each linear branch is integrated by backward Euler (1-DOF friction check:
    ``v_next = m/(m+dt*kf)*v``).

    The full (exact) gradient is taken: the linear solve's adjoint w.r.t. ``H_eff`` flows back through
    this accumulation into ``J`` and the contact points and through the H_base copy into the cached
    inertia, while the velocity->velocity contraction comes from ``H_eff``'s value in the solve plus the
    explicit forces in ``joint_tau``. FD-verified; pathological near-singular contacts can still produce
    a large (but correct) gradient -- left for the caller's gradient guard, not masked here.

    Contact set, materials, geometry, and the viscous predicate come from the SHARED
    :func:`~newton._src.solvers.semi_implicit.kernels_contact.eval_contact_kinematics`, so they are
    identical to the explicit force in ``eval_body_contact`` -- the force and its derivative cannot
    diverge.

    CROSS-ARTICULATION LIMITATION: the dense solver factors one independent block per articulation, so
    a contact between two DIFFERENT articulations has no off-diagonal block to fill -- each side gets
    its own positive diagonal block (a block-Jacobi approximation). Robot-ground contacts (the world
    side carries no articulation) and contacts within a single articulation are exact.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    # SHARED contact resolution. force_in_world_frame=False matches eval_body_contact's Featherstone
    # launch (relative point velocity uses the COM lever arm r, not the world point).
    ck = eval_contact_kinematics(
        tid,
        body_q,
        body_qd,
        body_com,
        shape_material_ke,
        shape_material_kd,
        shape_material_kf,
        shape_material_ka,
        shape_material_mu,
        shape_body,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_shape0,
        contact_shape1,
        contact_margin0,
        contact_margin1,
        rigid_contact_stiffness,
        rigid_contact_damping,
        rigid_contact_friction_scale,
        False,
    )
    if not ck.active or ck.d >= 0.0:
        return

    # NORMAL damping coefficient: backward-Euler fold of eval_body_contact's normal damping. implicitfast
    # uses the NON-depth-scaled dashpot fd = kd*min(vn,0)*step(d) (see eval_body_contact), so the matching
    # coefficient is C = kd (constant, NOT kd*(-d)). Active on approach (vn<0). Folding it makes large
    # (near-critical) kd stable, which is what lets the contact ABSORB impacts instead of rebounding.
    scale_n = float(0.0)
    if ck.vn < 0.0:
        scale_n = dt * ck.kd

    # TANGENTIAL viscous-friction coefficient, identical predicate to eval_body_contact's friction:
    # ft = (vt/|vt|) * min(kf*|vt|, -mu*fn). The min selects the linear viscous term kf*|vt|
    # (=> ft = kf*vt, foldable) iff kf*|vt| <= -mu*fn; otherwise Coulomb-capped -> stays explicit.
    scale_t = float(0.0)
    speed = wp.length(ck.vt)
    # NEWTON_FRICTION_CAP_TOTAL=1: same TOTAL-normal-force cap as eval_body_contact's explicit friction
    # (implicitfast dashpot fd = kd*min(vn,0); d < 0 is guaranteed by the early-out above), so the fold
    # predicate selects the viscous branch on exactly the contacts whose explicit min() does.
    cap = -ck.mu * ck.fn
    if FRICTION_CAP_TOTAL != 0:
        cap = -ck.mu * (ck.fn + wp.min(ck.vn, 0.0) * ck.kd)
    if speed > 1.0e-6 * friction_smoothing and ck.kf * speed <= cap:
        scale_t = dt * ck.kf

    if scale_n == 0.0 and scale_t == 0.0:
        return

    body_a = ck.body_a
    body_b = ck.body_b
    n = ck.n

    articulation_a = int(-1)
    articulation_b = int(-1)
    if body_a >= 0 and (body_flags[body_a] & BodyFlags.KINEMATIC) == 0:
        articulation_a = body_articulation[body_a]
    if body_b >= 0 and (body_flags[body_b] & BodyFlags.KINEMATIC) == 0:
        articulation_b = body_articulation[body_b]
    if articulation_a < 0 and articulation_b < 0:
        return

    if articulation_a >= 0 and articulation_a == articulation_b:
        # Both contact bodies in one articulation -> exact coupled relative block.
        accumulate_contact_block(
            articulation_a, body_a, ck.bx_a, body_b, ck.bx_b, n, scale_n, scale_t,
            body_jacobian_row, articulation_J_start, articulation_H_start, articulation_H_rows,
            body_solve_origin, J, H_eff,
        )
    else:
        # World contact or cross-articulation -> independent positive diagonal block(s).
        if articulation_a >= 0:
            accumulate_contact_block(
                articulation_a, body_a, ck.bx_a, -1, ck.bx_b, n, scale_n, scale_t,
                body_jacobian_row, articulation_J_start, articulation_H_start, articulation_H_rows,
                body_solve_origin, J, H_eff,
            )
        if articulation_b >= 0:
            accumulate_contact_block(
                articulation_b, -1, ck.bx_a, body_b, ck.bx_b, n, scale_n, scale_t,
                body_jacobian_row, articulation_J_start, articulation_H_start, articulation_H_rows,
                body_solve_origin, J, H_eff,
            )


@wp.kernel
def eval_implicit_joint_limit(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_damping: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    # outputs (accumulated in place; mirror the contact-friction fold's H_eff accumulation)
    H_eff: wp.array[float],
    tau: wp.array[float],
):
    """implicitfast ONLY: backward-Euler fold of the ACTIVE, UNSATURATED hard joint-limit spring/damper
    into the dense generalized inertia H_eff -- the joint-space analogue of ``eval_implicit_contact_friction``.

    The hard joint limit (``kernels_body.joint_force``) applies an explicit penalty ``limit_ke*(limit - q)``
    plus a damper ``-limit_kd*qd`` whenever q is past a limit. Integrated EXPLICITLY (as euler does, and as
    implicitfast did before this kernel), that stiff spring on a light distal link is an EXPANSIVE BPTT mode
    -- the per-substep velocity tangent is ``1 - dt^2*limit_ke/m`` whose magnitude exceeds 1, so the adjoint
    grows geometrically over H*substeps (the G1 epoch-1 inf: an ankle-roll driven ~0.01 rad past its limit;
    no contact involved). Folding the velocity-linear part by backward Euler makes the per-substep velocity
    contraction ``m/(m + dt^2*limit_ke + dt*limit_kd) < 1`` (unconditionally stable), exactly as the
    contact-friction fold tames the planted-foot gradient.

    For each revolute/prismatic/D6 DOF whose limit is violated (q<lower or q>upper) AND whose joint_force is
    NOT effort-saturated (a clamped force has zero tangent -> ``jcalc_tau``'s clamp already bounds it, so
    folding it would over-stiffen), add ``dt*limit_kd + dt^2*limit_ke`` to the H_eff diagonal and subtract
    ``dt*limit_ke*qd`` from tau -- the ``q_{n+1}=q_n+dt*qd_{n+1}`` spring-velocity coupling, the only RHS
    term not already in the explicit force ``jcalc_tau`` left in tau. The active+unsaturated predicate and
    all inputs mirror ``jcalc_tau`` exactly, so the folded force and its tangent cannot diverge from the
    explicit one. euler never launches this kernel -> its trajectory and gradient are byte-identical.
    """
    index = wp.tid()  # one thread per articulation
    H_start = articulation_H_start[index]
    H_rows = articulation_H_rows[index]
    dof_base = articulation_dof_start[index]
    start = articulation_start[index]
    end = articulation_end[index]

    for i in range(start, end):
        type = joint_type[i]
        if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
            dof_start = joint_qd_start[i]
            coord_start = joint_q_start[i]
            target_q_start = joint_target_q_start[i]
            axis_count = joint_dof_dim[i, 0] + joint_dof_dim[i, 1]
            for a in range(axis_count):
                j = dof_start + a
                q = joint_q[coord_start + a]
                qd = joint_qd[j]
                lower = joint_limit_lower[j]
                upper = joint_limit_upper[j]
                if q < lower or q > upper:
                    limit_ke = joint_limit_ke[j]
                    limit_kd = joint_limit_kd[j]
                    # Recompute the (unclamped) joint force to test effort saturation; identical inputs to
                    # jcalc_tau so this matches the value it clamps. When the limit is violated joint_force
                    # zeroes the target PD term, so this is limit + limit-damper + passive damping.
                    drive_f = joint_force(
                        q,
                        qd,
                        joint_target_q[target_q_start + a],
                        joint_target_qd[j],
                        joint_target_ke[j],
                        joint_target_kd[j],
                        lower,
                        upper,
                        limit_ke,
                        limit_kd,
                        joint_damping[j],
                    )
                    if wp.abs(drive_f) < joint_effort_limit[j]:  # UNSATURATED -> nonzero limit tangent
                        local = j - dof_base
                        wp.atomic_add(
                            H_eff, H_start + dense_index(H_rows, local, local), dt * limit_kd + dt * dt * limit_ke
                        )
                        wp.atomic_add(tau, j, -dt * limit_ke * qd)


@wp.func
def spatial_mass(
    body_I_s: wp.array[wp.spatial_matrix],
    joint_start: int,
    joint_count: int,
    M_start: int,
    # outputs
    M: wp.array[float],
):
    stride = joint_count * 6
    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + dense_index(stride, l * 6 + i, l * 6 + j)] = I[i, j]


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_M_start: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    M: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_end[index]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.func
def spatial_mass_blockdiag(
    body_I_s: wp.array[wp.spatial_matrix],
    joint_start: int,
    joint_count: int,
    M_start: int,
    # outputs
    M: wp.array[float],
):
    # COMPACT block-diagonal store: only the joint_count 6x6 diagonal blocks (body spatial inertias),
    # packed contiguously as M[M_start + l*36 + i*6 + j]. This is the (num_joints*6, 6) layout the tile
    # GEMM reads (body l at rows [l*6:l*6+6]); 30x smaller than the dense (6*nj)^2 mass matrix, which is
    # 96% zeros. The off-diagonal coupling lives in the Jacobian J, not M, so nothing is lost.
    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + l * 36 + i * 6 + j] = I[i, j]


@wp.kernel
def eval_rigid_mass_blockdiag(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_M_start: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    M: wp.array[float],
):
    # one thread per-articulation; writes the COMPACT block-diagonal mass matrix (use_tile_gemm path).
    index = wp.tid()
    joint_start = articulation_start[index]
    joint_count = articulation_end[index] - joint_start
    spatial_mass_blockdiag(body_I_s, joint_start, joint_count, articulation_M_start[index], M)


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A_start: int,
    B_start: int,
    C_start: int,
    A: wp.array[float],
    B: wp.array[float],
    # outputs
    C: wp.array[float],
):
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[A_start + a_i] * B[B_start + b_j]

            if add_to_C:
                C[C_start + i * n + j] += sum
            else:
                C[C_start + i * n + j] = sum


# @wp.func_grad(dense_gemm)
# def adj_dense_gemm(
#     m: int,
#     n: int,
#     p: int,
#     transpose_A: bool,
#     transpose_B: bool,
#     add_to_C: bool,
#     A_start: int,
#     B_start: int,
#     C_start: int,
#     A: wp.array[float],
#     B: wp.array[float],
#     # outputs
#     C: wp.array[float],
# ):
#     add_to_C = True
#     if transpose_A:
#         dense_gemm(p, m, n, False, True, add_to_C, A_start, B_start, C_start, B, wp.adjoint[C], wp.adjoint[A])
#         dense_gemm(p, n, m, False, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])
#     else:
#         dense_gemm(
#             m, p, n, False, not transpose_B, add_to_C, A_start, B_start, C_start, wp.adjoint[C], B, wp.adjoint[A]
#         )
#         dense_gemm(p, n, m, True, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])


def create_inertia_matrix_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_tile(J_arr: wp.array3d[float], M_arr: wp.array3d[float], H_arr: wp.array3d[float]):
        articulation = wp.tid()

        # H = J^T M J, where M is block-diagonal in 6x6 inertia blocks (one per body), so
        #   H = sum_body J_body^T M_body J_body.
        # Accumulate body-by-body: only a 6 x num_dofs Jacobian slice (and J_body^T M_body) plus the
        # num_dofs x num_dofs accumulator are ever live in shared memory. Materializing the full
        # (6*num_joints) x num_dofs J and P tiles overflowed the GPU shared-memory cap in the BACKWARD
        # pass (the adjoint also holds J/P gradients); the blocked form keeps both passes within budget.
        #
        # Associate as (J_body^T M_body) J_body, NOT J_body^T (M_body J_body). The latter form
        # (P = M_body J_body; H += J_body^T P) tickles a Warp tile-autodiff aliasing bug: when the same
        # loaded J_body tile feeds one matmul as the 2nd operand (M_body J_body) and another as the
        # transposed 1st operand (J_body^T P), the adjoint of the OTHER operand (M_body) is accumulated
        # twice -> dH/dM is 2x too large (dH/dJ stays correct). Computing Q = J_body^T M_body first and
        # then H += Q J_body avoids that aliasing and yields the correct adjoints for both J and M.
        H = wp.tile_zeros(shape=(num_dofs, num_dofs), dtype=float)

        for i in range(int(num_joints)):
            # body i's 6x6 spatial inertia, stored COMPACT block-diagonal: M_arr is (num_joints*6, 6)
            # per articulation, so body i's block is rows [i*6 : i*6+6] (not the dense (i*6, i*6) diagonal).
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, 0))

            # the 6 x num_dofs Jacobian rows for body i
            J_body = wp.tile_load(J_arr[articulation], shape=(6, num_dofs), offset=(i * 6, 0))

            # Q_body = J_body^T M_body   (num_dofs x 6)
            Q_body = wp.tile_matmul(wp.tile_transpose(J_body), M_body)

            # accumulate H += Q_body J_body = J_body^T M_body J_body
            wp.tile_matmul(Q_body, J_body, H)

        wp.tile_store(H_arr[articulation], H)

    return eval_dense_gemm_tile


def create_cholesky_kernel(num_dofs):
    # Tile-API Cholesky of the per-articulation system matrix: L L^T = A + diag(R), one thread BLOCK per
    # articulation (cooperative) instead of the scalar dense_cholesky's one thread (O(n^3) serial). Produces
    # the SAME lower-triangular L layout that dense_subs (eval_dense_solve_batched) reads, so it is a drop-in
    # forward replacement for eval_dense_cholesky_batched on the use_tile_gemm path. enable_backward=False:
    # dense_cholesky's adjoint is a nop (the H gradient flows through the solve's adj_dense_solve, which reads
    # the stored L), so no Cholesky backward is needed -- this also skips the wasteful forward-replay.
    @wp.kernel(enable_backward=False)
    def eval_tile_cholesky(A_arr: wp.array3d[float], R_arr: wp.array2d[float], L_arr: wp.array3d[float]):
        articulation = wp.tid()
        A = wp.tile_load(A_arr[articulation], shape=(num_dofs, num_dofs))
        R = wp.tile_load(R_arr[articulation], shape=num_dofs)
        L = wp.tile_cholesky(wp.tile_diag_add(A, R))
        wp.tile_store(L_arr[articulation], L)

    return eval_tile_cholesky


def create_batched_cholesky_kernel(num_dofs):
    assert num_dofs == 18

    @wp.kernel
    def eval_tiled_dense_cholesky_batched(A: wp.array3d[float], R: wp.array2d[float], L: wp.array3d[float]):
        articulation = wp.tid()

        a = wp.tile_load(A[articulation], shape=(num_dofs, num_dofs), storage="shared")
        r = wp.tile_load(R[articulation], shape=num_dofs, storage="shared")
        a_r = wp.tile_diag_add(a, r)
        l = wp.tile_cholesky(a_r)
        wp.tile_store(L[articulation], wp.tile_transpose(l))

    return eval_tiled_dense_cholesky_batched


def create_inertia_matrix_cholesky_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_and_cholesky_tile(
        J_arr: wp.array3d[float],
        M_arr: wp.array3d[float],
        R_arr: wp.array2d[float],
        H_arr: wp.array3d[float],
        L_arr: wp.array3d[float],
    ):
        articulation = wp.tid()

        # H = J^T M J, accumulated body-by-body so only small tiles are live in shared memory; see the
        # detailed rationale on create_inertia_matrix_kernel above (block-diagonal M, and the
        # (J_body^T M_body) J_body association that avoids the Warp tile-autodiff adj_M doubling).
        H = wp.tile_zeros(shape=(num_dofs, num_dofs), dtype=float)

        for i in range(int(num_joints)):
            # body i's 6x6 spatial inertia, stored COMPACT block-diagonal: M_arr is (num_joints*6, 6)
            # per articulation, so body i's block is rows [i*6 : i*6+6] (not the dense (i*6, i*6) diagonal).
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, 0))

            # the 6 x num_dofs Jacobian rows for body i
            J_body = wp.tile_load(J_arr[articulation], shape=(6, num_dofs), offset=(i * 6, 0))

            # Q_body = J_body^T M_body   (num_dofs x 6)
            Q_body = wp.tile_matmul(wp.tile_transpose(J_body), M_body)

            # accumulate H += Q_body J_body = J_body^T M_body J_body
            wp.tile_matmul(Q_body, J_body, H)

        wp.tile_store(H_arr[articulation], H)

        # cholesky L L^T = (H + diag(R))
        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")
        H_R = wp.tile_diag_add(H, R)
        L = wp.tile_cholesky(H_R)
        wp.tile_store(L_arr[articulation], L)

    return eval_dense_gemm_and_cholesky_tile


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array[int],
    n: wp.array[int],
    p: wp.array[int],
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array[int],
    B_start: wp.array[int],
    C_start: wp.array[int],
    A: wp.array[float],
    B: wp.array[float],
    C: wp.array[float],
):
    # on the CPU each thread computes the whole matrix multiply
    # on the GPU each block computes the multiply with one output per-thread
    batch = wp.tid()  # /kNumThreadsPerBlock;
    add_to_C = False

    dense_gemm(
        m[batch],
        n[batch],
        p[batch],
        transpose_A,
        transpose_B,
        add_to_C,
        A_start[batch],
        B_start[batch],
        C_start[batch],
        A,
        B,
        C,
    )


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array[float],
    R: wp.array[float],
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array[float],
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.func_grad(dense_cholesky)
def adj_dense_cholesky(
    n: int,
    A: wp.array[float],
    R: wp.array[float],
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array[float],
):
    # nop, use dense_solve to differentiate through (A^-1)b = x
    pass


@wp.kernel(enable_backward=False)
def eval_dense_cholesky_batched(
    A_starts: wp.array[int],
    A_dim: wp.array[int],
    R_starts: wp.array[int],
    A: wp.array[float],
    R: wp.array[float],
    L: wp.array[float],
):
    batch = wp.tid()

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = R_starts[batch]

    dense_cholesky(n, A, R, A_start, R_start, L)


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.func
def dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array[float],
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
    tmp: wp.array[float],
):
    # helper function to include tmp argument for backward pass
    dense_subs(n, L_start, b_start, L, b, x)


@wp.func_grad(dense_solve)
def adj_dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array[float],
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
    tmp: wp.array[float],
):
    if not tmp or not wp.adjoint[x] or not wp.adjoint[A] or not wp.adjoint[L]:
        return
    for i in range(n):
        tmp[b_start + i] = 0.0

    dense_subs(n, L_start, b_start, L, wp.adjoint[x], tmp)

    for i in range(n):
        wp.adjoint[b][b_start + i] += tmp[b_start + i]

    # A* = -adj_b*x^T
    for i in range(n):
        for j in range(n):
            wp.adjoint[L][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

    for i in range(n):
        for j in range(n):
            wp.adjoint[A][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]


@wp.kernel
def eval_dense_solve_batched(
    L_start: wp.array[int],
    L_dim: wp.array[int],
    b_start: wp.array[int],
    A: wp.array[float],
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
    tmp: wp.array[float],
):
    batch = wp.tid()

    dense_solve(L_dim[batch], L_start[batch], b_start[batch], A, L, b, x, tmp)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_X_c: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    parent = joint_parent[index]
    child = joint_child[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]

    jcalc_integrate(
        parent,
        joint_X_c[index],
        body_com[child],
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )


@wp.func
def integrate_body_pose_from_com_twist(
    X_wb: wp.transform,
    body_com: wp.vec3,
    qd_com_world: wp.spatial_vector,
    dt: float,
):
    q = wp.transform_get_rotation(X_wb)
    x_com = wp.transform_point(X_wb, body_com)

    v_com = wp.spatial_top(qd_com_world)
    w = wp.spatial_bottom(qd_com_world)

    drdt = wp.quat(w, 0.0) * q * 0.5
    q_new = wp.normalize(q + drdt * dt)
    x_com_new = x_com + v_com * dt
    x_origin_new = x_com_new - wp.quat_rotate(q_new, body_com)

    return wp.transform(x_origin_new, q_new)


@wp.kernel
def correct_free_distance_body_pose_from_world_twist(
    joint_indices: wp.array[int],
    joint_child: wp.array[int],
    body_com: wp.array[wp.vec3],
    body_q_in: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
    dt: float,
):
    joint_id = joint_indices[wp.tid()]
    child = joint_child[joint_id]
    body_q_out[child] = integrate_body_pose_from_com_twist(body_q_in[child], body_com[child], body_qd_out[child], dt)


@wp.kernel
def zero_kinematic_joint_qdd(
    joint_child: wp.array[int],
    body_flags: wp.array[wp.int32],
    joint_qd_start: wp.array[int],
    joint_qdd: wp.array[float],
):
    """Zero joint accelerations for joints whose child body is kinematic."""
    joint_id = wp.tid()
    child = joint_child[joint_id]
    if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
        return

    dof_start = joint_qd_start[joint_id]
    dof_end = joint_qd_start[joint_id + 1]
    for i in range(dof_start, dof_end):
        joint_qdd[i] = 0.0


@wp.kernel
def copy_kinematic_joint_state(
    joint_child: wp.array[int],
    body_flags: wp.array[wp.int32],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_in: wp.array[float],
    joint_qd_in: wp.array[float],
    joint_q_out: wp.array[float],
    joint_qd_out: wp.array[float],
):
    """Copy prescribed joint state through the solve for kinematic child bodies."""
    joint_id = wp.tid()
    child = joint_child[joint_id]
    if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
        return

    q_start = joint_q_start[joint_id]
    q_end = joint_q_start[joint_id + 1]
    for i in range(q_start, q_end):
        joint_q_out[i] = joint_q_in[i]

    qd_start = joint_qd_start[joint_id]
    qd_end = joint_qd_start[joint_id + 1]
    for i in range(qd_start, qd_end):
        joint_qd_out[i] = joint_qd_in[i]


# ============================================================================
# Forward Kinematics with Velocity Conversion for Featherstone
# ============================================================================
# Featherstone keeps FREE/DISTANCE joint speeds in its internal spatial-twist
# basis, but the public Newton state stores COM-referenced body twists. This
# helper evaluates FK from the internal joint_qd basis and writes public body_qd.


@wp.func
def eval_single_articulation_fk_with_velocity_conversion(
    joint_start: int,
    joint_end: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    for i in range(joint_start, joint_end):
        parent = joint_parent[i]
        child = joint_child[i]

        # compute transform across the joint
        type = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == JointType.PRISMATIC:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if type == JointType.REVOLUTE:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if type == JointType.BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])

            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(wp.vec3(), w)

        if type == JointType.FREE or type == JointType.DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )

            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )

            X_j = t
            v_j = v

        if type == JointType.D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            # unroll for loop to ensure joint actions remain differentiable
            # (since differentiating through a for loop that updates a local variable is not supported)

            if lin_axis_count > 0:
                axis = joint_axis[qd_start + 0]
                pos += axis * joint_q[q_start + 0]
                vel_v += axis * joint_qd[qd_start + 0]
            if lin_axis_count > 1:
                axis = joint_axis[qd_start + 1]
                pos += axis * joint_q[q_start + 1]
                vel_v += axis * joint_qd[qd_start + 1]
            if lin_axis_count > 2:
                axis = joint_axis[qd_start + 2]
                pos += axis * joint_q[q_start + 2]
                vel_v += axis * joint_qd[qd_start + 2]

            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            if ang_axis_count == 1:
                axis = joint_axis[iqd]
                rot = wp.quat_from_axis_angle(axis, joint_q[iq])
                vel_w = joint_qd[iqd] * axis
            if ang_axis_count == 2:
                rot, vel_w = compute_2d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                )
            if ang_axis_count == 3:
                rot, vel_w = compute_3d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_axis[iqd + 2],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_q[iq + 2],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                    joint_qd[iqd + 2],
                )

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)

        # transform from world to parent joint anchor frame
        X_wpj = X_pj
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        x_child_origin = wp.transform_get_translation(X_wc)
        v_parent_origin = wp.vec3()
        w_parent = wp.vec3()
        if parent >= 0:
            v_wp = body_qd[parent]
            w_parent = wp.spatial_bottom(v_wp)
            v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin)
        linear_joint_world = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_joint_world = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))
        if type == JointType.FREE or type == JointType.DISTANCE:
            v_j_world = transform_twist(X_wpj, v_j)
            linear_joint_origin = velocity_at_point(v_j_world, x_child_origin)
            angular_joint_world = wp.spatial_bottom(v_j_world)
        else:
            child_origin_offset_world = x_child_origin - wp.transform_get_translation(X_wcj)
            linear_joint_origin = linear_joint_world + wp.cross(angular_joint_world, child_origin_offset_world)

        v_wc_origin = wp.spatial_vector(v_parent_origin + linear_joint_origin, w_parent + angular_joint_world)

        body_q[child] = X_wc
        body_qd[child] = origin_twist_to_com_twist(v_wc_origin, X_wc, body_com[child])


@wp.kernel
def eval_articulation_fk_with_velocity_conversion(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_count: int,  # total number of articulations
    articulation_mask: wp.array[
        bool
    ],  # used to enable / disable FK for an articulation, if None then treat all as enabled
    articulation_indices: wp.array[int],  # can be None, articulation indices to process
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    # Determine which articulation to process
    if articulation_indices:
        # Using indices - get actual articulation ID from array
        articulation_id = articulation_indices[tid]
    else:
        # No indices - articulation ID is just the thread index
        articulation_id = tid

    # Bounds check
    if articulation_id < 0 or articulation_id >= articulation_count:
        return  # Invalid articulation index

    # early out if disabling FK for this articulation
    if articulation_mask:
        if not articulation_mask[articulation_id]:
            return

    joint_start = articulation_start[articulation_id]
    joint_end = articulation_end[articulation_id]

    eval_single_articulation_fk_with_velocity_conversion(
        joint_start,
        joint_end,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        # outputs
        body_q,
        body_qd,
    )


@wp.kernel
def eval_articulation_fk_with_velocity_conversion_from_joint(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_indices: wp.array[int],
    articulation_joint_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    articulation_id = articulation_indices[tid]
    joint_start = articulation_joint_start[tid]
    joint_end = articulation_end[articulation_id]

    eval_single_articulation_fk_with_velocity_conversion(
        joint_start,
        joint_end,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        # outputs
        body_q,
        body_qd,
    )


def eval_fk_with_velocity_conversion(
    model: Model,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    state: State,
    mask: wp.array[bool] | None = None,
    indices: wp.array[int] | None = None,
):
    """
    Evaluates Featherstone FK from internal free-joint speeds and writes public body twists.

    This helper mirrors :func:`newton.eval_fk`, but it expects Featherstone's
    internal FREE/DISTANCE ``joint_qd`` convention as input and still writes
    the public COM-referenced :attr:`State.body_qd` output.

    Args:
        model: The model to evaluate.
        joint_q: Generalized joint position coordinates, shape [joint_coord_count], float
        joint_qd: Generalized joint velocity coordinates, shape [joint_dof_count], float
        state: The state to update.
        mask: The mask to use to enable / disable FK for an articulation. If None then treat all as enabled, shape [articulation_count], bool
        indices: Integer indices of articulations to update. If None, updates all articulations.
                        Cannot be used together with mask parameter.
    """
    # Validate inputs
    if mask is not None and indices is not None:
        raise ValueError("Cannot specify both mask and indices parameters")

    # Determine launch dimensions
    if indices is not None:
        num_articulations = len(indices)
    else:
        num_articulations = model.articulation_count

    wp.launch(
        kernel=eval_articulation_fk_with_velocity_conversion,
        dim=num_articulations,
        inputs=[
            model.articulation_start,
            model.articulation_end,
            model.articulation_count,
            mask,
            indices,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
        ],
        outputs=[
            state.body_q,
            state.body_qd,
        ],
        device=model.device,
    )


def eval_fk_with_velocity_conversion_from_joint_starts(
    model: Model,
    articulation_indices: wp.array[int],
    articulation_joint_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    state: State,
):
    assert len(articulation_indices) == len(articulation_joint_start)

    wp.launch(
        kernel=eval_articulation_fk_with_velocity_conversion_from_joint,
        dim=len(articulation_indices),
        inputs=[
            model.articulation_start,
            model.articulation_end,
            articulation_indices,
            articulation_joint_start,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
        ],
        outputs=[
            state.body_q,
            state.body_qd,
        ],
        device=model.device,
    )


@wp.kernel
def compute_body_parent_f(
    body_q_com: wp.array[wp.transform],
    body_solve_origin: wp.array[wp.vec3],
    body_f_s: wp.array[wp.spatial_vector],
    body_ft_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    # output
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Populate ``State.body_parent_f`` from Featherstone's RNEA backward pass.

    The Featherstone backward pass leaves the per-body spatial wrench
    decomposed across three buffers:

    * ``body_f_s = I*a + spatial_cross_dual(v, I*v) - f_g_s``  (inertial bias minus gravity)
    * ``body_ft_s``                          (accumulated descendant wrenches)
    * ``body_f_ext``                         (external + contact wrenches,
      stored with the negated sign convention used by ``eval_rigid_tau``)

    Their sum is the spatial wrench transmitted from the parent through the
    inbound joint, expressed in Featherstone's internal solve frame. For
    floating-root articulations this frame is translated to the root COM; for
    other roots it remains at the world origin. We translate it to the body's
    COM (matching :class:`SolverMuJoCo` and the :attr:`State.body_parent_f`
    convention -- linear ``[N]`` first, torque ``[N·m]`` referenced to the COM,
    both in world frame).

    The kernel does not special-case roots: it writes the same
    RNEA-backward-pass sum for every body.  For a FREE-jointed body that
    has no kinematic parent the value is whatever wrench the recursion
    produces -- e.g. the residual needed to balance gravity against
    contacts/external forces in equilibrium, or the gyroscopic
    ``v x* (I*v)`` term during tumbling.  Treat it as a diagnostic
    rather than a true joint reaction in that case.
    """
    tid = wp.tid()

    # body_f_ext now holds the ORIGINAL COM-frame external/contact wrench: eval_rigid_tau no longer
    # overwrites it in-place with the converted/negated wrench (that in-place write corrupted warp's
    # reverse-mode tape -- it replayed the contact adjoint against the converted value, sign-flipping
    # d(wrench)/d(x_com)). So apply the same COM->solve-frame negation here, OUT-OF-PLACE, to reproduce
    # eval_rigid_tau's f_ext (the negated sign convention documented above).
    r_com = wp.transform_get_translation(body_q_com[tid]) - body_solve_origin[tid]
    f_ext_public = body_f_ext[tid]
    ext_force = wp.spatial_top(f_ext_public)
    ext_torque_com = wp.spatial_bottom(f_ext_public)
    f_ext = -wp.spatial_vector(ext_force, ext_torque_com + wp.cross(r_com, ext_force))

    f_s = body_f_s[tid] + body_ft_s[tid] + f_ext
    f_lin = wp.spatial_top(f_s)
    f_ang_at_origin = wp.spatial_bottom(f_s)

    f_ang_at_com = f_ang_at_origin - wp.cross(r_com, f_lin)

    body_parent_f[tid] = wp.spatial_vector(f_lin, f_ang_at_com)
