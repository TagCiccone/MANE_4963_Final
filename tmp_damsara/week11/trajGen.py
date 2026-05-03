import matplotlib.pyplot as plt
import numpy as np

from traj_opt_helper import (
    TrackDefinition,
    TrajOptProblem,
    TrajOptConfig,
    TrajOptResult,
    StateIdx,
    ControlIdx,
)

track = TrackDefinition.oval(
    straight_length=20.0,
    radius=15.0,
    n_laps=2,
    e_half_width=4.0,
    lead_in=50.0,
    lead_out=50.0,
)

config = TrajOptConfig(
    num_nodes=200,
    start_velocity=(3.0, 5.0),
    initial_state_bounds={
        "e": (-0.05, 0.05),
        "beta": (-0.05, 0.05),
        "delta": (-0.02, 0.02),
        "r": (-0.01, 0.01),
        "rear_wheel_torque": (0.0, 400.0),
    },
)

# problem = TrajOptProblem(track, config)
# problem.set_e_bounds(e_min=-1.0, e_max=1.0, s_range=(0.0, 500.0))
# problem.add_accel_bounds(a_min=0.0, a_max=3.0, s_range=(0.0, 500.0))
# result = problem.solve()

problem = TrajOptProblem(track, config)
problem.set_e_bounds(e_min=-2.0, e_max=2.0, s_range=(60.0, 120.0))
result = problem.solve()

# problem = TrajOptProblem(track, config)
# problem.set_state_range("rear_wheel_torque", lb=0.0, ub=3500.0)
# result = problem.solve()


# result.plot_states()

# result.plot_with_vehicles(
#     s_range=(track.s_start, track.s_end),
#     step=8,
#     color_state="V",
#     vehicle_scale=2.0,
# )

# plt.show()

force_data = result.compute_tire_forces()

result.plot_friction_circle(force_data)
plt.show()

result.plot_gg_diagram(force_data)
plt.show()

result.plot_force_vs_alpha_slip(force_data)
plt.show()

total_time = result.t[-1]
max_speed = result.V.max()
max_abs_e = np.abs(result.e).max()

beta_vals = result.beta

print(total_time, max_speed, max_abs_e)
print(f"beta = {beta_vals}")

# track.plot()
# plt.show()

# val_s = track.s
# kappa = track.kappa
# e_min = track.e_min
# e_max = track.e_max

# np.max(val_s)
# print(f"max s = {val_s}")

# plt.figure(1)
# plt.plot(val_s, kappa, label = "kappa vs s")
# plt.xlabel('s (m)')
# plt.ylabel('curvature (1/m)')
# plt.show()


# plt.figure(2)
# plt.plot(val_s, e_min, label = "e min vs s")
# plt.plot(val_s, e_max, label = "e max vs s")
# plt.xlabel('s (m)')
# plt.ylabel('error (m)')
# plt.show()

