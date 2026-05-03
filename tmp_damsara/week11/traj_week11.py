import importlib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from simulator_cartesian import FialaBicycleCartesianSimulator, InitialVehicleState, SimulatorConfig
from tracking_helper import TrackingReference
from visualization import LiveVehicleAnimator, MPCVisualizationConfig

week10_dir = Path.cwd().parent / "week10"
if str(week10_dir) not in sys.path:
    sys.path.insert(0, str(week10_dir))

week10_traj = importlib.import_module("traj_opt_helper")
TrackDefinition = week10_traj.TrackDefinition
TrajOptConfig = week10_traj.TrajOptConfig
TrajOptProblem = week10_traj.TrajOptProblem

from tracking_helper import TrackingReference


track = TrackDefinition.oval(
    straight_length=20.0,
    radius=15.0,
    n_laps=1,
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

problem = TrajOptProblem(track, config)
result = problem.solve()
reference = TrackingReference.from_traj_result(result)


# plt.show()

# result.plot_states()

# result.plot_with_vehicles(
#     s_range=(track.s_start, track.s_end),
#     step=8,
#     color_state="V",
#     vehicle_scale=2.0,
# )


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



sim = FialaBicycleCartesianSimulator(
    config=SimulatorConfig(dt=0.02, integration_substeps=4, seed=5)
)

sim.step(
    roadwheel_angle=0.0,
    rear_wheel_torque=0.0,
    brake=0.0,
)

_, _, body_yaw = result.get_xy_trajectory()
sim.reset(
    InitialVehicleState(
        x=float(reference.x[0]),
        y=float(reference.y[0]),
        psi=float(body_yaw[0]),
        V=float(result.V[0]),
        beta=float(result.beta[0]),
        rear_wheelspeed_ms=float(result.wr[0]),
    )
)

last_s = None
frenet_log = []

for _ in range(200):
    meas = sim.get_state(noisy=True)

    frenet = reference.cartesian_to_frenet(
        x=meas["x"],
        y=meas["y"],
        psi=meas["psi"],
        beta=meas["beta"],
        last_s=last_s,
    )
    
    last_s = frenet.s

    frenet_log.append(
        {
            "time": meas["time"],
            "s": frenet.s,
            "e": frenet.e,
            "dphi": frenet.dphi,
        }
    )

    sim.step(
        roadwheel_angle=0.0,
        rear_wheel_torque=0.0,
        brake=0.0,
    )

history = sim.get_history()

print(type(history))

import matplotlib.pyplot as plt
plt.pause(0.1)

f_times = [entry["time"] for entry in frenet_log]
f_s     = [entry["s"]    for entry in frenet_log]
f_e     = [entry["e"]    for entry in frenet_log]
f_dphi  = [entry["dphi"] for entry in frenet_log]

# true_time, true_x, true_y, meas_x, meas_y, true_V, true_beta, true_yaw_rate, true_rear_wheelspeed_ms

t          = history["true_time"]
true_x     = history["true_x"]
true_y     = history["true_y"]
meas_x     = history["meas_x"]
meas_y     = history["meas_y"]
V          = history["true_V"]
beta       = history["true_beta"]
yaw_rate   = history["true_yaw_rate"]
wheelspeed = history["true_rear_wheelspeed_ms"]

plt.plot(f_times, f_e)
plt.xlabel('time [s]')
plt.ylabel('error [m]')
plt.show()

plt.plot(f_s, f_dphi)
plt.xlabel('path distance [m]')
plt.ylabel('dphi [rad]')
plt.show()

plt.plot(f_s[:199], V[:199])
plt.xlabel('path distance [m]')
plt.ylabel('velocity [m/s]')
plt.show()

plt.figure(2)
track.plot()
plt.plot(true_x, true_y, label = 'True X-Y')
plt.plot(reference.x, reference.y, label = 'Reference')
plt.grid(True)
plt.ylabel('Y [m]')
plt.xlabel('X [m]')
plt.legend()
plt.show()

animator = LiveVehicleAnimator(
    reference,
    cfg=MPCVisualizationConfig(),
    title="Week 11 Zero-Control Simulator",
)

for _ in range(2000):
    meas = sim.get_state(noisy=True)

    record = {
        "vehicle_pose": {
            "x": float(meas["x"]),
            "y": float(meas["y"]),
            "psi": float(meas["psi"]),
        },
        "control": {
            "roadwheel_angle": 0.0,
            "rear_wheel_torque": 0.0,
        },
    }
    animator.update(record)

    sim.step(
        roadwheel_angle=0.0,
        rear_wheel_torque=0.0,
        brake=0.0,
    )

    plt.pause(0.1)


# from mpc_helper import FrenetTrackingMPC, TrackingMPCConfig

# cfg = TrackingMPCConfig(
#     horizon_steps=15,
#     prediction_ds=1.0,
#     weight_speed=4.0,
#     weight_e=2.0,
#     weight_dphi=2.0,
#     terminal_e=2.0,
#     terminal_dphi=2.0,
# )

# mpc = FrenetTrackingMPC(config=cfg)

# from visualization import LiveMPCAnimator, MPCVisualizationConfig

# animator = LiveMPCAnimator(reference, cfg=MPCVisualizationConfig())

# animator.close()
# plt.show()

