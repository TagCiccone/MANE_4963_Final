import importlib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from simulator_cartesian import FialaBicycleCartesianSimulator, InitialVehicleState, SimulatorConfig
from tracking_helper import TrackingReference
from visualization import LiveVehicleAnimator, MPCVisualizationConfig

week10_dir = Path.cwd().parent / "week11"
if str(week10_dir) not in sys.path:
    sys.path.insert(0, str(week10_dir))

week10_traj = importlib.import_module("traj_opt_helper")
TrackDefinition = week10_traj.TrackDefinition
TrajOptConfig = week10_traj.TrajOptConfig
TrajOptProblem = week10_traj.TrajOptProblem

from tracking_helper import TrackingReference


from visualization import LiveMPCAnimator, MPCVisualizationConfig


track = TrackDefinition.oval(
    straight_length=20.0,
    radius=15.0,
    n_laps=1,
    e_half_width=4.0,
    lead_in=50.0,
    lead_out=50.0,
)

config = TrajOptConfig(
    num_nodes=1000,
    start_velocity=(3.0, 5.0),
    initial_state_bounds={
        "e": (-0.02, 0.02),
        "beta": (-0.05, 0.05),
        "delta": (-0.02, 0.02),
        "r": (-0.01, 0.01),
        "rear_wheel_torque": (0.0, 400.0),
    },
)

problem = TrajOptProblem(track, config)
result = problem.solve()
reference = TrackingReference.from_traj_result(result, reference_mode="centerline")

sim = FialaBicycleCartesianSimulator(
    config=SimulatorConfig(dt=0.02, integration_substeps=4, seed=5)
)

# sim.step(
#     roadwheel_angle=0.0,
#     rear_wheel_torque=0.0,
#     brake=0.0,
# )



kappa_track = reference.kappa
s_track = reference.s

pause_state = {"paused": False}
def on_key(event):
    if event.key == "enter":
        pause_state["paused"] = not pause_state["paused"]
        print(f"paused = {pause_state['paused']}")
    elif event.key == "escape":
        pause_state["paused"] = False
        print("continuing")

animator = LiveMPCAnimator(reference, cfg=MPCVisualizationConfig())
fig = animator.fig if animator is not None else plt.gcf()
fig.canvas.mpl_connect("key_press_event", on_key)


from mpc_helper import FrenetTrackingMPC, TrackingMPCConfig

cfg = TrackingMPCConfig(
    horizon_steps=20,
    prediction_ds=1.0,
    weight_speed=2.0,
    weight_e=10.0,
    weight_dphi=10.0,
    terminal_e=0.0,
    terminal_dphi=0.0,
)

mpc = FrenetTrackingMPC(config=cfg)



prev_control = {
    "roadwheel_angle": 0.0,
    "rear_wheel_torque": 0.0,
}
last_s = None
closed_loop_log = []
horizon_length = cfg.horizon_steps * (cfg.prediction_ds or 1.0)
s_stop = reference.s[-1] - horizon_length

print(f"s_stop = {s_stop}")

step_count = 0
kappa_predicted_list = []
s_tracking_list = []

V_list = []
dphi_list = []
e_list = []
s_list = []

delta_list = []
torque_list = []

V_list_mu = []
dphi_list_mu = []
e_list_mu = []
s_list_mu = []

delta_list_mu = []
torque_list_mu = []

cases = [1,2]

for case_num in cases:
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
    step_count = 0
    while last_s is None or last_s < s_stop:
        meas = sim.get_state(noisy=True)

        frenet = reference.cartesian_to_frenet(
            x=meas["x"],
            y=meas["y"],
            psi=meas["psi"],
            beta=meas["beta"],
            last_s=last_s,
        )
        last_s = frenet.s

        # print(f"last s is = {last_s}")
        # print(tracking_ref)

        # TODO: inspect TrackingReference.get_ref_traj(...) in tracking_helper.py
        # and build the preview window from the current s position.
        tracking_ref = reference.get_ref_traj(    
            s_start=last_s,       
            horizon_steps=horizon_length,  
            ds=1.0
        )

        current_state = {
            "s": frenet.s,
            "time": meas["time"],
            "e": frenet.e,
            "dphi": frenet.dphi,
            "r": meas["yaw_rate"],
            "V": meas["V"],
            "beta": meas["beta"],
            "wr": meas["rear_wheelspeed_ms"],
        }
        V_list.append(meas["V"])
        e_list.append(frenet.e)
        dphi_list.append(frenet.dphi)
        s_list.append(frenet.s)


        s_tracking = tracking_ref["s"]
        kappa_traj_predicted = tracking_ref["kappa_traj"]

        s_tracking_list.append(s_tracking)
        kappa_predicted_list.append(kappa_traj_predicted)

        # print(f" current state is = {current_state}")

        # TODO: inspect FrenetTrackingMPC.solve(...) in mpc_helper.py
        # and solve one MPC step using current_state, your preview window,
        # and prev_control.
        solution = mpc.solve(current_state, tracking_ref)

        # TODO: extract the first control input from the solver output
        # and apply only that first command to the simulator.
        applied_control = solution["u0"] 

        


        print(f"applied control = {applied_control}")
        delta_list.append(applied_control["roadwheel_angle"])
        torque_list.append(applied_control["rear_wheel_torque"])

        sim.step(
            roadwheel_angle=applied_control["roadwheel_angle"],
            rear_wheel_torque=applied_control["rear_wheel_torque"],
            brake=0.0,
        )
        prev_control = dict(applied_control)

        # TODO: log the measured Frenet state, applied control, reference preview,
        # and predicted rollout for later plots.
        meas = sim.get_state(noisy=True)

        record = {
            "frenet": {
                "s": frenet.s,
                "e": frenet.e,
                "dphi": frenet.dphi,
            },
            "control": dict(applied_control),
            "vehicle_pose": {
                "x": float(meas["x"]),
                "y": float(meas["y"]),
                "psi": float(meas["psi"]),
            },
            "predicted_cartesian": {
                "x": tracking_ref["x"],
                "y": tracking_ref["y"],
                "psi": tracking_ref["yaw"],
            },
        }


        animator.update(record)
        print(
            f"step={last_s:.2f} "
            f"s={frenet.s:7.2f} "
            f"e={frenet.e: .3f} "
            f"dphi={frenet.dphi: .3f} "
            f"V={meas['V']: .2f} "
            f"delta={applied_control['roadwheel_angle']: .3f} "
            f"torque={applied_control['rear_wheel_torque']: .1f}"
        )

        if animator is not None:
            animator.update(record)

        # Small delay so you can see each iteration advancing.
        plt.pause(0.1)

        # Press Enter to pause or resume. Press Escape to force resume.
        while pause_state["paused"]:
            plt.pause(0.05)
        
        
        # if step_count > 200:
        #     animator.close()
        #     break

        step_count = step_count + 1



plt.figure(10)
fig, axes = plt.subplots(10, 1, figsize=(6, 10))  # 5 rows, 1 column
for i, ax in enumerate(axes):
    ax.plot(s_tracking_list[i], kappa_predicted_list[i])
    ax.set_xlabel('path distance [m]')
    ax.set_ylabel('kappa [1/m]')
    ax.grid(True)

plt.tight_layout()
plt.show()

plt.figure(21)
plt.plot(s_list, delta_list)
plt.xlabel('path distance [m]')
plt.ylabel('delta [rad]')
plt.grid(True)
plt.show()

plt.figure(22)
plt.plot(s_list, torque_list)
plt.xlabel('path distance [m]')
plt.ylabel('Torque [Nm]')
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(s_list, dphi_list)
plt.xlabel('path distance [m]')
plt.ylabel('dphi [rad]')
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(s_list, e_list)
plt.xlabel('path distance [m]')
plt.ylabel('e [m]')
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(s_list, V_list)
plt.xlabel('path distance [m]')
plt.ylabel('V [m/s]')
plt.grid(True)
plt.show()









# animator.close()
# plt.show()

