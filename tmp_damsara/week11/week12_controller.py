from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import rclpy


_THIS_DIR = Path(__file__).resolve().parent
_WEEK10_DIR = _THIS_DIR.parent / "week10"
if str(_WEEK10_DIR) not in sys.path:
	sys.path.insert(0, str(_WEEK10_DIR))

from bng_controller.torque_speed_controller import TorqueSpeedController
from bng_simulator.utils.math_op import convert_euler_to_quaternion
from bng_simulator.utils.services_utils import send_request
from mpc_helper import FrenetTrackingMPC, TrackingMPCConfig
from tracking_helper import TrackingReference
from week12_helper import (
	ControllerRuntime,
	Week12Logger,
	build_plan,
	publish_live_visualizer,
	sample_plan,
	start_live_visualizer,
	stop_live_visualizer,
)


_week10_traj = importlib.import_module("traj_opt_helper")
_week10_dyn = importlib.import_module("casadi_dynamics")
TrackDefinition = _week10_traj.TrackDefinition
TrajOptConfig = _week10_traj.TrajOptConfig
TrajOptProblem = _week10_traj.TrajOptProblem
FialaBicycleCasADi = _week10_dyn.FialaBicycleCasADi
fiala_params = _week10_dyn.fiala_params


vehicle_name = "EGO"
reference_mode = "trajectory"

STEER_TO_ROADWHEEL_ANGLE = -0.5814544122705007



OVAL_REFERENCE_PRESETS = {
	"mild": {
		"track": {
			"straight_length": 25.0,
			"radius": 18.0,
			"n_laps": 1,
			"e_half_width": 4.0,
			"lead_in": 60.0,
			"lead_out": 60.0,
		},
		"trajopt": {
			"num_nodes": 180,
			"start_velocity": (3.0, 4.5),
			"cost_control_rate_weight": 2.0,
			"ipopt_max_iter": 1800,
		},
	},
	"medium": {
		"track": {
			"straight_length": 20.0,
			"radius": 15.0,
			"n_laps": 2,
			"e_half_width": 4.0,
			"lead_in": 50.0,
			"lead_out": 50.0,
		},
		"trajopt": {
			"num_nodes": 200,
			"start_velocity": (3.0, 5.0),
			"cost_control_rate_weight": 1.0,
			"ipopt_max_iter": 2200,
		},
	},
	"aggressive": {
		"track": {
			"straight_length": 15.0,
			"radius": 12.0,
			"n_laps": 1,
			"e_half_width": 3.5,
			"lead_in": 40.0,
			"lead_out": 40.0,
		},
		"trajopt": {
			"num_nodes": 220,
			"start_velocity": (4.0, 6.0),
			"cost_control_rate_weight": 0.35,
			"ipopt_max_iter": 2600,
		},
	},
}

def build_oval_reference(level: str = "aggressive", mode: str = "trajectory") -> tuple[object, TrackingReference]:
	if level not in OVAL_REFERENCE_PRESETS:
		raise ValueError(f"Unknown oval aggressiveness '{level}'")

	preset = OVAL_REFERENCE_PRESETS[level]
	track = TrackDefinition.oval(**preset["track"])
	cfg = TrajOptConfig(
		num_nodes=preset["trajopt"]["num_nodes"],
		start_velocity=preset["trajopt"]["start_velocity"],
		initial_state_bounds={
			"e": (-0.05, 0.05),
			"beta": (-0.05, 0.05),
			"delta": (-0.02, 0.02),
			"r": (-0.01, 0.01),
			"rear_wheel_torque": (0.0, 400.0),
		},
		cost_control_rate_weight=preset["trajopt"]["cost_control_rate_weight"],
		ipopt_print_level=0,
		ipopt_max_iter=preset["trajopt"]["ipopt_max_iter"],
	)
	result = TrajOptProblem(track, cfg).solve()
	return result, TrackingReference.from_traj_result(result, reference_mode=mode)


def build_straight_reference(mode: str = "trajectory") -> tuple[object, TrackingReference]:
	kappa_fn = lambda _s: 0.0
	track = TrackDefinition.from_kappa_function(
		kappa_fn,
		s_start=0.0,
		s_end=100.0,
		ds=2.0,
		e_half_width=4.0,
	)
	cfg = TrajOptConfig(
		num_nodes=200,
		start_velocity=(3.0, 5.0),
		initial_state_bounds={
			"e": (-0.05, 0.05),
			"beta": (-0.05, 0.05),
			"delta": (-0.02, 0.02),
			"r": (-0.01, 0.01),
			"rear_wheel_torque": (0.0, 400.0),
		},
		target_steady_velocity=(100.0, 20.0), # Change this if you want a different steady condition.
		cost_control_rate_weight=1.0,
		ipopt_print_level=0,
		ipopt_max_iter=1200,
	)
	result = TrajOptProblem(track, cfg).solve()
	reference = TrackingReference.from_traj_result(result, reference_mode=mode)
	return result, reference

reference_shape = "straight"
oval_aggressiveness = "mild"

if reference_shape == "oval":
	results, tracking_ref = build_oval_reference(level=oval_aggressiveness, mode=reference_mode)
else:
	results, tracking_ref = build_straight_reference(reference_mode)
# results, tracking_ref = build_straight_reference("trajectory")

# print(tracking_ref)


# def plot_reference_summary(reference: TrackingReference):
# 	x_left, y_left, _ = reference.frenet_to_cartesian(reference.s, reference.e_max, np.zeros_like(reference.s))
# 	x_right, y_right, _ = reference.frenet_to_cartesian(reference.s, reference.e_min, np.zeros_like(reference.s))
# 	fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))

# 	axs[0].plot(reference.x, reference.y, lw=2.0)
# 	axs[0].plot(x_left, y_left, "k--", lw=1.2, alpha=0.8, label="left bound")
# 	axs[0].plot(x_right, y_right, "k--", lw=1.2, alpha=0.8, label="right bound")
# 	axs[0].set_aspect("equal")
# 	axs[0].grid(True, alpha=0.3)
# 	axs[0].legend()

# 	axs[1].plot(reference.s, reference.kappa, lw=2.0)
# 	axs[1].set_xlabel("s [m]")
# 	axs[1].set_ylabel("kappa [1/m]")
# 	axs[1].grid(True, alpha=0.3)

# 	axs[2].plot(reference.s, reference.state_profiles["wr"], lw=2.0, label="wr_ref")
# 	axs[2].plot(reference.s, reference.state_profiles["rear_wheel_torque"], lw=2.0, label="rear torque ref")
# 	axs[2].set_xlabel("s [m]")
# 	axs[2].grid(True, alpha=0.3)
# 	axs[2].legend()

# 	fig.tight_layout()
# 	return fig


# plot_reference_summary(reference)
# plt.show()

################

ca_drift = FialaBicycleCasADi(fiala_params)
powertrain = ca_drift.build_powertrain_functions()
torque_est_fn = powertrain["torque_est_fn"]
inv_throttle_fn = powertrain["inv_throttle_fn"]

if not rclpy.ok():
	rclpy.init()

ctl = TorqueSpeedController(vehicle_name=vehicle_name, spin_in_thread=True)
time.sleep(2.0) # pause to let the subscription come up

def spawn_at_reference(reference: TrackingReference) -> None:
	x0 = float(reference.x[0])
	y0 = float(reference.y[0])
	yaw0_deg = float(np.degrees(reference.yaw[0]))
	z0 = 0
	rot_quat = convert_euler_to_quaternion((0.0, 0.0, np.radians(yaw0_deg + 90.0)))
	send_request(
		"vehicle.teleport",
		{
			"vehicle_name": vehicle_name,
			"pos": [x0, y0, z0],
			"rot_quat": [float(q) for q in rot_quat],
			"reset": False,
		},
	)
	time.sleep(3.0)

spawn_at_reference(tracking_ref)
input("Press Enter to continue once you have checked the spawn in BeamNG...")
# prediction_ds = 2.0
# horizon_steps = 15


def build_mpc() -> FrenetTrackingMPC:
    # Will need to play with all of this for best perf
    # The values here are far from tuned.
	return FrenetTrackingMPC(
		TrackingMPCConfig(
			horizon_steps=10,
			prediction_ds=2.0,
			weight_speed=0.1,
			weight_beta=0.1,
			weight_wr=0.1,
			weight_e=0.1,
			weight_dphi=0.1,
			terminal_e=0.1,
			terminal_dphi=0.1,
			ipopt_print_level=0,
			ipopt_max_iter=500,
			max_lateral_error=6.0,
			weight_steer=0.01,
			weight_torque=0.1,
			weight_steer_increment=0.1,
			weight_torque_increment=0.1,
			warm_start_duals=True,
		),
		state_bounds={"e": (-6.0, 6.0)},
	)

runtime = ControllerRuntime()
ros_logger = Week12Logger(vehicle_name) #if publish_ros else None


mpc = build_mpc()
# preview_horizon = mpc.cfg.horizon_steps * float(mpc.cfg.prediction_ds or 1.0)
# tracking_ref = traj_reference.get_ref_traj(    
# 	s_start=reference.s[0],       
# 	horizon_steps=preview_horizon,  
# 	ds=float(mpc.cfg.prediction_ds or 1.0)
# )
# s_stop = float(tracking_ref.s[-1] - preview_horizon)

s_stop = tracking_ref.s[-1]
solve_log: list[dict[str, object]] = []


live_visualizer = True
if live_visualizer:
	plt.close("all")
visualizer = start_live_visualizer(tracking_ref) if live_visualizer else None



def on_state_msg(msg) -> None:
	runtime.publish_raw_state(ctl.state_msg_to_dict(msg))

ctl.add_state_listener(on_state_msg)

def build_projected_state(
    reference: TrackingReference,
    seq: int,
    state: dict[str, float],
    last_s: float | None,
    init_time: float,
) -> tuple[dict[str, object], float]:
    frenet = reference.cartesian_to_frenet(
        x=state["x"],
        y=state["y"],
        psi=state["yaw"],
        beta=state["beta"],
        last_s=last_s,
    )
    current_state = {
        "s": frenet.s,
        "time": state["t"] - init_time,
        "r": state["r"],
        "V": max(state["V"], 2.0),
        "beta": state["beta"],
        "wr": max(state["wr"], 2.0),
        "e": frenet.e,
        "dphi": frenet.dphi,
        "delta": state["delta"],
    }
    projected = {
        "seq": seq,
        "raw_state": dict(state),
        "current_state": current_state,
        "sim_time": float(state["t"]),
        "s": float(frenet.s),
        "e": float(frenet.e),
        "dphi": float(frenet.dphi),
        "x_ref": float(frenet.x_ref),
        "y_ref": float(frenet.y_ref),
        "yaw_ref": float(frenet.psi_ref),
        "proj_dist": float(frenet.proj_dist),
    }
    return projected, frenet.s
    

def compute_low_level_command(
	state: dict[str, float],
	projected: dict[str, object],
	plan: dict[str, object],
	integ_e_wr: float,
	dt: float,
) -> tuple[dict[str, float], float]:
	state_now = projected["current_state"]
	s_now = float(projected["s"])
    # This functioon finds where you are in the plan and returns the target values at that point.
	plan_sample = sample_plan(plan, s_now + 0.5) # Exlain why you need an offset here.

	roadwheel_target = float(plan_sample["roadwheel_angle"])
	rear_wheelspeed_target = float(plan_sample["rear_wheelspeed_ms"])
	rear_wheel_torque_target = float(plan_sample["rear_wheel_torque"])

	# Convert the planned road-wheel angle into the normalized BeamNG steering command.
	steering_cmd = roadwheel_target / STEER_TO_ROADWHEEL_ANGLE

	# Feedforward throttle from the inverse-throttle surrogate.
	# Arguments: engine speed [rad/s], boost pressure, rear wheel speed [m/s], desired rear-wheel torque [Nm].
	ff_throttle = float(
		inv_throttle_fn(
			state["we"],
			state["pb"],
			state_now["wr"],
			rear_wheel_torque_target,
		)
	)

	# Wheel-speed tracking error for your PI feedback term.
	wr_error = rear_wheelspeed_target - float(state_now["wr"])

	# TODO: choose your own Kp and Ki.
	# TODO: update integ_e_wr using dt and clip it if you want anti-windup.
	# TODO: define fb_throttle from wr_error and integ_e_wr.
	# Example intent only:
	Kp = 0.5
	Ki = 1.0
	integ_e_wr = integ_e_wr + wr_error * dt
	fb_throttle = Kp * (wr_error) + Ki * integ_e_wr

	# Start from the feedforward term, then add your feedback correction.
	throttle_cmd = ff_throttle
	throttle_cmd = ff_throttle + fb_throttle
	throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

	command = {
		"throttle": throttle_cmd,
		"steering": steering_cmd,
		"roadwheel_angle": roadwheel_target,
		"rear_wheel_torque": rear_wheel_torque_target,
		"rear_wheelspeed_target": rear_wheelspeed_target,
	}
	return command, integ_e_wr


def actuation_worker() -> None:
	last_seq = 0
	last_s = None
	init_time = None
	last_sim_time = None
	integ_e_wr = 0.0

	while not runtime.should_stop():
		item = runtime.wait_for_raw_state(last_seq)
		if item is None:
			break
		seq, state = item
		last_seq = seq

		if init_time is None:
			init_time = float(state["t"])

		projected, last_s = build_projected_state(tracking_ref, seq, state, last_s, init_time)
		runtime.publish_projected_state(projected)

		if projected["s"] >= s_stop:
			runtime.stop()
			break

		plan = runtime.get_plan()
		current_sim_time = float(state["t"])
		dt = 0.0 if last_sim_time is None else float(np.clip(current_sim_time - last_sim_time, 0.0, 0.1))
		last_sim_time = current_sim_time

		if plan is None:
			continue

		command, integ_e_wr = compute_low_level_command(state, projected, plan, integ_e_wr, dt)
		ctl.send_command(
			throttle=command["throttle"],
			brake=0.0,
			steering=command["steering"],
		)
		runtime.set_applied_control(command)
	ctl.send_command(
		throttle=0.0,
		brake=0.0,
		steering=0.0,
	)

# def mpc_worker() -> None:
# 	"""Fill this in next. It should wait for projected state, solve MPC, and store the plan."""
# 	raise NotImplementedError("Implement the Week 12 MPC worker next")


def run_worker(name: str, worker) -> None:
	try:
		worker()
	except Exception as exc:
		runtime.record_error(name, exc)



# Same iplementation as in Week 11, but now the state comes from the live projected stream instead of the raw state stream.
def solve_tracking_problem(
	mpc: FrenetTrackingMPC,
	reference: TrackingReference,
	projected: dict[str, object],
	applied_control: dict[str, float],
) -> tuple[dict[str, object], float]:
	# Build the preview window starting at the current Frenet position.
	ref_window = reference.get_ref_traj(
		s_start=float(projected["s"]),
		horizon_steps=mpc.cfg.horizon_steps,
		ds=float(mpc.cfg.prediction_ds or 1.0),
	)

	# Solve one receding-horizon problem from the current Frenet state.
	solve_start = time.perf_counter()
	solution = mpc.solve(
		projected["current_state"],
		ref_window,
		prev_control=applied_control,
	)
	solve_time = time.perf_counter() - solve_start
	return solution, solve_time


def mpc_worker() -> None:
	last_seq = 0
	solve_idx = 0

	while not runtime.should_stop():
		projected = runtime.wait_for_projected_state(last_seq)
		if projected is None:
			break
		last_seq = int(projected["seq"])

		if projected["s"] >= s_stop:
			runtime.stop()
			break

		applied_control = runtime.get_applied_control()
		solution, solve_time = solve_tracking_problem(mpc, tracking_ref, projected, applied_control)

		# Convert the raw MPC output into the plan format used by sample_plan(...)
		# and by the actuation worker.
		plan = build_plan(tracking_ref, projected, solution, applied_control, solve_time)
		runtime.set_plan(plan)
		solve_log.append(plan["record"])

		# Keep the live visualizer in sync with the newest planned rollout.
		publish_live_visualizer(visualizer, plan["record"])

		print(
			f"solve={solve_idx:03d} "
			f"s={projected['s']:7.2f} "
			f"e={projected['e']: .3f} "
			f"dphi={projected['dphi']: .3f} "
			f"V={projected['raw_state']['V']: .2f} "
			f"solve_t={solve_time * 1e3:6.1f}ms"
		)
		solve_idx += 1

actuation_thread = threading.Thread(target=lambda: run_worker("actuation", actuation_worker), daemon=True)
mpc_thread = threading.Thread(target=lambda: run_worker("mpc", mpc_worker), daemon=True)

actuation_thread.start()
mpc_thread.start()


# while last_s is None or last_s < s_stop:

# 	actuation_thread = threading.Thread(target=lambda: run_worker("actuation", actuation_worker), daemon=True)
# 	mpc_thread = threading.Thread(target=lambda: run_worker("mpc", mpc_worker), daemon=True)

# 	actuation_thread.start()
# 	mpc_thread.start()	

# 	try:
# 		while not runtime.should_stop():
# 			if visualizer is not None and not visualizer["process"].is_alive():
# 				runtime.stop()
# 				break
# 			time.sleep(0.05)
# 	except KeyboardInterrupt:
# 		runtime.stop()
# 	finally:
# 		actuation_thread.join(timeout=2.0)
# 		mpc_thread.join(timeout=2.0)
# 		ctl.remove_state_listener(on_state_msg)
# 		ctl.clear_targets()
# 		ctl.close()
# 		if ros_logger is not None:
# 			ros_logger.destroy_node()

# 	thread_error = runtime.get_first_error()
# 	visualizer_alive = visualizer is not None and visualizer["process"].is_alive()

# 	if thread_error is not None and visualizer_alive:
# 		name, exc = thread_error
# 		print(f"{name} thread failed: {exc}")
# 		print("controller stopped; close the Qt visualizer to exit")
# 		try:
# 			while visualizer["process"].is_alive():
# 				time.sleep(0.1)
# 		except KeyboardInterrupt:
# 			stop_live_visualizer(visualizer)
# 	elif visualizer_alive:
# 		print("simulation finished; close the Qt visualizer to exit")
# 		try:
# 			while visualizer["process"].is_alive():
# 				time.sleep(0.1)
# 		except KeyboardInterrupt:
# 			stop_live_visualizer(visualizer)

# 	if thread_error is not None:
# 		name, exc = thread_error
# 		raise RuntimeError(f"{name} thread failed") from exc

# 	plt.show()
# 	step_count = step_count + 1