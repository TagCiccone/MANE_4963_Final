from __future__ import annotations
from typing import Any, Dict
import numpy as np

fiala_params = {
    "vehicle": {
        "mass": 1224.87,
        "gravity": 9.81,
        "wheelbase": 2.653,
        "track_width": 1.863,
        "cogToFrontAxle": 1.508,
        "cogToRearAxle": 1.145,
        "cogHeight": 0.419,
        "inertia_zz": 2063.077,
        "center_of_mass_offset": {"x": 0.0, "y": 0.0},
        "roadwheel_angle_transform": {"scale": 1.0, "offset": 0.0},
        "wheel_speed_scale": {"front": 1.0, "rear": 1.0},
        "lateral_load_transfer_distribution": {"front": 1.0, "rear": 1.0},
        "Mzx_bias": 0.0,
        "aero_drag_coeff": [1.5343084335327148, 6.762434005737305],
    },
    "tires": {
        "fiala_pure_front": True,
        "mu_front": 1.084085285159017,
        "mu_rear": 1.0221728300844057,
        "Cf_front": 133589.60201937717,
        "Cf_rear": 171578.81080633533,
    },
    "wheel_dynamics": {
        "front_torque_enabled": False,
        "wheel_radius": 0.34,
        "front_wheel_inertia": 0.99,
        "rear_wheel_inertia": 0.99,
        "brake_rel_front": 6.6772685050964355,
        "brake_rel_rear": 8.44875717163086
    },
}

class FialaBicycleNumpy:

    SHORT_TO_VERBOSE = {
        "r": "yaw_rate",
        "vx": "vel_x",
        "vy": "vel_y",
        "wr": "rear_wheelspeed_ms",
        "delta": "roadwheel_angle",
    }

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.vehicle = params["vehicle"]
        self.tires = params["tires"]
        self.wheel = params["wheel_dynamics"]

    @staticmethod
    def _safe_positive(value, eps: float = 1e-4):
        return np.maximum(value, eps)

    @staticmethod
    def _clip(value, low: float, high: float):
        return np.clip(value, low, high)

    def _normalize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for short, verbose in self.SHORT_TO_VERBOSE.items():
            if short in data and verbose not in data:
                data[verbose] = data[short]
            if verbose in data and short not in data:
                data[short] = data[verbose]

        has_vx_vy = "vel_x" in data and "vel_y" in data
        has_v_beta = "V" in data and "beta" in data

        if has_v_beta and not has_vx_vy:
            data["vel_x"] = data["V"] * np.cos(data["beta"])
            data["vel_y"] = data["V"] * np.sin(data["beta"])
            data["vx"] = data["vel_x"]
            data["vy"] = data["vel_y"]
        elif has_vx_vy and not has_v_beta:
            data["V"] = np.sqrt(data["vel_x"] ** 2 + data["vel_y"] ** 2 + 1e-5)
            data["beta"] = np.arctan2(data["vel_y"], data["vel_x"])

        data.setdefault("brake", 0.0)
        return data

    def _powertrain_scaled_inputs(
        self,
        engine_speed_rads,
        boost_pressure,
        throttle,
        rear_wheelspeed_ms,
    ) -> Dict[str, Any]:
        wheel_radius = float(self.wheel["wheel_radius"])
        axle_speed_rads = rear_wheelspeed_ms / wheel_radius
        return {
            "engine_speed_rads": engine_speed_rads / 300.0,
            "boost_pressure": boost_pressure / 10.0,
            "throttle": throttle,
            "rear_wheelspeed_ms": rear_wheelspeed_ms / 5.0,
            "i_gr": 100.0 * (axle_speed_rads / (engine_speed_rads + 1e-4)),
            "gr": 0.01 * (engine_speed_rads / (axle_speed_rads + 1e-4)),
        }

    def _compute_wheel_torque_terms(self, state_control: Dict[str, Any]) -> Dict[str, Any]:
        engine_speed_rads = state_control["engine_speed_rads"]
        boost_pressure    = state_control["boost_pressure"]
        throttle          = state_control["throttle"]
        rear_wheelspeed_ms = state_control["rear_wheelspeed_ms"]

        d = self._powertrain_scaled_inputs(
            engine_speed_rads, boost_pressure, throttle, rear_wheelspeed_ms
        )
        eng   = d["engine_speed_rads"]
        boost = d["boost_pressure"]
        gr    = d["gr"]

        hidden_1 = np.tanh(((-0.299247 * eng) + (-0.0105762 * boost) + (2.09084  * throttle) + (-1.21916  * gr)) + 0.314798)
        hidden_2 = np.tanh(((-0.498726 * eng) + ( 0.0547674 * boost) + (-0.715504 * throttle) + ( 0.819941 * gr)) + 0.222718)
        hidden_3 = np.tanh(((-0.17094  * eng) + (-0.021485  * boost) + ( 0.0716762* throttle) + ( 3.68147  * gr)) + 0.0686006)
        hidden_4 = np.tanh(((  0.745817 * eng) + ( 1.42766  * boost) + (-1.60501  * throttle) + ( 0.00198479*gr)) + 0.389467)
        hidden_5 = np.tanh(((-0.212549 * eng) + (-1.00774  * boost) + ( 1.36391  * throttle) + ( 0.0194709* gr)) - 0.128168)
        hidden_6 = np.tanh(((  0.0431327* eng) + ( 0.806935 * boost) + (-0.641353 * throttle) + ( 0.350162 * gr)) - 0.0780202)

        h = [hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6]

        total_torque = 1000.0 * (
            ( 1.48922 * np.tanh(((-1.94056*h[0]) + (-0.0610694*h[1]) + ( 0.819076*h[2]) + ( 1.45404*h[3]) + (-0.054226 *h[4]) + ( 0.444599*h[5])) + 0.28967))
            +(-1.88444 * np.tanh(((-0.9117  *h[0]) + (-1.14981  *h[1]) + ( 2.47463 *h[2]) + (-1.55474*h[3]) + (-0.0627599*h[4]) + (-0.375705*h[5])) + 0.0779681))
            +(-0.461754* np.tanh(((-0.0409663*h[0])+ ( 0.145847 *h[1]) + ( 0.0101948*h[2])+ (-0.00553184*h[3])+(0.108535 *h[4]) + ( 0.110177*h[5])) + 0.0291548))
            +( 0.213776* np.tanh(((-0.149709 *h[0])+ (-0.423617 *h[1]) + ( 0.0647505*h[2])+ (-0.224323*h[3]) +(0.254653 *h[4]) + (-0.00174019*h[5]))- 0.0915727))
            +( 3.17494 * np.tanh((( 0.31846  *h[0])+ (-2.02533  *h[1]) + ( 2.54764  *h[2])+ (-2.3658  *h[3]) +(-0.378261*h[4]) + ( 0.280427*h[5])) - 0.173948))
            +( 0.888703 * np.tanh((( 1.31051  *h[0])+ (-2.8665   *h[1]) + (-0.153358 *h[2])+ (-1.22142 *h[3]) +( 1.16874 *h[4]) + (-0.546866*h[5])) + 0.36393))
            + 0.0936948
        )

        return {
            "data_scaled": d,
            "total_torque": total_torque,
            "rear_wheel_torque": total_torque,
            "front_wheel_torque": 0.0,
            "torque_dist_r": 1.0,
        }

    def estimate_inverse_throttle(self, state_control: Dict[str, Any], desired_torque) -> float:
        engine_speed_rads  = state_control["engine_speed_rads"]
        boost_pressure     = state_control["boost_pressure"]
        rear_wheelspeed_ms = state_control["rear_wheelspeed_ms"]

        wheel_radius           = float(self.wheel["wheel_radius"])
        engine_speed_scaled    = engine_speed_rads / 300.0
        boost_scaled           = boost_pressure / 10.0
        desired_torque_scaled  = desired_torque / 1000.0
        gear_ratio_scaled      = 0.01 * (engine_speed_rads / ((rear_wheelspeed_ms / wheel_radius) + 1e-4))

        relu = lambda x: np.maximum(x, 0.0)

        hidden_1 = relu(( 0.301603 * engine_speed_scaled) + (-0.170997 * boost_scaled) + (-0.157463 * desired_torque_scaled) + (-4.33797  * gear_ratio_scaled) + 0.143365)
        hidden_2 = relu((-0.0836236* engine_speed_scaled) + ( 0.129827 * boost_scaled) + (-0.141142 * desired_torque_scaled) + ( 6.88794  * gear_ratio_scaled) - 0.0446952)
        hidden_3 = relu((-0.0642931* engine_speed_scaled) + ( 0.241317 * boost_scaled) + ( 0.348286 * desired_torque_scaled) + (-0.241327 * gear_ratio_scaled) + 0.396857)
        hidden_4 = relu(( 0.166931 * engine_speed_scaled) + (-0.0917765* boost_scaled) + ( 0.618937 * desired_torque_scaled) + (-1.26335  * gear_ratio_scaled) + 0.046271)

        h = [hidden_1, hidden_2, hidden_3, hidden_4]

        return (
            (0.492427 * relu((-0.567724*h[0]) + (-1.39205*h[1]) + ( 0.410955*h[2]) + (-0.241824*h[3]) + 0.272251))
            +(0.841248 * relu((-0.338327*h[0]) + (-1.77441*h[1]) + (-0.427265*h[2]) + ( 0.553774*h[3]) - 0.0660385))
            +(0.202179 * relu((-0.815237*h[0]) + (-0.315648*h[1]) + ( 0.398362*h[2]) + ( 0.47149 *h[3]) + 0.231422))
            +(0.120461 * relu((-0.324339*h[0]) + (-0.34381 *h[1]) + (-0.129538*h[2]) + ( 0.487535*h[3]) + 0.0541269))
            - 0.000217977
        )

    def fiala_pure_lateral(self, alpha, c_alpha, mu, fz):
        f_max   = mu * fz
        tan_a   = np.tan(alpha)
        abs_tan_a = np.abs(tan_a)
        slip_limit = 3.0 * f_max / c_alpha
        fy = (
            -c_alpha * tan_a
            + (c_alpha**2 / (3.0 * f_max)) * abs_tan_a * tan_a
            - (c_alpha**3 / (27.0 * f_max**2)) * tan_a**3
        )
        return np.where(abs_tan_a < slip_limit, fy, -f_max * np.sign(alpha))

    def fiala_combined(self, alpha, kappa, c_alpha, mu, fz):
        tan_a       = np.tan(alpha)
        sigma_total = np.sqrt(tan_a**2 + kappa**2 + 1e-6)
        f_max       = mu * fz
        slip_limit  = 3.0 * f_max / c_alpha
        f_total = (
            c_alpha * sigma_total
            - (c_alpha**2 / (3.0 * f_max)) * sigma_total**2
            + (c_alpha**3 / (27.0 * f_max**2)) * sigma_total**3
        )
        f_total = np.where(sigma_total < slip_limit, f_total, f_max)
        fy = -f_total * (tan_a   / sigma_total)
        fx =  f_total * (kappa   / sigma_total)
        return fx, fy

    def get_tire_forces(self, alpha_f, alpha_r, kappa_x_front, kappa_x_rear, fzf, fzr):
        if self.tires.get("fiala_pure_front", True):
            fy_f = self.fiala_pure_lateral(
                alpha_f, self.tires["Cf_front"], self.tires["mu_front"], fzf
            )
            fx_f = 0.0
        else:
            fx_f, fy_f = self.fiala_combined(
                alpha_f, kappa_x_front, self.tires["Cf_front"], self.tires["mu_front"], fzf
            )
        fx_r, fy_r = self.fiala_combined(
            alpha_r, kappa_x_rear, self.tires["Cf_rear"], self.tires["mu_rear"], fzr
        )
        return fx_f, fy_f, fx_r, fy_r

    def get_vectorfield(self, state_control: Dict[str, Any]) -> Dict[str, Any]:
        data = self._normalize_input(state_control)

        mass    = self.vehicle["mass"]
        gravity = self.vehicle["gravity"]
        a       = self.vehicle["cogToFrontAxle"]
        b       = self.vehicle["cogToRearAxle"]
        length  = self.vehicle["wheelbase"]
        hcog    = self.vehicle["cogHeight"]
        inv_mass    = 1.0 / mass
        inv_inertia = 1.0 / self.vehicle["inertia_zz"]
        aero        = self.vehicle["aero_drag_coeff"]

        r     = data["yaw_rate"]
        vx    = data["vel_x"]
        vy    = data["vel_y"]
        delta = data["roadwheel_angle"]
        brake = data["brake"]
        delta_fx_hat = data.get("deltaFx_hat", data.get("accel_x", 0.0))

        # -- powertrain: compute rear wheel torque from engine inputs if provided --
        if "engine_speed_rads" in data and "throttle" in data:
            pt = self._compute_wheel_torque_terms(data)
            rear_wheel_torque = pt["rear_wheel_torque"]
        else:
            rear_wheel_torque = data.get("rear_wheel_torque", 0.0)

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        safe_vx = np.maximum(vx, 0.1)

        delta_fz = delta_fx_hat * (mass * hcog / length)
        fzf = np.maximum((b * mass * gravity) / length - delta_fz, 1.0)
        fzr = np.maximum((a * mass * gravity) / length + delta_fz, 1.0)

        vx_fw = vx * cos_d + (vy + a * r) * sin_d
        vy_fw = -vx * sin_d + (vy + a * r) * cos_d
        vx_rw = vx
        vy_rw = vy - b * r

        wf = data.get("front_wheelspeed_ms", vx_fw)
        wr = data["rear_wheelspeed_ms"]

        vx_fs = np.maximum(vx_fw, 0.1)
        vx_rs = np.maximum(vx_rw, 0.1)

        alpha_f = np.arctan2(vy + a * r, safe_vx) - delta
        alpha_r = np.arctan2(vy - b * r, safe_vx)

        tan_alpha_f = vy_fw / vx_fs
        tan_alpha_r = vy_rw / vx_rs
        kappa_x_front = (wf - vx_fw) / vx_fs
        kappa_x_rear  = (wr - vx_rw) / vx_rs

        fx_f, fy_f, fx_r, fy_r = self.get_tire_forces(
            alpha_f, alpha_r, kappa_x_front, kappa_x_rear, fzf, fzr
        )

        total_vel = np.sqrt(vx**2 + vy**2 + 1e-5)
        faero_x = -aero[0] * total_vel * vx
        faero_y = -aero[1] * total_vel * vy

        r_dot  = (a * fy_f * cos_d + a * fx_f * sin_d - b * fy_r) * inv_inertia
        vx_dot = (fx_f * cos_d - fy_f * sin_d + fx_r + faero_x) * inv_mass + r * vy
        vy_dot = (fx_f * sin_d + fy_f * cos_d + fy_r + faero_y) * inv_mass - r * vx

        wr_dot = (
            (self.wheel["wheel_radius"] / self.wheel["rear_wheel_inertia"]) * rear_wheel_torque
            - (self.wheel["wheel_radius"]**2 / self.wheel["rear_wheel_inertia"])
            * (fx_r + brake * self.wheel["brake_rel_rear"] * mass)
        )

        accel_x = vx_dot - r * vy
        accel_y = vy_dot + r * vx

        speed    = self._safe_positive(data["V"], 0.1)
        v_dot    = vx_dot * np.cos(data["beta"]) + vy_dot * np.sin(data["beta"])
        beta_dot = (vy_dot * np.cos(data["beta"]) - vx_dot * np.sin(data["beta"])) / speed

        tire_energy      = (np.abs(fy_r * vy_rw) + np.abs(fx_r * (wr - vx_rw))) / mass
        tire_saturation  = kappa_x_rear**2 + tan_alpha_r**2
        total_rear_force  = fx_r**2 + fy_r**2
        total_front_force = fx_f**2 + fy_f**2

        return {
            "Fxf": fx_f, "Fyf": fy_f, "Fxr": fx_r, "Fyr": fy_r,
            "Fzf": fzf,  "Fzr": fzr,  "deltaFz": delta_fz,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "tan_alpha_f": tan_alpha_f, "tan_alpha_r": tan_alpha_r,
            "kappa_x_front": kappa_x_front, "kappa_x_rear": kappa_x_rear,
            "vx_front_wheel": vx_fw, "vx_rear_wheel": vx_rw,
            "wr": wr,
            "yaw_rate_dot": r_dot,
            "vel_x_dot": vx_dot, "vel_y_dot": vy_dot,
            "rear_wheelspeed_ms_dot": wr_dot,
            "r_dot": r_dot, "vx_dot": vx_dot, "vy_dot": vy_dot,
            "accel_x": accel_x, "accel_y": accel_y,
            "wr_dot": wr_dot,
            "V_dot": v_dot, "beta_dot": beta_dot,
            "rear_wheel_torque": rear_wheel_torque,
            "tire_energy": tire_energy, "tire_saturation": tire_saturation,
            "total_rear_force": total_rear_force, "total_front_force": total_front_force,
        }

from scipy.integrate import solve_ivp

model = FialaBicycleNumpy(fiala_params)

def ode(t, state):
    vx, vy, r, wr, X, Y, psi = state

    control = {
        # "yaw_rate": r,
        # "vel_x": vx, 
        # "vel_y": vy, 
        "rear_wheelspeed_ms": wr,
        "roadwheel_angle": delta,
        # "brake": 0.0,

        # Option A — direct torque:
        # "rear_wheel_torque": 500.0,

        # Option B — engine inputs (overrides Option A if both provided):
        # "engine_speed_rads": 200.0,
        # "boost_pressure": 1.5,
        # "throttle": 0.6,
    }

    vf = model.get_vectorfield(control)

    X_dot   = vx * np.cos(psi) - vy * np.sin(psi)
    Y_dot   = vx * np.sin(psi) + vy * np.cos(psi)
    psi_dot = r

    return [vf["vx_dot"], vf["vy_dot"], vf["r_dot"], vf["wr_dot"], X_dot, Y_dot, psi_dot]


y0  = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
sol = solve_ivp(ode, t_span=[0, 10], y0=y0, method="RK45", max_step=0.01)