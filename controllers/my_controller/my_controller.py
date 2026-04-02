import base64
import math
import random
import tempfile
import threading
import time
import traceback
import sys

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

from controller import Supervisor

_root_dir = Path(__file__).resolve().parents[2]
_src_dir = _root_dir / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from api.http_api import ApiCommand, CommandDispatcher, start_api_server_in_thread


TIME_STEP = 16

MAX_WHEEL_SPEED = 3.0
WHEELS_DISTANCE = 0.4492
SUB_WHEELS_DISTANCE = 0.098
WHEEL_RADIUS = 0.08

TOLERANCE = 0.05
DRIVE_DISTANCE = 0.25


(
    FLL_WHEEL,
    FLR_WHEEL,
    FRL_WHEEL,
    FRR_WHEEL,
    BLL_WHEEL,
    BLR_WHEEL,
    BRL_WHEEL,
    BRR_WHEEL,
) = range(8)
FL_ROTATION, FR_ROTATION, BL_ROTATION, BR_ROTATION = range(4)
SHOULDER_ROLL, SHOULDER_LIFT, UPPER_ARM_ROLL, ELBOW_LIFT, WRIST_ROLL = range(5)
LEFT_FINGER, RIGHT_FINGER = range(2)


def almost_equal(a: float, b: float) -> bool:
    return (a < b + TOLERANCE) and (a > b - TOLERANCE)


def rotate_vector_axis_angle(vector: List[float], rotation: List[float]) -> List[float]:
    if len(vector) != 3 or len(rotation) != 4:
        return (
            [float(vector[0]), float(vector[1]), float(vector[2])]
            if len(vector) == 3
            else [0.0, 0.0, 0.0]
        )
    axis_x = float(rotation[0])
    axis_y = float(rotation[1])
    axis_z = float(rotation[2])
    angle = float(rotation[3])

    axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    if axis_norm < 1e-12:
        return [float(vector[0]), float(vector[1]), float(vector[2])]

    kx = axis_x / axis_norm
    ky = axis_y / axis_norm
    kz = axis_z / axis_norm

    vx = float(vector[0])
    vy = float(vector[1])
    vz = float(vector[2])

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    cross_x = ky * vz - kz * vy
    cross_y = kz * vx - kx * vz
    cross_z = kx * vy - ky * vx

    dot = kx * vx + ky * vy + kz * vz

    rx = vx * cos_a + cross_x * sin_a + kx * dot * (1.0 - cos_a)
    ry = vy * cos_a + cross_y * sin_a + ky * dot * (1.0 - cos_a)
    rz = vz * cos_a + cross_z * sin_a + kz * dot * (1.0 - cos_a)

    return [float(rx), float(ry), float(rz)]


class Pr2ApiController:
    def __init__(self) -> None:
        self.robot = Supervisor()

        self._self_node: Optional[object] = self.robot.getSelf()
        self._self_translation_field: Optional[object] = None
        self._self_rotation_field: Optional[object] = None

        if self._self_node is not None:
            self._self_translation_field = self._self_node.getField("translation")
            self._self_rotation_field = self._self_node.getField("rotation")

        self._start_robot_translation: Optional[List[float]] = None
        self._start_robot_z: float = 0.0
        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._pose_yaw: float = 0.0

        self._start_scene_nodes: List[Dict[str, Any]] = []
        for def_name in ["TABLE_X_POS", "TABLE_X_NEG", "TABLE_Y_POS", "TABLE_Y_NEG"]:
            node = self.robot.getFromDef(def_name)
            if node is None:
                continue
            translation_field = node.getField("translation")
            rotation_field = node.getField("rotation")
            self._start_scene_nodes.append(
                {
                    "node": node,
                    "translation_field": translation_field,
                    "rotation_field": rotation_field,
                    "translation": list(translation_field.getSFVec3f()),
                    "rotation": list(rotation_field.getSFRotation()),
                }
            )

        self._api_dispatcher: Optional[CommandDispatcher] = None
        self._api_thread: Optional[threading.Thread] = None
        self._api_start_error: Optional[str] = None

        self._viewpoint_node: Optional[object] = self.robot.getFromDef("VIEWPOINT")
        self._viewpoint_position_field: Optional[object] = None
        self._viewpoint_orientation_field: Optional[object] = None
        self._viewpoint_follow_field: Optional[object] = None
        self._viewpoint_fov_field: Optional[object] = None
        if self._viewpoint_node is not None:
            self._viewpoint_position_field = self._viewpoint_node.getField("position")
            self._viewpoint_orientation_field = self._viewpoint_node.getField(
                "orientation"
            )
            self._viewpoint_follow_field = self._viewpoint_node.getField("follow")
            self._viewpoint_fov_field = self._viewpoint_node.getField("fieldOfView")

        self.wheel_motors: List[object] = [None] * 8
        self.wheel_sensors: List[object] = [None] * 8
        self.rotation_motors: List[object] = [None] * 4
        self.rotation_sensors: List[object] = [None] * 4
        self.left_arm_motors: List[object] = [None] * 5
        self.left_arm_sensors: List[object] = [None] * 5
        self.right_arm_motors: List[object] = [None] * 5
        self.right_arm_sensors: List[object] = [None] * 5

        self.right_finger_motor: Optional[object] = None
        self.right_finger_sensor: Optional[object] = None
        self.left_finger_motor: Optional[object] = None
        self.left_finger_sensor: Optional[object] = None
        self.head_tilt_motor: Optional[object] = None
        self.torso_motor: Optional[object] = None
        self.torso_sensor: Optional[object] = None

        self.left_finger_contact_sensors: List[object] = [None] * 2
        self.right_finger_contact_sensors: List[object] = [None] * 2
        self.imu_sensor: Optional[object] = None
        self.wide_stereo_l_stereo_camera_sensor: Optional[object] = None
        self.wide_stereo_r_stereo_camera_sensor: Optional[object] = None
        self.high_def_sensor: Optional[object] = None
        self.r_forearm_cam_sensor: Optional[object] = None
        self.l_forearm_cam_sensor: Optional[object] = None
        self.laser_tilt: Optional[object] = None
        self.base_laser: Optional[object] = None

        self._wheel_torques: List[float] = [0.0] * 8
        self._gripper_first_call = True
        self._gripper_max_torque = 0.0

        self._holding_right: bool = False

        self._can_translation_fields: List[object] = []

        self._right_arm_stowed_position: List[float] = [0.0, 1.35, 0.0, -2.2, 0.0]
        self._right_arm_preextended_position: List[float] = [0.0, 0.5, 0.0, -0.5, 0.0]
        self._right_arm_extended_position: List[float] = [0.0, 0.5, 0.0, -0.5, 0.0]

        self._can_positions: List[List[float]] = [
            [1.13, -0.085, 0.801],
            [0.20, 1.255, 0.801],
            [-0.17, -1.03, 0.801],
            [-1.15, 0.305, 0.801],
        ]

        for def_name in ["CAN"]:
            can_node = self.robot.getFromDef(def_name)
            if can_node is None:
                continue
            translation_field = can_node.getField("translation")
            if translation_field is not None:
                self._can_translation_fields.append(translation_field)

    def step(self) -> None:
        if self.robot.step(TIME_STEP) == -1:
            raise SystemExit(0)

    def randomize_can_positions(self) -> None:
        if not self._can_translation_fields:
            return
        field = self._can_translation_fields[0]
        field.setSFVec3f(random.choice(self._can_positions))

    def wait_seconds(self, seconds: float) -> None:
        end_time = float(self.robot.getTime()) + seconds
        while float(self.robot.getTime()) < end_time:
            self.step()

    def initialize_devices(self) -> None:
        r = self.robot

        self.wheel_motors[FLL_WHEEL] = r.getDevice("fl_caster_l_wheel_joint")
        self.wheel_motors[FLR_WHEEL] = r.getDevice("fl_caster_r_wheel_joint")
        self.wheel_motors[FRL_WHEEL] = r.getDevice("fr_caster_l_wheel_joint")
        self.wheel_motors[FRR_WHEEL] = r.getDevice("fr_caster_r_wheel_joint")
        self.wheel_motors[BLL_WHEEL] = r.getDevice("bl_caster_l_wheel_joint")
        self.wheel_motors[BLR_WHEEL] = r.getDevice("bl_caster_r_wheel_joint")
        self.wheel_motors[BRL_WHEEL] = r.getDevice("br_caster_l_wheel_joint")
        self.wheel_motors[BRR_WHEEL] = r.getDevice("br_caster_r_wheel_joint")
        for i in range(8):
            self.wheel_sensors[i] = self.wheel_motors[i].getPositionSensor()

        self.rotation_motors[FL_ROTATION] = r.getDevice("fl_caster_rotation_joint")
        self.rotation_motors[FR_ROTATION] = r.getDevice("fr_caster_rotation_joint")
        self.rotation_motors[BL_ROTATION] = r.getDevice("bl_caster_rotation_joint")
        self.rotation_motors[BR_ROTATION] = r.getDevice("br_caster_rotation_joint")
        for i in range(4):
            self.rotation_sensors[i] = self.rotation_motors[i].getPositionSensor()

        self.left_arm_motors[SHOULDER_ROLL] = r.getDevice("l_shoulder_pan_joint")
        self.left_arm_motors[SHOULDER_LIFT] = r.getDevice("l_shoulder_lift_joint")
        self.left_arm_motors[UPPER_ARM_ROLL] = r.getDevice("l_upper_arm_roll_joint")
        self.left_arm_motors[ELBOW_LIFT] = r.getDevice("l_elbow_flex_joint")
        self.left_arm_motors[WRIST_ROLL] = r.getDevice("l_wrist_roll_joint")
        for i in range(5):
            self.left_arm_sensors[i] = self.left_arm_motors[i].getPositionSensor()

        self.right_arm_motors[SHOULDER_ROLL] = r.getDevice("r_shoulder_pan_joint")
        self.right_arm_motors[SHOULDER_LIFT] = r.getDevice("r_shoulder_lift_joint")
        self.right_arm_motors[UPPER_ARM_ROLL] = r.getDevice("r_upper_arm_roll_joint")
        self.right_arm_motors[ELBOW_LIFT] = r.getDevice("r_elbow_flex_joint")
        self.right_arm_motors[WRIST_ROLL] = r.getDevice("r_wrist_roll_joint")
        for i in range(5):
            self.right_arm_sensors[i] = self.right_arm_motors[i].getPositionSensor()

        self.left_finger_motor = r.getDevice("l_finger_gripper_motor::l_finger")
        self.left_finger_sensor = self.left_finger_motor.getPositionSensor()

        self.right_finger_motor = r.getDevice("r_finger_gripper_motor::l_finger")
        self.right_finger_sensor = self.right_finger_motor.getPositionSensor()

        self.head_tilt_motor = r.getDevice("head_tilt_joint")
        self.torso_motor = r.getDevice("torso_lift_joint")
        self.torso_sensor = r.getDevice("torso_lift_joint_sensor")

        self.left_finger_contact_sensors[LEFT_FINGER] = r.getDevice(
            "l_gripper_l_finger_tip_contact_sensor"
        )
        self.left_finger_contact_sensors[RIGHT_FINGER] = r.getDevice(
            "l_gripper_r_finger_tip_contact_sensor"
        )
        self.right_finger_contact_sensors[LEFT_FINGER] = r.getDevice(
            "r_gripper_l_finger_tip_contact_sensor"
        )
        self.right_finger_contact_sensors[RIGHT_FINGER] = r.getDevice(
            "r_gripper_r_finger_tip_contact_sensor"
        )

        self.imu_sensor = r.getDevice("imu_sensor")
        self.wide_stereo_l_stereo_camera_sensor = r.getDevice(
            "wide_stereo_l_stereo_camera_sensor"
        )
        self.wide_stereo_r_stereo_camera_sensor = r.getDevice(
            "wide_stereo_r_stereo_camera_sensor"
        )
        self.high_def_sensor = r.getDevice("high_def_sensor")
        self.r_forearm_cam_sensor = r.getDevice("r_forearm_cam_sensor")
        self.l_forearm_cam_sensor = r.getDevice("l_forearm_cam_sensor")
        self.laser_tilt = r.getDevice("laser_tilt")
        self.base_laser = r.getDevice("base_laser")

    def enable_devices(self) -> None:
        for i in range(8):
            self.wheel_sensors[i].enable(TIME_STEP)
            self.wheel_motors[i].setPosition(float("inf"))
            self.wheel_motors[i].setVelocity(0.0)

        for i in range(4):
            self.rotation_sensors[i].enable(TIME_STEP)

        for i in range(2):
            self.left_finger_contact_sensors[i].enable(TIME_STEP)
            self.right_finger_contact_sensors[i].enable(TIME_STEP)

        for _ in range(4):
            self.left_finger_sensor.enable(TIME_STEP)
            self.right_finger_sensor.enable(TIME_STEP)

        for i in range(5):
            self.left_arm_sensors[i].enable(TIME_STEP)
            self.right_arm_sensors[i].enable(TIME_STEP)

        self.torso_sensor.enable(TIME_STEP)

        if self.high_def_sensor is not None:
            self.high_def_sensor.enable(TIME_STEP)

    def capture_high_def_jpeg_base64(self, quality: int = 90) -> Dict[str, Any]:
        for _ in range(2):
            self.step()

        stamp = int(time.time() * 1000.0)

        debug: Dict[str, Any] = {
            "quality": int(quality),
            "stamp": int(stamp),
        }

        export_image = getattr(self.robot, "exportImage", None)
        debug["exportImage_callable"] = bool(callable(export_image))
        if callable(export_image):
            with tempfile.TemporaryDirectory(prefix="webots_capture_") as tmp_dir:
                target_png = Path(tmp_dir) / f"view_capture_{stamp}.png"

                previous_view: Optional[Dict[str, Any]] = None
                if (
                    self._viewpoint_position_field is not None
                    and self._viewpoint_orientation_field is not None
                ):
                    previous_view = {
                        "position": list(self._viewpoint_position_field.getSFVec3f()),
                        "orientation": list(
                            self._viewpoint_orientation_field.getSFRotation()
                        ),
                    }
                    if self._viewpoint_follow_field is not None:
                        previous_view["follow"] = str(
                            self._viewpoint_follow_field.getSFString()
                        )
                    if self._viewpoint_fov_field is not None:
                        previous_view["fieldOfView"] = float(
                            self._viewpoint_fov_field.getSFFloat()
                        )

                    self._viewpoint_position_field.setSFVec3f([0.0, 0.0, 12.0])
                    self._viewpoint_orientation_field.setSFRotation(
                        [-0.577, 0.577, 0.577, 2.09]
                    )
                    if self._viewpoint_fov_field is not None:
                        self._viewpoint_fov_field.setSFFloat(0.3)
                    if self._viewpoint_follow_field is not None:
                        self._viewpoint_follow_field.setSFString("")

                    for _ in range(8):
                        self.step()

                ok = False
                export_exc: Optional[str] = None
                try:
                    ok = bool(export_image(str(target_png), int(quality)))
                except Exception as exc:
                    export_exc = str(exc)
                finally:
                    if previous_view is not None:
                        self._viewpoint_position_field.setSFVec3f(
                            list(previous_view.get("position") or [0.0, 0.0, 0.0])
                        )
                        self._viewpoint_orientation_field.setSFRotation(
                            list(
                                previous_view.get("orientation") or [0.0, 1.0, 0.0, 0.0]
                            )
                        )
                        if self._viewpoint_follow_field is not None:
                            follow_value = previous_view.get("follow")
                            if follow_value is not None:
                                self._viewpoint_follow_field.setSFString(
                                    str(follow_value)
                                )
                        if self._viewpoint_fov_field is not None:
                            fov_value = previous_view.get("fieldOfView")
                            if fov_value is not None:
                                self._viewpoint_fov_field.setSFFloat(float(fov_value))
                        for _ in range(2):
                            self.step()

                debug["exportImage_return"] = bool(ok)
                if export_exc is not None:
                    debug["exportImage_exception"] = export_exc

                if target_png.exists() and target_png.stat().st_size > 0:
                    data = target_png.read_bytes()
                    image_b64 = base64.b64encode(data).decode("ascii")
                    return {
                        "mime_type": "image/png",
                        "image_b64": image_b64,
                        "sim_time_s": float(self.robot.getTime()),
                    }

        debug["high_def_sensor_available"] = self.high_def_sensor is not None
        if self.high_def_sensor is None:
            raise RuntimeError(str({"reason": "exportImage_failed", **debug}))

        with tempfile.TemporaryDirectory(prefix="webots_capture_") as tmp_dir:
            target_jpg = Path(tmp_dir) / f"high_def_capture_{stamp}.jpg"
            ok = bool(self.high_def_sensor.saveImage(str(target_jpg), int(quality)))
            debug["highDef_return"] = bool(ok)
            debug["highDef_exists"] = bool(target_jpg.exists())
            if (not ok and not target_jpg.exists()) or (
                target_jpg.exists() and target_jpg.stat().st_size == 0
            ):
                raise RuntimeError(str({"reason": "highDef_saveImage_failed", **debug}))

            data = target_jpg.read_bytes()
            image_b64 = base64.b64encode(data).decode("ascii")
            return {
                "mime_type": "image/jpeg",
                "image_b64": image_b64,
                "sim_time_s": float(self.robot.getTime()),
            }

    def set_wheels_speeds(
        self,
        fll: float,
        flr: float,
        frl: float,
        frr: float,
        bll: float,
        blr: float,
        brl: float,
        brr: float,
    ) -> None:
        self.wheel_motors[FLL_WHEEL].setVelocity(fll)
        self.wheel_motors[FLR_WHEEL].setVelocity(flr)
        self.wheel_motors[FRL_WHEEL].setVelocity(frl)
        self.wheel_motors[FRR_WHEEL].setVelocity(frr)
        self.wheel_motors[BLL_WHEEL].setVelocity(bll)
        self.wheel_motors[BLR_WHEEL].setVelocity(blr)
        self.wheel_motors[BRL_WHEEL].setVelocity(brl)
        self.wheel_motors[BRR_WHEEL].setVelocity(brr)

    def set_wheels_speed(self, speed: float) -> None:
        self.set_wheels_speeds(speed, speed, speed, speed, speed, speed, speed, speed)

    def stop_wheels(self) -> None:
        self.set_wheels_speeds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def enable_passive_wheels(self, enable: bool) -> None:
        if enable:
            for i in range(8):
                self._wheel_torques[i] = float(
                    self.wheel_motors[i].getAvailableTorque()
                )
                self.wheel_motors[i].setAvailableTorque(0.0)
        else:
            for i in range(8):
                self.wheel_motors[i].setAvailableTorque(self._wheel_torques[i])

    def set_rotation_wheels_angles(
        self, fl: float, fr: float, bl: float, br: float, wait_on_feedback: bool
    ) -> None:
        if wait_on_feedback:
            self.stop_wheels()
            self.enable_passive_wheels(True)

        self.rotation_motors[FL_ROTATION].setPosition(fl)
        self.rotation_motors[FR_ROTATION].setPosition(fr)
        self.rotation_motors[BL_ROTATION].setPosition(bl)
        self.rotation_motors[BR_ROTATION].setPosition(br)

        if wait_on_feedback:
            target = [fl, fr, bl, br]
            while True:
                all_reached = True
                for i in range(4):
                    current_position = float(self.rotation_sensors[i].getValue())
                    if not almost_equal(current_position, target[i]):
                        all_reached = False
                        break
                if all_reached:
                    break
                self.step()
            self.enable_passive_wheels(False)

    def robot_rotate(self, angle: float) -> None:
        requested_angle = float(angle)
        if abs(requested_angle) < 1e-6:
            return

        start_yaw = self._read_current_yaw()
        if start_yaw is None:
            return
        target_yaw = self._normalize_angle(start_yaw + requested_angle)

        self.set_rotation_wheels_angles(
            -(math.pi / 4.0),
            math.pi / 4.0,
            math.pi / 4.0,
            -(math.pi / 4.0),
            True,
        )

        direction = 1.0 if requested_angle > 0.0 else -1.0
        max_speed = 0.6 * MAX_WHEEL_SPEED
        min_speed = 0.2

        start_time = float(self.robot.getTime())
        timeout = max(2.0, abs(requested_angle) / 0.2)

        while True:
            self.step()
            current_yaw = self._read_current_yaw()
            if current_yaw is None:
                break
            remaining = self._normalize_angle(target_yaw - current_yaw)
            if abs(remaining) <= 0.02:
                break
            if float(self.robot.getTime()) - start_time > timeout:
                break

            scale = min(1.0, max(0.0, abs(remaining) / 0.6))
            speed = max(min_speed, max_speed * scale)
            left_speed = -direction * speed
            right_speed = direction * speed
            self.set_wheels_speeds(
                left_speed,
                left_speed,
                right_speed,
                right_speed,
                left_speed,
                left_speed,
                right_speed,
                right_speed,
            )

        self.stop_wheels()
        self.set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, False)

    def _snap_yaw_to_right_angle(self, yaw: float) -> float:
        right_angle = math.pi / 2.0
        snapped = round(float(yaw) / right_angle) * right_angle
        return self._normalize_angle(float(snapped))

    def robot_rotate_quantized_right_angle(self, angle: float) -> None:
        requested_angle = float(angle)
        if abs(requested_angle) < 1e-6:
            return

        start_yaw = self._read_current_yaw()
        if start_yaw is None:
            return

        snapped_start = self._snap_yaw_to_right_angle(float(start_yaw))
        target_yaw = self._normalize_angle(snapped_start + requested_angle)

        self.set_rotation_wheels_angles(
            -(math.pi / 4.0),
            math.pi / 4.0,
            math.pi / 4.0,
            -(math.pi / 4.0),
            True,
        )

        direction = 1.0 if requested_angle > 0.0 else -1.0
        max_speed = 0.6 * MAX_WHEEL_SPEED
        min_speed = 0.2

        start_time = float(self.robot.getTime())
        timeout = max(2.0, abs(requested_angle) / 0.2)

        while True:
            self.step()
            current_yaw = self._read_current_yaw()
            if current_yaw is None:
                break
            remaining = self._normalize_angle(target_yaw - float(current_yaw))
            if abs(remaining) <= 0.02:
                break
            if float(self.robot.getTime()) - start_time > timeout:
                break

            scale = min(1.0, max(0.0, abs(remaining) / 0.6))
            speed = max(min_speed, max_speed * scale)
            left_speed = -direction * speed
            right_speed = direction * speed
            self.set_wheels_speeds(
                left_speed,
                left_speed,
                right_speed,
                right_speed,
                left_speed,
                left_speed,
                right_speed,
                right_speed,
            )

        self.stop_wheels()
        self.set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, False)

        if self._self_rotation_field is not None:
            self._self_rotation_field.setSFRotation([0.0, 0.0, 1.0, float(target_yaw)])
            if self._self_node is not None and not self._holding_right:
                self._self_node.resetPhysics()

    def robot_go_forward(self, distance: float) -> None:
        requested_distance = float(distance)
        if abs(requested_distance) < 1e-6:
            return

        self.set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, True)

        direction = 1.0 if requested_distance > 0.0 else -1.0
        speed = 0.6 * MAX_WHEEL_SPEED

        start_xy = self._read_current_xy()
        start_wheels = self._read_wheel_sensor_values()
        start_time = float(self.robot.getTime())
        target = abs(requested_distance)
        timeout = max(2.0, target / 0.05)

        self.set_wheels_speed(direction * speed)
        while True:
            self.step()
            travelled = self._estimate_travel_distance(start_xy, start_wheels)
            if travelled >= target:
                break
            if float(self.robot.getTime()) - start_time > timeout:
                break

        self.stop_wheels()

    def go_home(self) -> None:
        self._ensure_pose_initialized()
        if self._start_robot_translation is None:
            return

        current_xy = self._read_current_xy()
        current_yaw = self._read_current_yaw()
        if current_xy is None or current_yaw is None:
            return

        target_x = float(self._start_robot_translation[0])
        target_y = float(self._start_robot_translation[1])
        dx = target_x - float(current_xy[0])
        dy = target_y - float(current_xy[1])
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 0.02:
            delta_yaw = self._normalize_angle(
                float(self._pose_yaw) - float(current_yaw)
            )
            if abs(delta_yaw) > 0.02:
                self.robot_rotate(delta_yaw)
            return

        heading_to_start = math.atan2(dy, dx)
        heading_away = self._normalize_angle(heading_to_start + math.pi)
        delta_to_away = self._normalize_angle(heading_away - float(current_yaw))
        if abs(delta_to_away) > 0.02:
            self.robot_rotate(delta_to_away)

        self.robot_go_forward(-float(distance))

        new_yaw = self._read_current_yaw()
        if new_yaw is not None:
            delta_back = self._normalize_angle(float(self._pose_yaw) - float(new_yaw))
            if abs(delta_back) > 0.02:
                self.robot_rotate(delta_back)

    def move_forward(self, distance: float = DRIVE_DISTANCE) -> None:
        self.robot_go_forward(abs(distance))

    def move_backward(self, distance: float = DRIVE_DISTANCE) -> None:
        self.robot_go_forward(-abs(distance))

    def rotate_right_90(self) -> None:
        self.robot_rotate_quantized_right_angle(-(math.pi / 2.0))

    def rotate_left_90(self) -> None:
        self.robot_rotate_quantized_right_angle(math.pi / 2.0)

    def rotate_back(self) -> None:
        self.robot_rotate_quantized_right_angle(math.pi)

    def grab_right(self, torque_when_gripping: float = 30.0) -> None:
        self.set_right_arm_position(
            self._right_arm_preextended_position[0],
            self._right_arm_preextended_position[1],
            self._right_arm_preextended_position[2],
            self._right_arm_preextended_position[3],
            self._right_arm_preextended_position[4],
            True,
        )
        self.set_right_arm_position(
            self._right_arm_extended_position[0],
            self._right_arm_extended_position[1],
            self._right_arm_extended_position[2],
            self._right_arm_extended_position[3],
            self._right_arm_extended_position[4],
            True,
        )
        self.wait_seconds(0.2)
        self.set_gripper(False, False, torque_when_gripping, True)
        self._holding_right = True
        self.set_right_arm_position(
            self._right_arm_stowed_position[0],
            self._right_arm_stowed_position[1],
            self._right_arm_stowed_position[2],
            self._right_arm_stowed_position[3],
            self._right_arm_stowed_position[4],
            True,
        )

    def release_right(self) -> None:
        self.set_right_arm_position(
            self._right_arm_preextended_position[0],
            self._right_arm_preextended_position[1],
            self._right_arm_preextended_position[2],
            self._right_arm_preextended_position[3],
            self._right_arm_preextended_position[4],
            True,
        )
        self.set_right_arm_position(
            self._right_arm_extended_position[0],
            self._right_arm_extended_position[1],
            self._right_arm_extended_position[2],
            self._right_arm_extended_position[3],
            self._right_arm_extended_position[4],
            True,
        )
        self.wait_seconds(0.2)
        self.set_gripper(False, True, 0.0, True)
        self._holding_right = False
        self.set_right_arm_position(
            self._right_arm_stowed_position[0],
            self._right_arm_stowed_position[1],
            self._right_arm_stowed_position[2],
            self._right_arm_stowed_position[3],
            self._right_arm_stowed_position[4],
            True,
        )

    def set_gripper(
        self,
        left: bool,
        open_: bool,
        torque_when_gripping: float,
        wait_on_feedback: bool,
    ) -> None:
        motor = self.left_finger_motor if left else self.right_finger_motor
        finger_sensor = self.left_finger_sensor if left else self.right_finger_sensor

        contacts = [None, None]
        contacts[LEFT_FINGER] = (
            self.left_finger_contact_sensors[LEFT_FINGER]
            if left
            else self.right_finger_contact_sensors[LEFT_FINGER]
        )
        contacts[RIGHT_FINGER] = (
            self.left_finger_contact_sensors[RIGHT_FINGER]
            if left
            else self.right_finger_contact_sensors[RIGHT_FINGER]
        )

        if self._gripper_first_call:
            self._gripper_max_torque = float(motor.getAvailableTorque())
            self._gripper_first_call = False

        for _ in range(4):
            motor.setAvailableTorque(self._gripper_max_torque)

        if open_:
            target_open_value = 0.5
            for _ in range(4):
                motor.setPosition(target_open_value)

            if wait_on_feedback:
                while not almost_equal(
                    float(finger_sensor.getValue()), target_open_value
                ):
                    self.step()
        else:
            target_close_value = 0.0
            for _ in range(4):
                motor.setPosition(target_close_value)

            if wait_on_feedback:
                while (
                    float(contacts[LEFT_FINGER].getValue()) == 0.0
                    or float(contacts[RIGHT_FINGER].getValue()) == 0.0
                ) and not almost_equal(
                    float(finger_sensor.getValue()), target_close_value
                ):
                    self.step()

                current_position = float(finger_sensor.getValue())
                for _ in range(4):
                    motor.setAvailableTorque(torque_when_gripping)
                    motor.setPosition(max(0.0, 0.95 * current_position))

    def set_right_arm_position(
        self,
        shoulder_roll: float,
        shoulder_lift: float,
        upper_arm_roll: float,
        elbow_lift: float,
        wrist_roll: float,
        wait_on_feedback: bool,
    ) -> None:
        self.right_arm_motors[SHOULDER_ROLL].setPosition(shoulder_roll)
        self.right_arm_motors[SHOULDER_LIFT].setPosition(shoulder_lift)
        self.right_arm_motors[UPPER_ARM_ROLL].setPosition(upper_arm_roll)
        self.right_arm_motors[ELBOW_LIFT].setPosition(elbow_lift)
        self.right_arm_motors[WRIST_ROLL].setPosition(wrist_roll)

        if wait_on_feedback:
            while (
                not almost_equal(
                    float(self.right_arm_sensors[SHOULDER_ROLL].getValue()),
                    shoulder_roll,
                )
                or not almost_equal(
                    float(self.right_arm_sensors[SHOULDER_LIFT].getValue()),
                    shoulder_lift,
                )
                or not almost_equal(
                    float(self.right_arm_sensors[UPPER_ARM_ROLL].getValue()),
                    upper_arm_roll,
                )
                or not almost_equal(
                    float(self.right_arm_sensors[ELBOW_LIFT].getValue()), elbow_lift
                )
                or not almost_equal(
                    float(self.right_arm_sensors[WRIST_ROLL].getValue()), wrist_roll
                )
            ):
                self.step()

    def set_left_arm_position(
        self,
        shoulder_roll: float,
        shoulder_lift: float,
        upper_arm_roll: float,
        elbow_lift: float,
        wrist_roll: float,
        wait_on_feedback: bool,
    ) -> None:
        self.left_arm_motors[SHOULDER_ROLL].setPosition(shoulder_roll)
        self.left_arm_motors[SHOULDER_LIFT].setPosition(shoulder_lift)
        self.left_arm_motors[UPPER_ARM_ROLL].setPosition(upper_arm_roll)
        self.left_arm_motors[ELBOW_LIFT].setPosition(elbow_lift)
        self.left_arm_motors[WRIST_ROLL].setPosition(wrist_roll)

        if wait_on_feedback:
            while (
                not almost_equal(
                    float(self.left_arm_sensors[SHOULDER_ROLL].getValue()),
                    shoulder_roll,
                )
                or not almost_equal(
                    float(self.left_arm_sensors[SHOULDER_LIFT].getValue()),
                    shoulder_lift,
                )
                or not almost_equal(
                    float(self.left_arm_sensors[UPPER_ARM_ROLL].getValue()),
                    upper_arm_roll,
                )
                or not almost_equal(
                    float(self.left_arm_sensors[ELBOW_LIFT].getValue()), elbow_lift
                )
                or not almost_equal(
                    float(self.left_arm_sensors[WRIST_ROLL].getValue()), wrist_roll
                )
            ):
                self.step()

    def set_torso_height(self, height: float, wait_on_feedback: bool) -> None:
        self.torso_motor.setPosition(height)

        if wait_on_feedback:
            while not almost_equal(float(self.torso_sensor.getValue()), height):
                self.step()

    def set_initial_position(self) -> None:
        self.set_left_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, False)
        self.set_right_arm_position(
            self._right_arm_stowed_position[0],
            self._right_arm_stowed_position[1],
            self._right_arm_stowed_position[2],
            self._right_arm_stowed_position[3],
            self._right_arm_stowed_position[4],
            False,
        )

        self.set_gripper(False, True, 0.0, False)
        self.set_gripper(True, True, 0.0, False)

        self._holding_right = False

        self.set_torso_height(0.2, True)

    def start_api(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if self._api_dispatcher is not None:
            return
        self._api_dispatcher = CommandDispatcher()
        try:
            self._api_thread = start_api_server_in_thread(
                self._api_dispatcher, host, port
            )
        except BaseException as exc:
            self._api_start_error = str(exc)
            _ = exc
            raise

        time.sleep(0.2)
        if self._api_thread is None or not self._api_thread.is_alive():
            self._api_start_error = "api thread exited immediately"
            raise RuntimeError("API thread exited immediately")

        _ = host
        _ = port

    def _rotation_to_yaw(self, rotation: List[float]) -> float:
        if len(rotation) != 4:
            return 0.0
        axis_x = float(rotation[0])
        axis_y = float(rotation[1])
        axis_z = float(rotation[2])
        angle = float(rotation[3])

        axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
        if axis_norm < 1e-12:
            return 0.0

        kx = axis_x / axis_norm
        ky = axis_y / axis_norm
        kz = axis_z / axis_norm

        half = 0.5 * angle
        sin_half = math.sin(half)
        qw = math.cos(half)
        qx = kx * sin_half
        qy = ky * sin_half
        qz = kz * sin_half

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _read_current_xy(self) -> Optional[Tuple[float, float]]:
        if self._self_translation_field is None:
            return None
        translation = list(self._self_translation_field.getSFVec3f())
        if len(translation) != 3:
            return None
        return float(translation[0]), float(translation[1])

    def _read_current_yaw(self) -> Optional[float]:
        if self._self_rotation_field is None:
            return None
        rotation = list(self._self_rotation_field.getSFRotation())
        return float(self._rotation_to_yaw(rotation))

    def _read_wheel_sensor_values(self) -> List[float]:
        values: List[float] = []
        for i in range(8):
            sensor = self.wheel_sensors[i]
            values.append(float(sensor.getValue()))
        return values

    def _estimate_travel_distance(
        self, start_xy: Optional[Tuple[float, float]], start_wheels: List[float]
    ) -> float:
        current_xy = self._read_current_xy()
        if start_xy is not None and current_xy is not None:
            dx = float(current_xy[0]) - float(start_xy[0])
            dy = float(current_xy[1]) - float(start_xy[1])
            return math.sqrt(dx * dx + dy * dy)

        current_wheels = self._read_wheel_sensor_values()
        if len(current_wheels) != 8 or len(start_wheels) != 8:
            return 0.0
        total = 0.0
        for i in range(8):
            total += abs(float(current_wheels[i]) - float(start_wheels[i]))
        avg_rotation = total / 8.0
        return avg_rotation * WHEEL_RADIUS

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _ensure_pose_initialized(self) -> None:
        if self._start_robot_translation is not None:
            return
        if self._self_translation_field is None or self._self_rotation_field is None:
            self._start_robot_translation = [0.0, 0.0, 0.0]
            self._start_robot_z = 0.0
            self._pose_x = 0.0
            self._pose_y = 0.0
            self._pose_yaw = 0.0
            return

        start_translation = list(self._self_translation_field.getSFVec3f())
        start_rotation = list(self._self_rotation_field.getSFRotation())

        self._start_robot_translation = start_translation
        self._start_robot_z = float(start_translation[2])
        self._pose_x = float(start_translation[0])
        self._pose_y = float(start_translation[1])
        self._pose_yaw = self._rotation_to_yaw(start_rotation)

    def _snap_robot_pose(self) -> None:
        self._ensure_pose_initialized()
        if self._self_translation_field is None or self._self_rotation_field is None:
            return
        self.stop_wheels()
        self.set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, False)

        self._self_translation_field.setSFVec3f(
            [float(self._pose_x), float(self._pose_y), float(self._start_robot_z)]
        )
        self._self_rotation_field.setSFRotation([0.0, 0.0, 1.0, float(self._pose_yaw)])
        if self._self_node is not None:
            if not self._holding_right:
                self._self_node.resetPhysics()

    def _snap_scene_nodes_to_start(self) -> None:
        for entry in self._start_scene_nodes:
            translation_field = entry.get("translation_field")
            rotation_field = entry.get("rotation_field")
            translation = entry.get("translation")
            rotation = entry.get("rotation")

            if translation_field is not None and translation is not None:
                translation_field.setSFVec3f(
                    [
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2]),
                    ]
                )
            if rotation_field is not None and rotation is not None:
                rotation_field.setSFRotation(
                    [
                        float(rotation[0]),
                        float(rotation[1]),
                        float(rotation[2]),
                        float(rotation[3]),
                    ]
                )
            node = entry.get("node")
            if node is not None:
                node.resetPhysics()

    def _execute_api_command(
        self, name: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if name == "move_forward":
            self.move_forward(float(params.get("distance", DRIVE_DISTANCE)))
            return None
        if name == "move_backward":
            self.move_backward(float(params.get("distance", DRIVE_DISTANCE)))
            return None
        if name == "rotate_right_90":
            self.rotate_right_90()
            return None
        if name == "rotate_left_90":
            self.rotate_left_90()
            return None
        if name == "rotate_back":
            self.rotate_back()
            return None
        if name == "grab_right":
            self.grab_right(float(params.get("torque_when_gripping", 30.0)))
            return None
        if name == "release_right":
            self.release_right()
            return None
        if name == "go_home":
            self.go_home()
            return None
        if name == "capture_high_def":
            return self.capture_high_def_jpeg_base64()
        raise ValueError(f"Unknown command: {name}")

    def _process_one_api_command(self) -> None:
        if self._api_dispatcher is None:
            return
        command = self._api_dispatcher.get_nowait()
        if command is None:
            return
        self._finalize_api_command(command)

    def _finalize_api_command(self, command: ApiCommand) -> None:
        try:
            payload = self._execute_api_command(command.name, command.params)
            if payload is None:
                command.response = {"ok": True}
            else:
                command.response = {"ok": True, **payload}
        except SystemExit:
            command.response = {"ok": False, "error": "simulation terminated"}
            command.done.set()
            raise
        except BaseException as exc:
            command.response = {
                "ok": False,
                "command": str(command.name),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            command.done.set()

    def run(self) -> None:
        self.randomize_can_positions()
        self.initialize_devices()
        self.enable_devices()
        self.set_initial_position()
        self._snap_scene_nodes_to_start()

        self.start_api()

        while True:
            try:
                self.step()
            except SystemExit:
                return
            self._process_one_api_command()


def main() -> None:
    controller = Pr2ApiController()
    controller.run()


if __name__ == "__main__":
    main()
