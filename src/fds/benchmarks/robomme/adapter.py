from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import numpy as np

from fds.common.paths import workspace_root
from fds.types import ActionSpec, Frame, Observation, StepResult

TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class RoboMME:
    splits = ("train", "val", "test")

    @staticmethod
    def task_language_conventions():
        """Static instruction coordinates; independent of task, episode and simulator state."""
        return {
            "source": "benchmark_instruction_coordinate_convention",
            "direction_reference": "robot viewpoint unless the instruction explicitly names an image/view",
            "robot_base_axes": {
                "forward": "+x (away from robot base)",
                "backward": "-x (toward robot base)",
                "left": "+y",
                "right": "-y",
                "up": "+z",
                "down": "-z",
            },
            "front_camera": "Faces the robot: robot-left appears image-right and robot-right appears image-left.",
            "pixels": "Image pixel x still increases to the right and y downward; never mirror pixel coordinates.",
        }

    def __init__(self, task="PickXtimes", split="val", max_steps=1300, calibration=False):
        if task not in TASKS or split not in self.splits:
            raise ValueError("Invalid RoboMME task or split")
        self.builder = self._make_builder(task, split, max_steps)
        self.calibration = calibration
        self.env = None
        self.goal = ""
        self.frame_index = 0
        self.stopped = False

    def _make_builder(self, task, split, max_steps):
        import robomme

        self.repo = workspace_root() / "third_party/robomme"
        spec = tomllib.loads((workspace_root() / "configs/dependencies.toml").read_text())["robomme"]
        self.revision = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if self.revision != spec["revision"]:
            raise RuntimeError("RoboMME source differs from configs/dependencies.toml")
        if not Path(robomme.__file__).resolve().is_relative_to(self.repo):
            raise RuntimeError(
                "RoboMME was imported outside FDS/third_party/robomme; use .venv-robomme"
            )
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        return BenchmarkEnvBuilder(
            env_id=task,
            dataset=split,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )

    def _make_action_spec(self):
        return ActionSpec(
            array(self.env.action_space.low).reshape(-1),
            array(self.env.action_space.high).reshape(-1),
        )

    def _record_info(self, info):
        """Optional benchmark accounting; never copy arbitrary info into observations."""

    def reset(self, episode: int) -> Observation:
        self.close()
        if not 0 <= episode < self.builder.get_episode_num():
            raise ValueError(f"Episode {episode} outside the official split")
        self.env = self.builder.make_env_for_episode(
            episode,
            include_front_camera_intrinsic=self.calibration,
            include_wrist_camera_intrinsic=self.calibration,
            include_front_camera_extrinsic=self.calibration,
            include_wrist_camera_extrinsic=self.calibration,
        )
        obs, info = self.env.reset()
        self._record_info(info)
        self.action_spec = self._make_action_spec()
        if self.action_spec.dim not in (7, 8):
            raise ValueError("Expected Panda absolute joint control (7 or 8 dimensions)")
        goal = info["task_goal"]
        self.goal = goal[0] if isinstance(goal, (list, tuple)) else str(goal)
        self.frame_index, self.stopped = 0, False
        self.intrinsics = {}
        if self.calibration:
            for camera in ("front", "wrist"):
                # RoboMME returns one 3x3 matrix, not a per-frame sequence.
                intrinsic = array(info[f"{camera}_camera_intrinsic"])
                if (
                    intrinsic.shape != (3, 3)
                    or not np.isfinite(intrinsic).all()
                    or intrinsic[0, 0] <= 0
                    or intrinsic[1, 1] <= 0
                    or not np.allclose(intrinsic[2], [0, 0, 1])
                ):
                    raise ValueError(
                        f"Invalid {camera} camera intrinsic matrix; expected finite 3x3 K"
                    )
                self.intrinsics[camera] = intrinsic.copy()
        return self._observation(obs, initial=True)

    def _observation(self, obs, initial=False) -> Observation:
        # Explicit sensor allowlist: no info dictionaries, masks, object identities,
        # task indices, simulator state, reward, or oracle subgoals cross this boundary.
        keys = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
        count = len(obs[keys[0]])
        if count == 0 or any(len(obs[k]) != count for k in keys):
            raise ValueError("Missing or misaligned RoboMME observation sequences")
        frames = []
        for i in range(count):
            joints = array(obs["joint_state_list"][i]).reshape(-1)
            fingers = array(obs["gripper_state_list"][i]).reshape(-1)
            if len(joints) != 7 or not len(fingers):
                raise ValueError("Invalid robot state")
            calibration = {}
            if self.calibration:
                for camera in ("front", "wrist"):
                    calibration[camera] = {
                        "K": self.intrinsics[camera].tolist(),
                        "world_to_camera": array(
                            obs[f"{camera}_camera_extrinsic_list"][i]
                        ).tolist(),
                    }
            frames.append(
                Frame(
                    index=self.frame_index,
                    images={
                        k: array(obs[f"{k}_rgb_list"][i]).astype(np.uint8)
                        for k in ("front", "wrist")
                    },
                    state=np.concatenate([joints, fingers[:1]]).astype(np.float32),
                    cue=initial and i < count - 1,
                    calibration=calibration,
                )
            )
            self.frame_index += 1
        return Observation(self.goal, frames)

    def step(self, action: np.ndarray) -> StepResult:
        if self.env is None or self.stopped:
            raise RuntimeError("Reset before stepping a missing or terminated environment")
        obs, _reward, terminated, truncated, info = self.env.step(self.action_spec.validate(action))
        self._record_info(info)
        status = str(info.get("status", "unknown"))
        result = StepResult(
            self._observation(obs) if obs is not None else None,
            bool(terminated),
            bool(truncated),
            status,
        )
        self.stopped = result.done
        return result

    def get_kinematics(self):
        """Robot structure only; no object state or privileged TCP pose is read."""
        from fds.common.kinematics import URDFKinematics

        if self.env is None:
            raise RuntimeError("Reset before requesting robot kinematics")
        agent = self.env.unwrapped.agent
        return URDFKinematics(agent.urdf_path, agent.arm_joint_names, agent.ee_link_name)

    def gripper_command(self, position):
        """Convert measured finger position using the public actuator configuration."""
        if self.env is None or self.action_spec.dim != 8:
            raise ValueError("No initialized gripper controller")
        controller = self.env.unwrapped.agent.controller.controllers["gripper"]
        config = controller.config
        if not np.isfinite(position):
            raise ValueError("Nonfinite gripper position")
        command = float(position)
        if config.normalize_action:
            if config.lower is None or config.upper is None:
                raise ValueError("Gripper controller must declare its physical target range")
            lower, upper = float(config.lower), float(config.upper)
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("Invalid gripper controller target range")
            command = 2 * (command - lower) / (upper - lower) - 1
        return float(np.clip(command, self.action_spec.low[7], self.action_spec.high[7]))

    def get_robot_geometry(self):
        """Public robot base transform, not object poses or simulator task state."""
        if self.env is None:
            raise RuntimeError("Reset before requesting robot geometry")
        agent = self.env.unwrapped.agent
        transform = array(agent.robot.pose.to_transformation_matrix()).reshape(-1, 4, 4)
        if len(transform) != 1 or not np.isfinite(transform).all():
            raise ValueError("Expected one finite robot base transform")
        return {
            "source": "own_robot_mount_and_URDF",
            "base_to_world": transform[0].tolist(),
            "tcp": self.get_kinematics().geometry(),
            "gripper": self.get_gripper_measurement_spec(),
            "camera_calibration_enabled": self.calibration,
            "note": "TCP is the URDF ee_link_name; finger joint travel is not object contact. "
            "World points = base_to_world @ base points. No object depth or object poses supplied.",
        }

    def get_gripper_measurement_spec(self):
        """Identify the public finger coordinate supplied by DemonstrationWrapper.

        The wrapper emits qpos[7:9], and this adapter uses its first entry.
        Verify that layout before attaching URDF limits; stick robots have no
        measured gripper channel. No contact or object state is queried.
        """
        if self.env is None:
            raise RuntimeError("Reset before requesting gripper measurement structure")
        if self.action_spec.dim != 8:
            return None
        from fds.common.kinematics import prismatic_joint_spec

        agent = self.env.unwrapped.agent
        fingers = list(agent.gripper_joint_names)
        names = [joint.name for joint in agent.robot.get_active_joints()]
        if len(fingers) != 2 or names != list(agent.arm_joint_names) + fingers:
            raise ValueError("Unverified robot-state to finger-joint mapping")
        spec = prismatic_joint_spec(agent.urdf_path, fingers[0])
        if self.gripper_command(spec["lower_m"]) >= self.gripper_command(spec["upper_m"]):
            raise ValueError("Gripper closing direction does not match the supplied coordinate")
        return {**spec, "state_index": 7, "closing_direction": "decreasing_position"}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
