"""CSV 数据读取。"""

from __future__ import annotations

from gimbal_sysid import config

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gimbal_sysid.model import get_model_joint_names


@dataclass
class TrajectoryData:
    """以数组形式保存的一段采样轨迹。"""

    pos_real: np.ndarray
    vel_real: np.ndarray
    tor_real: np.ndarray
    # 这些关节编号来自完整 MJCF 模型，而不是 CSV 内部的列编号。
    active_joint_ids: np.ndarray | None = None
    model_joint_names: tuple[str, ...] = ()
    locked_joint_ids: np.ndarray | None = None
    locked_angles: np.ndarray | None = None

    @property
    def num_steps(self) -> int:
        return self.pos_real.shape[0]

    @property
    def num_joints(self) -> int:
        return self.pos_real.shape[1]

    @property
    def measurement_mask(self) -> np.ndarray:
        """返回完整模型中的测量掩码，已记录关节为 1。"""
        if self.active_joint_ids is None or not self.model_joint_names:
            return np.ones(self.num_joints, dtype=np.float32)

        mask = np.zeros(len(self.model_joint_names), dtype=np.float32)
        mask[self.active_joint_ids] = 1.0
        return mask

    def make_full_torque(self) -> np.ndarray:
        """将 CSV 中的局部力矩扩展到完整模型的关节数量。"""
        if self.active_joint_ids is None or not self.model_joint_names:
            return self.tor_real.copy()

        full_torque = np.zeros(
            (self.num_steps, len(self.model_joint_names)),
            dtype=np.float32,
        )
        full_torque[:, self.active_joint_ids] = self.tor_real
        return full_torque

    def with_model_configuration(
        self,
        model_joint_names: Sequence[str],
        active_joint_ids: Sequence[int],
        locked_joint_ids: Sequence[int],
        locked_angles: Sequence[float],
    ) -> "TrajectoryData":
        """附加完整模型关节映射和未测量关节的锁定角度。"""
        return TrajectoryData(
            pos_real=self.pos_real,
            vel_real=self.vel_real,
            tor_real=self.tor_real,
            active_joint_ids=np.asarray(active_joint_ids, dtype=np.int32),
            model_joint_names=tuple(model_joint_names),
            locked_joint_ids=np.asarray(locked_joint_ids, dtype=np.int32),
            locked_angles=np.asarray(locked_angles, dtype=np.float32),
        )


def select_recorded_joints(xml_path: str | Path) -> list[str]:
    """显示模型关节，并让用户选择当前 CSV 记录的关节。"""
    joint_names = get_model_joint_names(xml_path)

    print("模型中的关节：")
    for joint_id, joint_name in enumerate(joint_names):
        print(f"  {joint_id}: {joint_name}")

    joint_input = input(
        "请输入当前 CSV 记录的关节编号，用逗号分隔（例如 0,2）："
    ).strip()
    if not joint_input:
        raise ValueError("至少需要选择一个关节")

    try:
        joint_ids = [int(value.strip()) for value in joint_input.split(",")]
    except ValueError as exc:
        raise ValueError("关节编号必须是用逗号分隔的整数，例如 0,2") from exc

    if len(set(joint_ids)) != len(joint_ids):
        raise ValueError("不能重复选择同一个关节")

    invalid_ids = [joint_id for joint_id in joint_ids
                   if joint_id < 0 or joint_id >= len(joint_names)]
    if invalid_ids:
        raise ValueError(
            f"关节编号超出范围: {invalid_ids}；有效范围是 0 到 {len(joint_names) - 1}"
        )

    selected_names = [joint_names[joint_id] for joint_id in joint_ids]
    print(f"本次 CSV 选择的关节: {selected_names}")
    return selected_names


def ask_locked_joints(
    model_joint_names: Sequence[str],
    recorded_joint_names: Sequence[str],
) -> tuple[list[int], list[float]]:
    """询问没有 CSV 数据的关节及其锁定角度。"""
    recorded_set = set(recorded_joint_names)
    locked_ids = []
    locked_angles = []

    for joint_id, joint_name in enumerate(model_joint_names):
        if joint_name in recorded_set:
            continue

        angle_input = input(
            f"请输入未记录关节 {joint_name} 的锁定角度（单位 rad）："
        ).strip()
        try:
            angle = float(angle_input)
        except ValueError as exc:
            raise ValueError(
                f"关节 {joint_name} 的锁定角度必须是数字"
            ) from exc

        locked_ids.append(joint_id)
        locked_angles.append(angle)

    if locked_ids:
        print("锁定关节配置：")
        for joint_id, angle in zip(locked_ids, locked_angles):
            print(f"  {model_joint_names[joint_id]} = {angle} rad")

    return locked_ids, locked_angles


def _ask_column_mapping(
    frame: pd.DataFrame,
    selected_joints: Sequence[str],
) -> tuple[list[dict[str, int]], list[float]]:
    """逐列询问 CSV 列对应的关节信号。"""
    choices = []
    for joint_name in selected_joints:
        choices.extend(
            [
                f"{joint_name}角度",
                f"{joint_name}速度",
                f"{joint_name}力矩",
                f"{joint_name}电流",
            ]
        )

    print("CSV 列：")
    for column_id, column_name in enumerate(frame.columns):
        print(f"  {column_id}: {column_name}")

    print("可填写的列含义：")
    print("、".join(choices))
    print("不参与辨识的列请输入：跳过")

    mapping = [dict() for _ in selected_joints]
    joint_id_by_name = {
        joint_name: joint_id
        for joint_id, joint_name in enumerate(selected_joints)
    }

    for column_id, column_name in enumerate(frame.columns):
        meaning = input(
            f"第{column_id + 1}列「{column_name}」对应什么？："
        ).strip()
        if meaning == "跳过" or not meaning:
            continue

        matched = False
        for joint_name, joint_id in joint_id_by_name.items():
            signal_names = {
                f"{joint_name}角度": "position",
                f"{joint_name}速度": "velocity",
                f"{joint_name}力矩": "torque",
                f"{joint_name}电流": "current",
            }
            if meaning in signal_names:
                signal = signal_names[meaning]
                if signal in mapping[joint_id]:
                    raise ValueError(f"重复指定了 {meaning}")
                mapping[joint_id][signal] = column_id
                matched = True
                break

        if not matched:
            raise ValueError(
                f"无法识别“{meaning}”，请严格填写上面列出的含义"
            )

    for joint_name, item in zip(selected_joints, mapping):
        if "position" not in item or "velocity" not in item:
            raise ValueError(f"{joint_name} 缺少角度列或速度列")
        if "torque" not in item and "current" not in item:
            raise ValueError(f"{joint_name} 缺少力矩列或电流列")

    motor_constants = []
    for joint_name, item in zip(selected_joints, mapping):
        if "current" in item:
            constant_input = input(
                f"请输入关节 {joint_name} 的力矩常数 Kt："
            ).strip()
            try:
                motor_constants.append(float(constant_input))
            except ValueError as exc:
                raise ValueError("力矩常数必须是数字") from exc
        else:
            motor_constants.append(1.0)

    return mapping, motor_constants


def _read_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"缺少{name}列: {missing}; "
            f"当前列为: {list(frame.columns)}"
        )

    return frame[list(columns)].to_numpy(dtype=np.float32)


def _read_column_by_index(
    frame: pd.DataFrame,
    column_index: int,
    description: str,
) -> np.ndarray:
    """按照 CSV 的实际列序号读取一列数据。"""
    if column_index < 0 or column_index >= frame.shape[1]:
        raise ValueError(
            f"{description}的列编号 {column_index} 超出范围，"
            f"当前 CSV 一共有 {frame.shape[1]} 列"
        )
    return frame.iloc[:, column_index].to_numpy(dtype=np.float32)


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """按常见编码读取 CSV 文件。"""
    last_error: UnicodeDecodeError | None = None
    for encoding in config.CSV_ENCODINGS:
        try:
            frame = pd.read_csv(csv_path, encoding=encoding)
            print(f"CSV 编码: {encoding}")
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法读取 CSV 编码，请将文件另存为 UTF-8。原始错误: {last_error}",
    )


def load_csv(
    csv_path: str | Path,
    position_columns: Sequence[str] = config.POSITION_COLUMNS,
    velocity_columns: Sequence[str] = config.VELOCITY_COLUMNS,
    torque_columns: Sequence[str] = config.TORQUE_COLUMNS,
    iq_columns: Sequence[str] = config.CURRENT_COLUMNS,
    motor_constants: Sequence[float] | None = None,
    angle_unit: str = config.ANGLE_UNIT,
    input_type: str = config.INPUT_TYPE,
    input_types: Sequence[str] | None = None,
    column_mapping: Sequence[dict[str, int]] | None = None,
) -> TrajectoryData:
    """读取 CSV 中的位置、速度和输入数据。

    ``input_type`` 用于所有关节使用同一种输入类型；
    ``input_types`` 用于为每个关节分别指定输入类型。

    电流会根据 ``torque = current * motor_constant`` 转换为力矩。
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    frame = _read_csv_with_fallback(csv_path)
    if column_mapping is not None:
        # 每个字典对应一个关节，例如：
        # {"position": 0, "velocity": 1, "torque": 2}
        pos_real = np.stack(
            [
                _read_column_by_index(frame, item["position"], "位置")
                for item in column_mapping
            ],
            axis=1,
        )
        vel_real = np.stack(
            [
                _read_column_by_index(frame, item["velocity"], "速度")
                for item in column_mapping
            ],
            axis=1,
        )

        torque_values = []
        mapped_input_types = []
        for item in column_mapping:
            if "torque" in item:
                torque_values.append(
                    _read_column_by_index(frame, item["torque"], "力矩")
                )
                mapped_input_types.append("torque")
            elif "current" in item:
                torque_values.append(
                    _read_column_by_index(frame, item["current"], "电流")
                )
                mapped_input_types.append("current")
            else:
                raise ValueError("每个关节都必须指定 torque 或 current 列")

        input_types = mapped_input_types
        tor_real = np.stack(torque_values, axis=1).astype(np.float32)

        if motor_constants is None:
            constants = np.ones(len(input_types), dtype=np.float32)
        else:
            constants = np.asarray(motor_constants, dtype=np.float32)
            if constants.shape != (len(input_types),):
                raise ValueError("motor_constants 数量必须与关节数量一致")

        for joint_id, signal_type in enumerate(input_types):
            if signal_type == "current":
                tor_real[:, joint_id] *= constants[joint_id]
    else:
        pos_real = _read_columns(frame, position_columns, "位置")
        vel_real = _read_columns(frame, velocity_columns, "速度")

        if input_types is None:
            if input_type not in {"auto", "torque", "current"}:
                raise ValueError("input_type 只能是 auto、torque 或 current")

            has_torque = all(column in frame.columns for column in torque_columns)
            has_current = all(column in frame.columns for column in iq_columns)

            if input_type == "torque" or (input_type == "auto" and has_torque):
                input_types = ["torque"] * len(torque_columns)
            elif input_type == "current" or (input_type == "auto" and has_current):
                input_types = ["current"] * len(iq_columns)
            else:
                raise ValueError("CSV 中没有完整的力矩列或电流列")
        else:
            input_types = [value.strip().lower() for value in input_types]
            if len(input_types) != len(position_columns):
                raise ValueError("input_types 数量必须与关节数量一致")
            if any(value not in {"torque", "current"} for value in input_types):
                raise ValueError("input_types 中只能填写 torque 或 current")

        if motor_constants is None:
            constants = np.ones(len(input_types), dtype=np.float32)
        else:
            constants = np.asarray(motor_constants, dtype=np.float32)
            if constants.shape != (len(input_types),):
                raise ValueError("motor_constants 数量必须与关节数量一致")

        torque_values = []
        for joint_id, signal_type in enumerate(input_types):
            if signal_type == "torque":
                values = _read_columns(
                    frame, [torque_columns[joint_id]], "力矩"
                )[:, 0]
            else:
                current = _read_columns(
                    frame, [iq_columns[joint_id]], "电流"
                )[:, 0]
                values = current * constants[joint_id]
            torque_values.append(values)

        tor_real = np.stack(torque_values, axis=1).astype(np.float32)

    if angle_unit == "deg":
        pos_real = np.deg2rad(pos_real).astype(np.float32)
        vel_real = np.deg2rad(vel_real).astype(np.float32)
    elif angle_unit != "rad":
        raise ValueError("angle_unit 只能是 'rad' 或 'deg'")

    lengths = {
        pos_real.shape[0],
        vel_real.shape[0],
        tor_real.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError("位置、速度和力矩的采样长度不一致")

    return TrajectoryData(
        pos_real=pos_real,
        vel_real=vel_real,
        tor_real=tor_real,
    )


def main() -> None:
    xml_input = input("请输入 MJCF 模型绝对路径：").strip()
    if not xml_input:
        raise ValueError("MJCF 模型路径不能为空")

    model_joint_names = get_model_joint_names(xml_input)
    selected_joints = select_recorded_joints(xml_input)
    active_joint_ids = [
        model_joint_names.index(joint_name)
        for joint_name in selected_joints
    ]
    locked_joint_ids, locked_angles = ask_locked_joints(
        model_joint_names,
        selected_joints,
    )

    csv_input = input("请输入 CSV 绝对路径：").strip()
    if not csv_input:
        raise ValueError("CSV 路径不能为空")

    csv_path = Path(csv_input).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    frame = _read_csv_with_fallback(csv_path)
    column_mapping, motor_constants = _ask_column_mapping(
        frame,
        selected_joints,
    )

    data = load_csv(
        csv_input,
        motor_constants=motor_constants,
        column_mapping=column_mapping,
    )
    data = data.with_model_configuration(
        model_joint_names=model_joint_names,
        active_joint_ids=active_joint_ids,
        locked_joint_ids=locked_joint_ids,
        locked_angles=locked_angles,
    )

    print(f"samples(采样点数量) = {data.num_steps}")
    print(f"joints(当前 CSV 记录的关节数量) = {data.num_joints}")
    print(f"pos shape = {data.pos_real.shape}")
    print(f"vel shape = {data.vel_real.shape}")
    print(f"torque shape = {data.tor_real.shape}")
    print(f"active joint ids(对应的模型关节编号) = {data.active_joint_ids.tolist()}")
    print(f"measurement mask(测量关节掩码) = {data.measurement_mask.tolist()}")
    print(f"full torque shape() = {data.make_full_torque().shape}")


if __name__ == "__main__":
    main()
