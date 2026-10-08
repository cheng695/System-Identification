"""集中配置；修改后重新启动脚本生效。路径、dt 和列映射仍由终端输入。"""

from pathlib import Path

# 训练设置
# 最多执行的参数更新次数；达到次数不代表已经收敛。
NUM_EPOCHS = 5000
# 每个训练窗口包含的采样点数量。
# 长序列一次性反向传播容易造成梯度数值不稳定；训练时会把整段 CSV
# 切成多个窗口，并从每个窗口的真实初始状态重新开始仿真。
TRAINING_WINDOW_STEPS = 20
# 每次参数更新累积多少个窗口的梯度；每遍打乱顺序并遍历全部训练窗口。
# 单窗口编译复用，增加批量不会展开多份动力学计算图。
TRAINING_WINDOW_COUNT = 10
VALIDATION_FRACTION = 0.2
VALIDATION_EVERY = 50
TRAINING_RANDOM_SEED = 42
# 训练后的分窗评估仍覆盖完整 CSV；每隔多少个窗口报告进度。
EVALUATION_LOG_EVERY = 50
# 连续自由仿真耗时较长，默认关闭；分窗结果保存后才执行此可选步骤。
SAVE_FREE_ROLLOUT = True
# Adam 优化器的学习率，控制优化变量的更新幅度。
# 当前优化变量是参数的对数，因此不是每次直接给物理参数加 0.01。
LEARNING_RATE = 5e-5
# 每隔多少次更新打印一次损失和参数；初始状态也会打印。
LOG_EVERY = 10
# 全部参数梯度的整体范数上限；超过时按比例缩小梯度，再交给 Adam。
GRADIENT_CLIP_NORM = 0.02

# 摩擦初值也用于未参与辨识的关节；保持原有取值。
# 粘性摩擦系数 B 的初始值，单位 N·m·s/rad；粘性摩擦力矩为 B × 角速度。
INITIAL_VISCOUS = 0.005
# 库伦摩擦系数 Fc 的初始值，单位 N·m；远离零速时摩擦力矩幅值接近 Fc。
INITIAL_COULOMB = 0.005
# B 和 Fc 优化时共用的数值下限；两者单位分别与各自初值相同。
FRICTION_LOWER_BOUND = 1.0e-6
# B 和 Fc 优化时共用的数值上限；这是数值搜索范围，不代表真实物理上限。
# 如果出现 NaN，应优先处理反向传播和优化步长，而不是随意压低这个范围。
FRICTION_UPPER_BOUND = 0.2
# 平滑库伦摩擦 Fc*tanh(k*角速度) 中的 k，单位 s/rad。
# k 越大，零速附近过渡越陡；它是固定设置，不参与参数辨识。
FRICTION_K = 20.0
# 小 yaw 参数辨识不依赖地板、轮子和底盘接触；训练阶段关闭接触以稳定梯度。
# viewer 仍然使用 robot.xml 中的地板和碰撞体。
DISABLE_CONTACTS_DURING_TRAINING = True
# 惯量模式下优化固定锁定姿态的关节等效惯量，单位 kg·m²。
# 范围用于限制数值搜索，不是对完整惯性张量的物理约束。
INERTIA_LOWER_BOUND = 1.0e-4
INERTIA_UPPER_BOUND = 5.0e-2

# 损失权重（角度、速度均方误差）
# 角度均方误差的乘数，越大越重视角度拟合；当前误差单位为 rad²。
POSITION_LOSS_WEIGHT = 1.0
# 速度均方误差的乘数，越大越重视速度拟合；当前误差单位为 (rad/s)²。
# 总损失 = 角度权重 × 角度均方误差 + 速度权重 × 速度均方误差。
VELOCITY_LOSS_WEIGHT = 1.0
# 暂时恢复经过验证的原始单位 MSE 基线；归一化会改变速度与角度的权衡。
# 训练目标为：角度 MSE + 速度 MSE，最终仍用物理单位 RMSE 评估。
NORMALIZE_LOSS_BY_DATA_STD = False
LOSS_SCALE_EPSILON = 1.0e-6

# MJX 求解器：保留现有原型设置。
# 每个仿真步中约束求解器的最大迭代次数，影响接触、关节限位等约束求解。
SOLVER_ITERATIONS = 1
# 适用的求解器在线搜索时允许的最大迭代次数；不是训练迭代次数。
SOLVER_LS_ITERATIONS = 1

# CSV 默认读取设置；逐列交互映射会覆盖默认列名。
# CSV 解码依次尝试的字符编码，用于兼容 UTF-8 和常见中文编码文件。
CSV_ENCODINGS = ("utf-8-sig", "gb18030", "gbk")
# CSV 角度单位：rad 表示弧度，deg 表示度；速度必须相应为 rad/s 或 deg/s。
# 选择 deg 时读取器将角度和速度一起转换为 rad 和 rad/s。
# 此设置不改变交互输入的锁定角度单位，锁定角度仍需填写 rad。
ANGLE_UNIT = "rad"
# 未逐关节指定输入类型时的默认规则：torque=力矩，current=电流，auto=自动判断。
# auto 优先使用完整力矩列，否则尝试完整电流列；电流通过力矩常数换算。
INPUT_TYPE = "auto"
# 按列名读取时默认的位置列，排列顺序对应局部关节顺序。
POSITION_COLUMNS = ("yaw_pos", "pitch_pos")
# 默认速度列，顺序应与位置列一致。
VELOCITY_COLUMNS = ("yaw_vel", "pitch_vel")
# 默认力矩列，数据应为 N·m，顺序应与位置列一致。
TORQUE_COLUMNS = ("yaw_torque", "pitch_torque")
# 默认电流列；电流应与输入的力矩常数单位匹配，例如 A 与 N·m/A。
CURRENT_COLUMNS = ("yaw_iq", "pitch_iq")

# 结果输出和绘图
# 项目根目录：由当前 config.py 的位置推导，与运行命令所在目录无关。
PROJECT_DIR = Path(__file__).resolve().parents[1]
# 训练结果的默认输出目录，也是结果分析和回放默认查找的位置。
RESULTS_DIR = PROJECT_DIR / "results"
# 保存轨迹、辨识参数和 dt 等数组的结果文件名；再次训练会覆盖同名文件。
RESULT_FILENAME = "identification_result.npz"
# 图片子目录名，默认创建在所读取结果文件的同级目录下。
FIGURES_DIRNAME = "figures"
# 真实角度与仿真角度对比图的文件名。
POSITION_FIGURE = "position_comparison.png"
# 真实速度与仿真速度对比图的文件名。
VELOCITY_FIGURE = "velocity_comparison.png"
# 角度和速度残差图的文件名，残差定义为仿真值减去真实值。
RESIDUAL_FIGURE = "residuals.png"
# 图片保存分辨率，单位为每英寸像素数；像素尺寸约为英寸尺寸乘 DPI。
FIGURE_DPI = 150
# 对比图和残差图的宽度，单位为英寸。
FIGURE_WIDTH = 12
# 对比图的最小总高度，单位为英寸。
FIGURE_MIN_HEIGHT = 4
# 对比图中每个关节分配的高度，单位为英寸。
# 总高度取“最小高度”和“该值 × 测量关节数”中的较大值。
FIGURE_HEIGHT_PER_JOINT = 3
# 残差图的总高度，单位为英寸，包含角度和速度两个子图。
RESIDUAL_FIGURE_HEIGHT = 7
# 仿真曲线的不透明度，范围 0～1；0 完全透明，1 完全不透明。
SIMULATION_LINE_ALPHA = 0.8
# Viewer 默认回放倍率：1 正常、2 两倍、0.5 半速；仅影响播放，不影响辨识。
PLAYBACK_SPEED = 1.0
