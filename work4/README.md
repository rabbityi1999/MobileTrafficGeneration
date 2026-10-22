# 工作四代码导航

`experiment/work4_zero_label_20261018/` 保留原始实验代码布局。

| 文件 | 用途 |
| --- | --- |
| `train_stage0.py` | 130 维条件、168 维输出的普通 CFM；含 CPU 合成 smoke 检查 |
| `classical_v1/run_classical.py` | 源均值、经验分布、Ridge 残差重采样历史对照 |
| `gaussian_prior_cfm_v1/prior_ops.py` | 高斯起点的损失与采样 |
| `spatial_prior_cfm_v1/spatial_ops.py` | 输入消融、仅由训练区域估计的高斯矩、输入包契约 |
| `spatial_prior_cfm_v1/selftest.py` | 条件干预、协方差、起点等价性、梯度有限性检查 |
| `spatial_prior_cfm_v1/remote_package/` | 最近空间隔离实验的训练核心和空闲 GPU 检查器；不含 SSH 或凭据 |
| `spatial_prior_cfm_v1/assessment_v1/metrics.py` | 样本 CRPS、公平 CRPS、90% 覆盖及时间滞后结构诊断 |

`research/.../objective_check.py` 用有理数构造 29 项检查，区分平均风险、固定基线增量与 oracle regret；不构成真实数据泛化证据。

## 数据契约

普通基线读取 `source_only.npz`、`SOURCE_PACKAGE.json` 和 `STAGE0_SEAL.json`。这些私有实验产物不在本仓库中。

空间实验 worker 需要相邻的 `CONFIG.json`、`PACKAGE_SEAL.json` 与 `fold0.npz` 至 `fold3.npz`。每个 fold 包包含：

- `x_outer_train`: `(N, 130)`，`y_outer_train`: `(N, 2, 168)`。
- `rows_outer_train`: `(N,)`。
- `x_query`: `(128, 130)`，`rows_query`: `(128,)`；不得包含查询流量真值。
- `means`: `(2, 168)`，`factors`: `(2, 168, 168)`；只能由本 fold 训练标签估计。

模型处理的 y 为 `[0,1]` profile，训练内部映射到 `[-1,1]`。配置中的 24 fits = 4 folds × 2 channels × 3 condition arms。训练种子为 1；三个采样种子生成 72 份预测。

独立评价需要另备查询真值，且在全部候选与对照预测封存后才访问。首次发布未包含本地凭据读取器、私有数据导出器、完整远端调度/收集流程和历史论文生成脚本。
