# GPU Configuration Guide for FramerTurbo Training

快速决策指南：根据你能申请到的GPU资源选择最佳配置。

## 🎯 快速决策

### 情况1：申请2-3块GPU ✅ 完全可行

| GPU数量 | 有效Batch | 训练时间 | Slurm脚本 |
|---------|----------|---------|----------|
| 2卡 | 8 (1×4×2) | ~1.5x基线 | `sbatch scripts/slurm/train_lora_2gpu.sh` |
| 3卡 | 9 (1×3×3) | ~2.2x基线 | `sbatch scripts/slurm/train_lora_3gpu.sh` |
| 4卡 | 8 (1×2×4) | ~3.2x基线 | `sbatch scripts/slurm/train_lora_4gpu.sh` |

**显存需求**：每张卡仍然需要 ~25-27GB（ZeRO-2）

**结论**：2-3卡完全可行，只是训练速度稍慢，但质量完全一样！

### 情况2：混合GPU型号 ⚠️ 可行但不推荐

#### 问题分析

```
假设你分配到:
- 2x A100-80GB (快，显存大)
- 2x V100-32GB (慢，显存小)

会发生什么？
┌────────────────────────────────────────┐
│ GPU 0 (A100): 计算完成 ████████████ 100% │ ← 等待慢卡
│ GPU 1 (A100): 计算完成 ████████████ 100% │ ← 等待慢卡
│ GPU 2 (V100): 计算中   ██████░░░░░  60% │ ← 瓶颈！
│ GPU 3 (V100): 计算中   ██████░░░░░  60% │ ← 瓶颈！
└────────────────────────────────────────┘
所有GPU必须同步，速度 = 最慢的GPU
```

#### 混合GPU的影响

1. **性能瓶颈**
   - 速度由最慢的GPU决定
   - A100快40%，但要等V100
   - 实际加速比 = V100的速度 × GPU数量

2. **显存不匹配**
   - V100-32GB可能接近上限（需要ZeRO-3）
   - A100-80GB大量闲置
   - DeepSpeed的分摊机制可能不均衡

3. **通信问题**
   - 不同世代GPU的NVLink/PCIe速度不同
   - 可能导致通信成为瓶颈

#### 如果必须使用混合GPU

使用灵活配置脚本：
```bash
sbatch scripts/slurm/train_lora_flexible.sh
```

这个脚本会：
- ✅ 自动检测GPU型号
- ✅ 如果有V100或32GB显存GPU，自动切换到ZeRO-3
- ✅ 显示警告信息
- ✅ 继续训练（速度慢但能跑）

## 📊 配置对比表

### 同构GPU（推荐）

| 配置 | 提交命令 | 显存/卡 | 速度 | 排队时间 |
|------|---------|---------|------|---------|
| 4x A100-80GB | `sbatch train_lora_4gpu.sh` | 25GB | 最快 | 长 |
| 4x A100-40GB | `sbatch train_lora_4gpu.sh` | 25GB | 最快 | 中 |
| 3x A100-40GB | `sbatch train_lora_3gpu.sh` | 25GB | 快 | 短 |
| 2x A100-40GB | `sbatch train_lora_2gpu.sh` | 25GB | 中 | 很短 |
| 4x V100-32GB | `sbatch train_lora_4gpu.sh`* | 21GB | 中慢 | 短 |

*需修改脚本使用 `deepspeed_zero3.json`

### 混合GPU（不推荐）

| 配置 | 提交命令 | 预期问题 |
|------|---------|---------|
| 2xA100 + 2xV100 | `sbatch train_lora_flexible.sh` | 速度慢40%，通信可能不稳定 |
| 混合不同代A100 | `sbatch train_lora_flexible.sh` | 小问题，可能速度不均 |

## 🚀 推荐策略

### 策略1：快速启动（推荐）
```bash
# 申请2卡A100，排队时间最短
#SBATCH --gres=gpu:a100:2

sbatch scripts/slurm/train_lora_2gpu.sh
```
- ✅ 排队快
- ✅ 资源充足
- ✅ 速度够用（1.5-2x单卡）
- ⚠️ 训练时间稍长（约1.5天 vs 4卡的1天）

### 策略2：平衡性价比
```bash
# 申请3卡A100
#SBATCH --gres=gpu:a100:3

sbatch scripts/slurm/train_lora_3gpu.sh
```
- ✅ 性价比高
- ✅ 速度快（2.2x）
- ⚠️ 有效batch=9（略高于8）

### 策略3：最快训练
```bash
# 申请4卡A100，可能需要等待
#SBATCH --gres=gpu:a100:4

sbatch scripts/slurm/train_lora_4gpu.sh
```
- ⚠️ 排队时间长
- ✅ 速度最快（3.2x）
- ✅ 有效batch=8（标准）

### 策略4：灵活应对（备选）
```bash
# 接受任意4卡，不指定型号
#SBATCH --gres=gpu:4

sbatch scripts/slurm/train_lora_flexible.sh
```
- ✅ 排队最快
- ⚠️ 可能分配到混合GPU
- ⚠️ 速度不可预测

## 🔍 如何检查分配的GPU

提交任务后，查看日志：
```bash
# 等任务开始后
tail -f logs/train_XXX_JOBID.out

# 查看GPU信息部分
```

输出示例：
```
========================================
GPU Info
========================================
index, name, memory.total, memory.free
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 80000 MiB
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 80000 MiB
2, NVIDIA A100-SXM4-80GB, 81920 MiB, 80000 MiB
3, NVIDIA A100-SXM4-80GB, 81920 MiB, 80000 MiB

✅ Using ZeRO-2 (faster, sufficient for A100-40GB+)
```

如果看到混合：
```
⚠️  WARNING: Mixed GPU types detected!
┌───────┬──────────────────────┬──────────────┐
│ index │ name                 │ memory.total │
├───────┼──────────────────────┼──────────────┤
│ 0     │ A100-SXM4-80GB      │ 81920 MiB    │
│ 1     │ A100-SXM4-80GB      │ 81920 MiB    │
│ 2     │ Tesla V100-SXM2-32GB│ 32768 MiB    │
│ 3     │ Tesla V100-SXM2-32GB│ 32768 MiB    │
└───────┴──────────────────────┴──────────────┘
```

## 💡 最终建议

1. **排队时间不是问题** → 申请 `4x A100`，等值得
2. **希望快速开始** → 申请 `2x A100`，立即开始
3. **平衡选择** → 申请 `3x A100`
4. **只有V100可用** → 申请 `4x V100`，记得改用ZeRO-3
5. **可能混合GPU** → 使用 `train_lora_flexible.sh`，做好速度慢的准备

## 📝 修改Slurm脚本示例

### 申请2卡
```bash
#SBATCH --gres=gpu:a100:2  # 指定2卡A100
#SBATCH --ntasks-per-node=2
sbatch scripts/slurm/train_lora_2gpu.sh
```

### 申请V100
```bash
#SBATCH --gres=gpu:v100:4  # 指定4卡V100

# 训练脚本需要改用ZeRO-3
# 编辑 scripts/train/train_lora_4gpu.sh:
# DEEPSPEED_CONFIG="configs/deepspeed_zero3.json"

sbatch scripts/slurm/train_lora_4gpu.sh
```

### 不指定型号（灵活）
```bash
#SBATCH --gres=gpu:4  # 任意4卡
sbatch scripts/slurm/train_lora_flexible.sh  # 自动适配
```
