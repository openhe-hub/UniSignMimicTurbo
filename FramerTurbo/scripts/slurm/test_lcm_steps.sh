#!/bin/bash
# Framer Turbo V2 - LCM 调度器步数测试脚本
# 专门测试 LCM 在不同步数 (20/15/10/5) 下的性能和质量

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 测试数据路径
TEST_DATA="assets/test_single"
MODEL_PATH="checkpoints/framer_512x320"

# 输出目录
OUTPUT_BASE="outputs/lcm_steps_comparison"
mkdir -p "$OUTPUT_BASE"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# 辅助函数
# ============================================================================

# 读取推理时间（从 Python 脚本输出的文件）
read_inference_time() {
    local output_dir=$1
    local time_file="$output_dir/.inference_time.txt"

    if [ -f "$time_file" ]; then
        # 读取第一行：总推理时间
        head -n 1 "$time_file"
    else
        echo "0"
    fi
}

print_header() {
    echo ""
    echo "=================================================================="
    echo "$1"
    echo "=================================================================="
    echo ""
}

# ============================================================================
# 预检查
# ============================================================================

print_header "Framer Turbo V2 - LCM 调度器步数测试"

echo "测试配置:"
echo "  - 测试数据: $TEST_DATA"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 输出目录: $OUTPUT_BASE"
echo "  - 调度器: LCM"
echo "  - 测试步数: 20, 15, 10, 5"
echo ""

# 检查测试数据
if [ ! -d "$TEST_DATA" ]; then
    echo "❌ 错误: 测试数据目录不存在: $TEST_DATA"
    echo "请先创建测试目录并放入图像对 (例如: 0_start.jpg, 0_end.jpg)"
    exit 1
fi

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  警告: 模型目录不存在: $MODEL_PATH"
    echo "请确保模型已下载"
fi

# 检查 Python 脚本
if [ ! -f "cli_infer_turbo_v2.py" ]; then
    echo "❌ 错误: cli_infer_turbo_v2.py 不存在"
    exit 1
fi

echo "✓ 预检查通过"
echo ""

# ============================================================================
# 测试 1: LCM (20 steps)
# ============================================================================

print_header "测试 1/4: LCM 调度器 (20 steps)"

OUTPUT_LCM_20="$OUTPUT_BASE/lcm_20steps_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM_20"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM_20" \
    --scheduler lcm \
    --num_inference_steps 20

LCM_20_TIME=$(read_inference_time "$OUTPUT_LCM_20")
echo "✓ LCM (20 steps) 完成，纯推理时间: ${LCM_20_TIME}s"

# ============================================================================
# 测试 2: LCM (15 steps)
# ============================================================================

print_header "测试 2/4: LCM 调度器 (15 steps)"

OUTPUT_LCM_15="$OUTPUT_BASE/lcm_15steps_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM_15"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM_15" \
    --scheduler lcm \
    --num_inference_steps 15

LCM_15_TIME=$(read_inference_time "$OUTPUT_LCM_15")
echo "✓ LCM (15 steps) 完成，纯推理时间: ${LCM_15_TIME}s"

# ============================================================================
# 测试 3: LCM (10 steps)
# ============================================================================

print_header "测试 3/4: LCM 调度器 (10 steps)"

OUTPUT_LCM_10="$OUTPUT_BASE/lcm_10steps_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM_10"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM_10" \
    --scheduler lcm \
    --num_inference_steps 10

LCM_10_TIME=$(read_inference_time "$OUTPUT_LCM_10")
echo "✓ LCM (10 steps) 完成，纯推理时间: ${LCM_10_TIME}s"

# ============================================================================
# 测试 4: LCM (5 steps)
# ============================================================================

print_header "测试 4/4: LCM 调度器 (5 steps)"

OUTPUT_LCM_5="$OUTPUT_BASE/lcm_5steps_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM_5"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM_5" \
    --scheduler lcm \
    --num_inference_steps 5

LCM_5_TIME=$(read_inference_time "$OUTPUT_LCM_5")
echo "✓ LCM (5 steps) 完成，纯推理时间: ${LCM_5_TIME}s"

# ============================================================================
# 性能汇总
# ============================================================================

print_header "性能汇总报告"

# 计算加速比（使用 bc 进行浮点运算，以 20 steps 为基准）
SPEEDUP_15=$(echo "scale=2; $LCM_20_TIME / $LCM_15_TIME" | bc)
SPEEDUP_10=$(echo "scale=2; $LCM_20_TIME / $LCM_10_TIME" | bc)
SPEEDUP_5=$(echo "scale=2; $LCM_20_TIME / $LCM_5_TIME" | bc)

echo "注意：以下时间为纯推理时间，不包括模型加载"
echo ""
echo "LCM 调度器步数对比         步数    推理时间    相对加速比"
echo "-------------------------------------------------------------------"
printf "%-25s    %-4s    %-8ss    %-6s\n" "LCM (baseline)" "20" "$LCM_20_TIME" "1.00x"
printf "%-25s    %-4s    %-8ss    %-6s\n" "LCM (推荐)" "15" "$LCM_15_TIME" "${SPEEDUP_15}x"
printf "%-25s    %-4s    %-8ss    %-6s\n" "LCM (激进)" "10" "$LCM_10_TIME" "${SPEEDUP_10}x"
printf "%-25s    %-4s    %-8ss    %-6s\n" "LCM (超激进)" "5" "$LCM_5_TIME" "${SPEEDUP_5}x"
echo "-------------------------------------------------------------------"

# ============================================================================
# 保存报告
# ============================================================================

REPORT_FILE="$OUTPUT_BASE/lcm_steps_report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" <<EOF
Framer Turbo V2 - LCM 调度器步数测试报告
====================================
测试时间: $(date)
测试数据: $TEST_DATA
模型路径: $MODEL_PATH
调度器: LCM
系统信息: $(uname -a)
GPU 信息: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")

注意：以下时间为纯推理时间，不包括模型加载

性能结果:
------------------------------------
步数配置          步数    推理时间    相对加速比
------------------------------------
LCM (baseline)    20      ${LCM_20_TIME}s    1.00x
LCM (推荐)        15      ${LCM_15_TIME}s    ${SPEEDUP_15}x
LCM (激进)        10      ${LCM_10_TIME}s    ${SPEEDUP_10}x
LCM (超激进)      5       ${LCM_5_TIME}s     ${SPEEDUP_5}x

输出目录:
------------------------------------
LCM 20 steps:   $OUTPUT_LCM_20
LCM 15 steps:   $OUTPUT_LCM_15
LCM 10 steps:   $OUTPUT_LCM_10
LCM 5 steps:    $OUTPUT_LCM_5

质量评估:
------------------------------------
请打开输出目录中的 GIF/MP4 文件，对比视觉质量：
cd $OUTPUT_BASE

推荐使用以下命令下载到本地查看：
scp -r user@server:$OUTPUT_BASE ./local_results/

建议:
------------------------------------
- 20 steps: 最佳质量，推理时间较长
- 15 steps: 推荐配置，质量和速度平衡
- 10 steps: 激进模式，质量略有下降
- 5 steps:  超激进模式，最快速度，质量可能明显下降

EOF

echo ""
echo "📊 测试报告已保存: $REPORT_FILE"
echo ""

# 显示报告内容
cat "$REPORT_FILE"

# 清理临时文件
rm -f "/tmp/framer_last_time_$$.txt"

print_header "测试完成！"

echo "下一步："
echo "  1. 查看输出目录中的视频文件对比质量"
echo "  2. 根据质量要求选择合适的步数配置"
echo "  3. 如需下载结果到本地："
echo "     scp -r user@server:$OUTPUT_BASE ./local_results/"
echo ""
