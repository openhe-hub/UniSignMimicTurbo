#!/bin/bash
# Framer Turbo V2 测试脚本 (Ubuntu 服务器版本)
# 对比三种调度器的性能和质量

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 测试数据路径
TEST_DATA="assets/test_single"
MODEL_PATH="checkpoints/framer_512x320"

# 输出目录
OUTPUT_BASE="outputs/scheduler_comparison"
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

print_header "Framer Turbo V2 调度器性能测试"

echo "测试配置:"
echo "  - 测试数据: $TEST_DATA"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 输出目录: $OUTPUT_BASE"
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
# 测试 1: Euler (原版，30步)
# ============================================================================

print_header "测试 1/5: Euler 调度器 (30 steps)"

OUTPUT_EULER="$OUTPUT_BASE/euler_${TIMESTAMP}"
mkdir -p "$OUTPUT_EULER"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_EULER" \
    --scheduler euler \
    --num_inference_steps 30

EULER_TIME=$(read_inference_time "$OUTPUT_EULER")
echo "✓ Euler 完成，纯推理时间: ${EULER_TIME}s"

# ============================================================================
# 测试 2: DPM++ (推荐，15步)
# ============================================================================

print_header "测试 2/5: DPM++ 调度器 (20 steps)"

OUTPUT_DPM="$OUTPUT_BASE/dpm++_${TIMESTAMP}"
mkdir -p "$OUTPUT_DPM"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_DPM" \
    --scheduler "dpm++" \
    --num_inference_steps 20

DPM_TIME=$(read_inference_time "$OUTPUT_DPM")
echo "✓ DPM++ 完成，纯推理时间: ${DPM_TIME}s"

# ============================================================================
# 测试 3: DPM++ with Karras (15步)
# ============================================================================

print_header "测试 3/5: DPM++ 激进模式 (10 steps)"

OUTPUT_DPM_K="$OUTPUT_BASE/dpm++_karras_${TIMESTAMP}"
mkdir -p "$OUTPUT_DPM_K"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_DPM_K" \
    --scheduler "dpm++" \
    --num_inference_steps 10
    # --dpm_use_karras_sigmas

DPM_K_TIME=$(read_inference_time "$OUTPUT_DPM_K")
echo "✓ DPM++ 激进模式 完成，纯推理时间: ${DPM_K_TIME}s"

# ============================================================================
# 测试 4: LCM (快速，6步)
# ============================================================================

print_header "测试 4/5: LCM 调度器 (6 steps)"

OUTPUT_LCM="$OUTPUT_BASE/lcm_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM" \
    --scheduler lcm \
    --num_inference_steps 6

LCM_TIME=$(read_inference_time "$OUTPUT_LCM")
echo "✓ LCM 完成，纯推理时间: ${LCM_TIME}s"

# ============================================================================
# 测试 5: LCM 激进模式 (4步)
# ============================================================================

print_header "测试 5/5: LCM 激进模式 (4 steps)"

OUTPUT_LCM_4="$OUTPUT_BASE/lcm_4steps_${TIMESTAMP}"
mkdir -p "$OUTPUT_LCM_4"

python cli_infer_turbo_v2.py \
    --input_dir "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --output_dir "$OUTPUT_LCM_4" \
    --scheduler lcm \
    --num_inference_steps 4

LCM_4_TIME=$(read_inference_time "$OUTPUT_LCM_4")
echo "✓ LCM 激进 完成，纯推理时间: ${LCM_4_TIME}s"

# ============================================================================
# 性能汇总
# ============================================================================

print_header "性能汇总报告"

# 计算加速比（使用 bc 进行浮点运算）
DPM_SPEEDUP=$(echo "scale=2; $EULER_TIME / $DPM_TIME" | bc)
DPM_K_SPEEDUP=$(echo "scale=2; $EULER_TIME / $DPM_K_TIME" | bc)
LCM_SPEEDUP=$(echo "scale=2; $EULER_TIME / $LCM_TIME" | bc)
LCM_4_SPEEDUP=$(echo "scale=2; $EULER_TIME / $LCM_4_TIME" | bc)

echo "注意：以下时间为纯推理时间，不包括模型加载"
echo ""
echo "调度器配置              步数    推理时间    加速比"
echo "-------------------------------------------------------------------"
printf "%-20s    %-4s    %-8ss    %-6s\n" "Euler (baseline)" "30" "$EULER_TIME" "1.00x"
printf "%-20s    %-4s    %-8ss    %-6s\n" "DPM++" "20" "$DPM_TIME" "${DPM_SPEEDUP}x"
printf "%-20s    %-4s    %-8ss    %-6s\n" "DPM++ (激进)" "10" "$DPM_K_TIME" "${DPM_K_SPEEDUP}x"
printf "%-20s    %-4s    %-8ss    %-6s\n" "LCM" "6" "$LCM_TIME" "${LCM_SPEEDUP}x"
printf "%-20s    %-4s    %-8ss    %-6s\n" "LCM (激进)" "4" "$LCM_4_TIME" "${LCM_4_SPEEDUP}x"
echo "-------------------------------------------------------------------"

# ============================================================================
# 保存报告
# ============================================================================

REPORT_FILE="$OUTPUT_BASE/benchmark_report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" <<EOF
Framer Turbo V2 性能测试报告
====================================
测试时间: $(date)
测试数据: $TEST_DATA
模型路径: $MODEL_PATH
系统信息: $(uname -a)
GPU 信息: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")

注意：以下时间为纯推理时间，不包括模型加载

性能结果:
------------------------------------
调度器              步数    推理时间    加速比
------------------------------------
Euler (baseline)    30      ${EULER_TIME}s    1.00x
DPM++              20      ${DPM_TIME}s     ${DPM_SPEEDUP}x
DPM++ (激进)       10      ${DPM_K_TIME}s   ${DPM_K_SPEEDUP}x
LCM                6       ${LCM_TIME}s      ${LCM_SPEEDUP}x
LCM (激进)         4       ${LCM_4_TIME}s    ${LCM_4_SPEEDUP}x

输出目录:
------------------------------------
Euler:          $OUTPUT_EULER
DPM++:          $OUTPUT_DPM
DPM++ (激进):   $OUTPUT_DPM_K
LCM:            $OUTPUT_LCM
LCM (激进):     $OUTPUT_LCM_4

质量评估:
------------------------------------
请打开输出目录中的 GIF/MP4 文件，对比视觉质量：
cd $OUTPUT_BASE

推荐使用以下命令下载到本地查看：
scp -r user@server:$OUTPUT_BASE ./local_results/

EOF

echo ""
echo "📊 测试报告已保存: $REPORT_FILE"
echo ""

# 显示报告内容
cat "$REPORT_FILE"

# 清理临时文件
rm -f "/tmp/framer_last_time_$$.txt"

print_header "测试完成！"
