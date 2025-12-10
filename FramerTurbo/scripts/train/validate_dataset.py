"""
数据集验证脚本
用于检查训练数据是否符合要求
"""

import os
import sys
import cv2
from pathlib import Path

sys.path.insert(0, os.getcwd())
from training.train_dataset import VideoFrameDataset, ImagePairDataset


def check_video_quality(video_path):
    """检查单个视频的质量"""
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
    }


def validate_video_dataset(data_dir):
    """验证视频数据集"""
    print("=" * 70)
    print("验证视频数据集")
    print("=" * 70)

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 目录不存在: {data_dir}")
        return False

    print(f"📁 数据目录: {data_dir}\n")

    # 收集视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    if len(video_files) == 0:
        print(f"❌ 错误: 未找到视频文件")
        print(f"   支持的格式: {', '.join(video_extensions)}")
        return False

    print(f"✓ 找到 {len(video_files)} 个视频文件\n")

    # 检查每个视频
    valid_videos = []
    invalid_videos = []

    print("检查视频质量...")
    for idx, video_path in enumerate(video_files, 1):
        try:
            info = check_video_quality(video_path)

            # 检查帧数
            if info['frame_count'] < 16:
                invalid_videos.append((video_path, f"帧数不足 ({info['frame_count']} < 16)"))
                continue

            # 检查分辨率
            if info['width'] == 0 or info['height'] == 0:
                invalid_videos.append((video_path, "无法读取分辨率"))
                continue

            valid_videos.append((video_path, info))

            if idx <= 5:  # 显示前 5 个视频的详细信息
                print(f"  [{idx}] {os.path.basename(video_path)}")
                print(f"      分辨率: {info['width']}x{info['height']}")
                print(f"      帧数: {info['frame_count']} @ {info['fps']:.1f} fps")
                print(f"      时长: {info['duration']:.2f} 秒")

        except Exception as e:
            invalid_videos.append((video_path, str(e)))

    if len(valid_videos) > 5:
        print(f"  ... 还有 {len(valid_videos) - 5} 个视频")

    # 统计信息
    print(f"\n{'=' * 70}")
    print("统计信息")
    print("=" * 70)
    print(f"✓ 有效视频: {len(valid_videos)}")
    print(f"✗ 无效视频: {len(invalid_videos)}")

    if invalid_videos:
        print(f"\n无效视频列表:")
        for video_path, reason in invalid_videos[:10]:
            print(f"  - {os.path.basename(video_path)}: {reason}")
        if len(invalid_videos) > 10:
            print(f"  ... 还有 {len(invalid_videos) - 10} 个")

    # 分辨率统计
    if valid_videos:
        resolutions = {}
        for _, info in valid_videos:
            res = f"{info['width']}x{info['height']}"
            resolutions[res] = resolutions.get(res, 0) + 1

        print(f"\n分辨率分布:")
        for res, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {res}: {count} 个视频")

    # 尝试加载数据集
    print(f"\n{'=' * 70}")
    print("测试数据集加载")
    print("=" * 70)

    try:
        dataset = VideoFrameDataset(
            video_dir=data_dir,
            num_frames=3,
            height=320,
            width=512,
            min_video_frames=16,
        )

        print(f"✓ 数据集创建成功")
        print(f"✓ 可用样本数: {len(dataset)}")

        # 测试加载第一个样本
        print(f"\n测试加载样本...")
        sample = dataset[0]
        print(f"✓ 样本形状: {sample['pixel_values'].shape}")
        print(f"  - pixel_values: {sample['pixel_values'].shape}")
        print(f"  - first_frame: {sample['first_frame'].shape}")
        print(f"  - last_frame: {sample['last_frame'].shape}")
        print(f"  - video_path: {os.path.basename(sample['video_path'])}")

        print(f"\n{'=' * 70}")
        print("✅ 数据集验证通过！")
        print("=" * 70)

        # 给出建议
        if len(dataset) < 50:
            print(f"⚠️  警告: 数据量较少 ({len(dataset)} 个视频)")
            print(f"   建议: 准备至少 100 个视频以获得更好效果")
        elif len(dataset) < 100:
            print(f"ℹ️  提示: 数据量适中 ({len(dataset)} 个视频)")
            print(f"   建议: 可以开始训练，但更多数据会有更好效果")
        else:
            print(f"✅ 数据量充足 ({len(dataset)} 个视频)")

        return True

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_image_pair_dataset(data_dir):
    """验证图像对数据集"""
    print("=" * 70)
    print("验证图像对数据集")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"❌ 错误: 目录不存在: {data_dir}")
        return False

    print(f"📁 数据目录: {data_dir}\n")

    try:
        dataset = ImagePairDataset(
            data_dir=data_dir,
            height=320,
            width=512,
            num_frames=3,
        )

        print(f"✓ 找到 {len(dataset)} 个图像对")

        if len(dataset) == 0:
            print(f"❌ 错误: 未找到有效的图像对")
            print(f"   期望的文件命名格式:")
            print(f"   - sample_001_start.jpg / sample_001_end.jpg")
            print(f"   - sample_002_start.png / sample_002_end.png")
            return False

        # 测试第一个样本
        sample = dataset[0]
        print(f"\n样本信息:")
        print(f"  - pixel_values: {sample['pixel_values'].shape}")
        print(f"  - first_frame: {sample['first_frame'].shape}")
        print(f"  - last_frame: {sample['last_frame'].shape}")
        print(f"  - start_path: {os.path.basename(sample['start_path'])}")
        print(f"  - end_path: {os.path.basename(sample['end_path'])}")

        print(f"\n{'=' * 70}")
        print("✅ 图像对数据集验证通过！")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证 FramerTurbo 训练数据集")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据目录路径"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="video",
        choices=["video", "image_pair"],
        help="数据集类型"
    )

    args = parser.parse_args()

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "FramerTurbo 数据集验证工具" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    if args.type == "video":
        success = validate_video_dataset(args.data_dir)
    else:
        success = validate_image_pair_dataset(args.data_dir)

    print("\n")
    if success:
        print("🎉 验证完成！你可以开始训练了：")
        print("   bash scripts/train_lora.sh")
    else:
        print("❌ 验证失败，请检查数据集")
        print("   查看文档: docs/DATA_PREPARATION.md")
    print("\n")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
