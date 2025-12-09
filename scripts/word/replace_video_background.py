"""
Replace video background with solid color using background removal.
Processes MP4 videos frame by frame, removes background and replaces with specified color.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace video background with solid color"
    )
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        required=True,
        help="Path to output video file",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="white",
        help="Background color: 'white', 'black', 'green', or hex color like '#CBC4B7' (default: white)",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply sharpening filter to improve quality",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rembg",
        choices=["rembg", "grabcut"],
        help="Background removal method (default: rembg)",
    )
    return parser.parse_args()


def get_bg_color(color_name):
    """Get BGR color values from name or hex string."""
    # Predefined colors
    colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "green": (0, 255, 0),
    }

    # Check if it's a predefined color
    if color_name.lower() in colors:
        return colors[color_name.lower()]

    # Try to parse as hex color
    if color_name.startswith('#'):
        hex_color = color_name.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)  # OpenCV uses BGR

    # Default to white if parsing fails
    print(f"Warning: Could not parse color '{color_name}', using white")
    return (255, 255, 255)


def sharpen_frame(frame):
    """Apply sharpening filter to improve image quality."""
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply kernel
    sharpened = cv2.filter2D(frame, -1, kernel)

    # Blend with original to avoid over-sharpening
    result = cv2.addWeighted(frame, 0.5, sharpened, 0.5, 0)

    return result


def remove_background_rembg(frame, bg_color):
    """Remove background using rembg library."""
    from rembg import remove
    from PIL import Image

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Remove background (returns RGBA)
    output = remove(pil_image)
    output_np = np.array(output)

    # Extract alpha channel
    alpha = output_np[:, :, 3] / 255.0

    # Create background
    h, w = frame.shape[:2]
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Blend foreground with background
    result = np.zeros_like(frame)
    for c in range(3):
        result[:, :, c] = (alpha * output_np[:, :, c] +
                          (1 - alpha) * background[:, :, c])

    # Convert RGB back to BGR
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return result


def remove_background_grabcut(frame, bg_color):
    """Remove background using GrabCut algorithm (OpenCV built-in)."""
    h, w = frame.shape[:2]

    # Create mask
    mask = np.zeros((h, w), np.uint8)

    # Define rectangle around person (assume center region)
    margin = int(min(h, w) * 0.1)
    rect = (margin, margin, w - 2*margin, h - 2*margin)

    # Temporary arrays for GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create binary mask (0 and 2 are background, 1 and 3 are foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Create background
    background = np.full_like(frame, bg_color)

    # Blend
    result = frame * mask2[:, :, np.newaxis] + background * (1 - mask2[:, :, np.newaxis])

    return result.astype(np.uint8)


def process_video(input_path, output_path, bg_color_name, method, sharpen=False):
    """Process video frame by frame."""

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")
    print(f"Background color: {bg_color_name}")
    print(f"Method: {method}")
    print(f"Sharpening: {'enabled' if sharpen else 'disabled'}")

    # Get background color
    bg_color = get_bg_color(bg_color_name)

    # Create output video writer with high quality settings
    # Try to use H264 codec for better quality, fallback to mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    # Select background removal method
    if method == "rembg":
        remove_bg_func = remove_background_rembg
    else:
        remove_bg_func = remove_background_grabcut

    # Process frames
    print("\nProcessing frames...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Remove background and replace
        processed_frame = remove_bg_func(frame, bg_color)

        # Apply sharpening if requested
        if sharpen:
            processed_frame = sharpen_frame(processed_frame)

        # Write frame
        out.write(processed_frame)

    # Cleanup
    cap.release()
    out.release()

    print(f"\n[OK] Output saved to: {output_path}")


def main():
    args = parse_args()

    # Check if rembg is available for rembg method
    if args.method == "rembg":
        try:
            import rembg
        except ImportError:
            print("[ERROR] rembg is not installed. Install with: pip install rembg")
            print("Or use --method grabcut to use OpenCV's built-in method (lower quality)")
            return

    # Create output directory
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

    # Process video
    process_video(args.input_video, args.output_video, args.bg_color, args.method, args.sharpen)


if __name__ == "__main__":
    main()

# Example usage:
# pip install rembg
# python scripts/word/replace_video_background.py --input-video output/word_level/merged_videos_576x576/test01.mp4 --output-video output/word_level/merged_videos_576x576/test01_beige_bg.mp4 --bg-color '#CBC4B7' --sharpen
