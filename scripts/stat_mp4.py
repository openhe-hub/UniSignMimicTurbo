# save as get_video_info.py
import argparse
import os
import cv2


def get_video_info(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, frame_count


def main():
    parser = argparse.ArgumentParser(
        description="Print FPS and resolution of a video file."
    )
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    fps, w, h, n = get_video_info(args.video_path)

    print(f"Path       : {args.video_path}")
    print(f"FPS        : {fps:g}")
    print(f"Resolution : {w}x{h}")
    print(f"Frames     : {n}")


if __name__ == "__main__":
    main()
