"""Evaluate PR 5627 camera-orientation ONNX on straight-walk direction samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_DATASET = Path(r"D:/kang/maaend/test/map_direction_samples/20260905_185001")
DEFAULT_MODEL = Path(r"D:/kang/maaend/MaaEnd/assets/resource/model/map/cameraorientation.onnx")
DEFAULT_OUTPUT = Path(r"D:/kang/maaend/test/pr5627_camera_orientation_report")
IMAGE_WIDTH, IMAGE_HEIGHT = 360, 42
INNER_RADIUS, OUTER_RADIUS = 12.0, 54.0
BASE_CENTER = (108.0, 111.0)
REFINE_RADIUS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_error(predicted: float, target: float) -> float:
    return (predicted - target + 180.0) % 360.0 - 180.0


def polar_unwrap(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale_x = width / 1280.0
    scale_y = height / 720.0
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.01:
        raise ValueError(f"non-uniform frame scale: {scale_x:.4f} x {scale_y:.4f}")
    cx, cy = BASE_CENTER[0] * scale_x, BASE_CENTER[1] * scale_y
    r_in, r_out = INNER_RADIUS * scale_x, OUTER_RADIUS * scale_x
    if not (r_out + 1 <= cx <= width - r_out - 1 and r_out + 1 <= cy <= height - r_out - 1):
        raise ValueError(f"frame {width}x{height} is too small for the orientation ring")
    step = (r_out - r_in) / IMAGE_HEIGHT
    radii = r_in + step * (np.arange(IMAGE_HEIGHT, dtype=np.float32) + 0.5)
    theta = np.deg2rad(np.arange(IMAGE_WIDTH, dtype=np.float32))
    map_x = (cx + radii[:, None] * np.sin(theta)[None, :]).astype(np.float32)
    map_y = (cy - radii[:, None] * np.cos(theta)[None, :]).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def decode_pmf(pmf: np.ndarray) -> tuple[float, float, int]:
    center = int(np.argmax(pmf))
    columns = (center + np.arange(-REFINE_RADIUS, REFINE_RADIUS + 1)) % IMAGE_WIDTH
    radians = np.deg2rad(columns)
    decoded = (
        math.degrees(
            math.atan2(
                float(np.sum(pmf[columns] * np.sin(radians))),
                float(np.sum(pmf[columns] * np.cos(radians))),
            )
        )
        % 360.0
    )
    all_radians = np.deg2rad(np.arange(IMAGE_WIDTH, dtype=np.float64))
    sin_sum = float(np.sum(pmf * np.sin(all_radians)))
    cos_sum = float(np.sum(pmf * np.cos(all_radians)))
    resultant_angle = math.degrees(math.atan2(sin_sum, cos_sum))
    confidence = float(
        np.clip(
            math.hypot(sin_sum, cos_sum) * math.cos(math.radians(abs(decoded - resultant_angle))),
            0.0,
            1.0,
        )
    )
    return decoded, confidence, center


def summarize(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "signed_mean_deg": float(values.mean()),
        "mae_deg": float(absolute.mean()),
        "rmse_deg": float(np.sqrt(np.mean(values * values))),
        "median_abs_deg": float(np.median(absolute)),
        "p90_abs_deg": float(np.percentile(absolute, 90)),
        "p95_abs_deg": float(np.percentile(absolute, 95)),
        "max_abs_deg": float(absolute.max()),
        "within_1_deg": float(np.mean(absolute <= 1.0)),
        "within_2_deg": float(np.mean(absolute <= 2.0)),
        "within_5_deg": float(np.mean(absolute <= 5.0)),
        "within_10_deg": float(np.mean(absolute <= 10.0)),
    }


def choose_example(records: list[dict[str, float | str]]) -> dict[str, float | str]:
    median = float(np.median([abs(float(row["model_error_deg"])) for row in records]))
    return min(
        records,
        key=lambda row: (
            abs(abs(float(row["model_error_deg"])) - median),
            -float(row["confidence"]),
        ),
    )


def draw_direction_line(
    image: np.ndarray,
    center: tuple[int, int],
    angle: float,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    radians = math.radians(angle)
    end = (
        int(round(center[0] + math.sin(radians) * radius)),
        int(round(center[1] - math.cos(radians) * radius)),
    )
    cv2.arrowedLine(image, center, end, color, thickness, cv2.LINE_AA, tipLength=0.20)


def draw_pmf_plot(pmf: np.ndarray, truth: float, prediction: float) -> np.ndarray:
    width, height = 720, 220
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0, x1, y1 = 52, 18, width - 18, height - 34
    y_max = max(float(pmf.max()) * 1.15, 1e-6)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(y1 - fraction * (y1 - y0)))
        cv2.line(canvas, (x0, y), (x1, y), (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{fraction * y_max:.2f}",
            (4, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    for degree in range(0, 361, 45):
        x = int(round(x0 + degree / 360.0 * (x1 - x0)))
        cv2.line(canvas, (x, y1), (x, y1 + 5), (150, 150, 150), 1, cv2.LINE_AA)
        label = str(degree)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(
            canvas,
            label,
            (x - text_width // 2, y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
    values = np.append(pmf, pmf[0])
    xs = x0 + np.arange(361) * ((x1 - x0) / 360.0)
    ys = y1 - values / y_max * (y1 - y0)
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [points], False, (0, 255, 255), 2, cv2.LINE_AA)
    truth_x = int(round(x0 + truth / 360.0 * (x1 - x0)))
    prediction_x = int(round(x0 + prediction / 360.0 * (x1 - x0)))
    cv2.line(canvas, (truth_x, y0), (truth_x, y1), (80, 220, 80), 2, cv2.LINE_AA)
    cv2.line(canvas, (prediction_x, y0), (prediction_x, y1), (80, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, "PMF", (x0, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "walk heading",
        (x0 + 70, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 220, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "model",
        (x0 + 220, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_example(
    dataset: Path,
    record: dict[str, float | str],
    strip: np.ndarray,
    pmf: np.ndarray,
    output: Path,
) -> None:
    frame = cv2.imread(str(dataset / str(record["frame"])), cv2.IMREAD_COLOR)
    minimap = frame[51:171, 49:167].copy()
    scale = 4
    minimap = cv2.resize(minimap, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((700, 1420, 3), 22, dtype=np.uint8)
    left_x, top_y = 48, 96
    canvas[top_y : top_y + minimap.shape[0], left_x : left_x + minimap.shape[1]] = minimap
    center = (left_x + 59 * scale, top_y + 60 * scale)
    cv2.circle(canvas, center, int(54 * scale), (80, 80, 80), 2, cv2.LINE_AA)
    cv2.circle(canvas, center, int(12 * scale), (80, 80, 80), 2, cv2.LINE_AA)
    draw_direction_line(
        canvas, center, float(record["heading_deg"]), int(54 * scale), (80, 220, 80), 4
    )
    draw_direction_line(
        canvas, center, float(record["predicted_deg"]), int(54 * scale), (80, 80, 255), 3
    )
    cv2.putText(
        canvas,
        f"Example: {record['frame']}",
        (left_x, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"walk heading={float(record['heading_deg']):.3f} deg | "
            f"model={float(record['predicted_deg']):.3f} deg | "
            f"error={float(record['model_error_deg']):+.3f} deg | "
            f"conf={float(record['confidence']):.3f}"
        ),
        (left_x, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    strip_view = cv2.resize(strip, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_NEAREST)
    strip_x, strip_y = 640, 96
    canvas[strip_y : strip_y + strip_view.shape[0], strip_x : strip_x + strip_view.shape[1]] = (
        strip_view
    )
    cv2.putText(
        canvas,
        "polar strip (360 x 42, 1 deg/column)",
        (strip_x, strip_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    plot = draw_pmf_plot(pmf, float(record["heading_deg"]), float(record["predicted_deg"]))
    plot_x, plot_y = strip_x, 240
    canvas[plot_y : plot_y + plot.shape[0], plot_x : plot_x + plot.shape[1]] = plot
    cv2.putText(
        canvas,
        "walk heading",
        (strip_x, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 220, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "model prediction",
        (strip_x + 150, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "model PMF",
        (strip_x + 330, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Example is cropped to the minimap ROI; no UID or full game frame is included.",
        (left_x, 670),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.dataset / "labels.jsonl"
    rows = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no labels found: {labels_path}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    records: list[dict[str, float | str]] = []
    strips: dict[str, np.ndarray] = {}
    pmfs: dict[str, np.ndarray] = {}
    for row in rows:
        frame_path = args.dataset / row["frame"]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read {frame_path}")
        strip = polar_unwrap(frame)
        pmf = session.run([output_name], {input_name: strip[None, ...]})[0][0]
        predicted, confidence, center = decode_pmf(pmf)
        truth = float(row["heading_deg"])
        records.append(
            {
                "frame": row["frame"],
                "heading_deg": truth,
                "predicted_deg": predicted,
                "model_error_deg": signed_error(predicted, truth),
                "confidence": confidence,
                "pmf_center": center,
                "end_raw_rot_deg": float(row["end_raw"]["rot"]),
                "end_raw_rot_error_deg": signed_error(float(row["end_raw"]["rot"]), truth),
                "distance_px": float(row["distance_px"]),
            }
        )
        strips[row["frame"]] = strip
        pmfs[row["frame"]] = pmf

    example = choose_example(records)
    render_example(
        args.dataset,
        example,
        strips[str(example["frame"])],
        pmfs[str(example["frame"])],
        args.output_dir / "example.png",
    )

    model_errors = [float(row["model_error_deg"]) for row in records]
    arrow_errors = [float(row["end_raw_rot_error_deg"]) for row in records]
    direct_errors = [
        signed_error(float(row["predicted_deg"]), float(row["end_raw_rot_deg"])) for row in records
    ]
    confidence = np.asarray([float(row["confidence"]) for row in records])
    metrics = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "samples": len(records),
        "input_contract": {
            "name": input_name,
            "type": session.get_inputs()[0].type,
            "shape": session.get_inputs()[0].shape,
        },
        "preprocessing": {
            "center_720p": list(BASE_CENTER),
            "inner_radius_720p": INNER_RADIUS,
            "outer_radius_720p": OUTER_RADIUS,
            "output_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "color_order": "BGR",
            "angle_axis": "1 degree per column, north=0, clockwise",
        },
        "camera_model_vs_walk_heading": summarize(model_errors),
        "existing_end_raw_rot_vs_walk_heading": summarize(arrow_errors),
        "camera_model_vs_existing_end_raw_rot": summarize(direct_errors),
        "model_confidence": {
            "mean": float(confidence.mean()),
            "median": float(np.median(confidence)),
            "min": float(confidence.min()),
            "max": float(confidence.max()),
        },
        "model_error_histogram_deg": [
            {
                "lower": lower,
                "upper": upper,
                "count": int(sum(lower <= abs(error) < upper for error in model_errors)),
            }
            for lower, upper in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 180))
        ],
        "example": example,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"samples={metrics['samples']}")
    print(f"model_MAE={metrics['camera_model_vs_walk_heading']['mae_deg']:.4f} deg")
    print(f"model_RMSE={metrics['camera_model_vs_walk_heading']['rmse_deg']:.4f} deg")
    print(f"model_within_5={metrics['camera_model_vs_walk_heading']['within_5_deg']:.2%}")
    print(
        f"example={example['frame']} "
        f"error={example['model_error_deg']:+.4f} deg "
        f"conf={example['confidence']:.4f}"
    )


if __name__ == "__main__":
    main()
