"""
VisionPlot - Complete Curve Extraction Pipeline

Pipeline
-------
1. Preprocess image
2. Detect chart axes
3. Detect geometric ticks
4. OCR numeric tick labels
5. Clean/filter OCR labels
6. Build calibration from OCR labels
7. Detect/isolate curve
8. Remove axes/grid
9. Skeletonise curve
10. Extract pixel coordinates
11. Convert pixel coordinates -> graph coordinates
12. Resample
13. Validate
14. Return ExtractionResult

Important:
-----------
The curve extraction is NOT restricted to one quadrant.
Curves can exist:
    - left or right of the Y-axis
    - above or below the X-axis

This is important for graphs such as:
    y = 3x + 2
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from . import preprocess
from . import axes
from . import ocr
from . import calibration
from . import curve


# =====================================================================
# RESULT
# =====================================================================

@dataclass
class ExtractionResult:
    points: np.ndarray
    calibration: object
    x_confidence: float
    y_confidence: float
    warnings: list[str]

    reprojected_pixels: Optional[np.ndarray] = None

    # Extra information for dashboard/debugging
    frame: object = None
    x_ticks: list = None
    y_ticks: list = None
    x_labels: list = None
    y_labels: list = None


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def _safe_array(value, dtype=np.float64) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=dtype)

    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return np.empty((0,), dtype=dtype)


def _get_attr(obj, name, default=None):
    return getattr(obj, name, default)


# =====================================================================
# OCR LABEL NORMALISATION
# =====================================================================

def _merge_ocr_labels(labels):
    """
    Remove duplicate OCR detections.

    Expected format:

        (pixel_x, pixel_y, value, confidence)
    """

    if not labels:
        return []

    cleaned = []

    for item in labels:

        if item is None or len(item) < 3:
            continue

        try:
            px = float(item[0])
            py = float(item[1])
            value = float(item[2])

            conf = (
                float(item[3])
                if len(item) > 3
                else 1.0
            )

        except Exception:
            continue

        if not np.isfinite(px):
            continue

        if not np.isfinite(py):
            continue

        if not np.isfinite(value):
            continue

        if not np.isfinite(conf):
            conf = 0.0

        cleaned.append(
            (
                px,
                py,
                value,
                conf,
            )
        )

    cleaned.sort(
        key=lambda z: (z[0], z[1])
    )

    result = []

    for label in cleaned:

        px, py, value, conf = label

        duplicate = False

        for index, old in enumerate(result):

            opx, opy, oval, oconf = old

            if (
                abs(px - opx) <= 8.0
                and abs(py - opy) <= 8.0
            ):

                duplicate = True

                if conf > oconf:
                    result[index] = label

                break

        if not duplicate:
            result.append(label)

    return result


# =====================================================================
# OCR API COMPATIBILITY
# =====================================================================

def _read_x_labels(gray, frame):

    x_row = float(
        frame.x_axis.pixel_position
    )

    y_col = float(
        frame.y_axis.pixel_position
    )

    candidates = []

    if hasattr(
        ocr,
        "read_x_axis_labels"
    ):

        try:

            candidates = ocr.read_x_axis_labels(
                gray,
                x_row,
                y_col,
            )

        except TypeError:

            try:

                candidates = ocr.read_x_axis_labels(
                    gray,
                    x_row,
                    y_col,
                    strip_height=60,
                )

            except Exception:

                candidates = []

        except Exception:

            candidates = []

    if (
        not candidates
        and hasattr(
            ocr,
            "read_x_axis_labels_per_tick"
        )
    ):

        try:

            candidates = (
                ocr.read_x_axis_labels_per_tick(
                    gray,
                    x_row,
                    y_col,
                )
            )

        except Exception:

            candidates = []

    return _normalise_ocr_output(
        candidates
    )


def _read_y_labels(gray, frame):

    x_row = float(
        frame.x_axis.pixel_position
    )

    y_col = float(
        frame.y_axis.pixel_position
    )

    candidates = []

    if hasattr(
        ocr,
        "read_y_axis_labels"
    ):

        try:

            candidates = ocr.read_y_axis_labels(
                gray,
                x_row,
                y_col,
            )

        except TypeError:

            try:

                candidates = ocr.read_y_axis_labels(
                    gray,
                    x_row,
                    y_col,
                    strip_width=60,
                )

            except Exception:

                candidates = []

        except Exception:

            candidates = []

    if (
        not candidates
        and hasattr(
            ocr,
            "read_y_axis_labels_per_tick"
        )
    ):

        try:

            candidates = (
                ocr.read_y_axis_labels_per_tick(
                    gray,
                    x_row,
                    y_col,
                )
            )

        except Exception:

            candidates = []

    return _normalise_ocr_output(
        candidates
    )


def _normalise_ocr_output(labels):

    if labels is None:
        return []

    result = []

    for item in labels:

        # -------------------------------------------------------------
        # Tuple/list format
        # -------------------------------------------------------------

        if (
            isinstance(
                item,
                (tuple, list)
            )
            and len(item) >= 4
        ):

            try:

                px = float(item[0])
                py = float(item[1])
                value = float(item[2])
                conf = float(item[3])

                result.append(
                    (
                        px,
                        py,
                        value,
                        conf,
                    )
                )

                continue

            except Exception:

                pass

        # -------------------------------------------------------------
        # Dictionary format
        # -------------------------------------------------------------

        if isinstance(item, dict):

            try:

                px = item.get(
                    "x",
                    item.get(
                        "center_x",
                        item.get(
                            "pixel",
                            0
                        )
                    )
                )

                py = item.get(
                    "y",
                    item.get(
                        "center_y",
                        0
                    )
                )

                value = item.get(
                    "value",
                    item.get(
                        "text"
                    )
                )

                conf = item.get(
                    "confidence",
                    item.get(
                        "conf",
                        1.0
                    )
                )

                if value is None:
                    continue

                result.append(
                    (
                        float(px),
                        float(py),
                        float(value),
                        float(conf),
                    )
                )

            except Exception:

                continue

    return _merge_ocr_labels(
        result
    )


# =====================================================================
# AXIS LABEL FILTERING
# =====================================================================

def _filter_axis_labels(
    x_labels,
    y_labels,
    frame,
    image_width,
    image_height,
):

    """
    Separate X and Y OCR labels.

    X labels should normally be below X-axis.

    Y labels should normally be left of Y-axis.
    """

    x_axis_row = float(
        frame.x_axis.pixel_position
    )

    y_axis_col = float(
        frame.y_axis.pixel_position
    )

    x_clean = []
    y_clean = []

    # ================================================================
    # X LABELS
    # ================================================================

    for label in x_labels:

        px, py, value, conf = label

        below_axis = (
            py >= x_axis_row + 5.0
        )

        inside_horizontal_range = (
            px >= max(
                0.0,
                y_axis_col - 80.0
            )
            and
            px <= image_width + 30.0
        )

        if (
            below_axis
            and inside_horizontal_range
        ):

            x_clean.append(label)

    # ================================================================
    # Y LABELS
    # ================================================================

    for label in y_labels:

        px, py, value, conf = label

        left_of_axis = (
            px <= y_axis_col - 5.0
        )

        inside_vertical_range = (
            py >= -30.0
            and
            py <= min(
                image_height + 30.0,
                x_axis_row + 10.0
            )
        )

        if (
            left_of_axis
            and inside_vertical_range
        ):

            y_clean.append(label)

    x_clean.sort(
        key=lambda z: z[0]
    )

    y_clean.sort(
        key=lambda z: z[1]
    )

    x_clean = _merge_ocr_labels(
        x_clean
    )

    y_clean = _merge_ocr_labels(
        y_clean
    )

    return (
        x_clean,
        y_clean
    )


# =====================================================================
# CALIBRATION AXIS
# =====================================================================

def _make_calibration_axis(
    labels,
    orientation
):

    ticks = []

    for label in labels:

        px, py, value, conf = label

        if orientation == "x":
            pixel = float(px)
        else:
            pixel = float(py)

        tick = SimpleNamespace(
            pixel=pixel,
            value=float(value),
            confidence=float(conf),
        )

        ticks.append(tick)

    return SimpleNamespace(
        orientation=orientation,
        ticks=ticks,
        confidence=1.0,
        fit_residual=0.0,
    )


# =====================================================================
# BUILD CALIBRATION
# =====================================================================

def _build_calibration_from_labels(
    x_labels,
    y_labels,
    frame=None,
    manual=None,
):

    # ================================================================
    # MANUAL CALIBRATION
    # ================================================================

    if manual is not None:

        try:

            ax = float(
                manual["ax"]
            )

            bx = float(
                manual["bx"]
            )

            ay = float(
                manual["ay"]
            )

            by = float(
                manual["by"]
            )

            x_scale = manual.get(
                "x_scale",
                "linear"
            )

            y_scale = manual.get(
                "y_scale",
                "linear"
            )

            return SimpleNamespace(
                ax=ax,
                bx=bx,
                ay=ay,
                by=by,
                x_scale=x_scale,
                y_scale=y_scale,
            )

        except Exception:

            pass

    # ================================================================
    # VALIDATION
    # ================================================================

    if len(x_labels) < 2:

        raise ValueError(
            "Need at least 2 X-axis OCR labels; "
            f"got {len(x_labels)}."
        )

    if len(y_labels) < 2:

        raise ValueError(
            "Need at least 2 Y-axis OCR labels; "
            f"got {len(y_labels)}."
        )

    x_axis = _make_calibration_axis(
        x_labels,
        "x"
    )

    y_axis = _make_calibration_axis(
        y_labels,
        "y"
    )

    return calibration.build_calibration(
        x_axis,
        y_axis,
    )


# =====================================================================
# PIXEL -> DATA
# =====================================================================

def _pixel_to_data(
    cal,
    pixels,
    axis,
):

    pixels = np.asarray(
        pixels,
        dtype=np.float64
    )

    if axis == "x":

        transformed = (
            float(cal.ax)
            * pixels
            + float(cal.bx)
        )

        scale = getattr(
            cal,
            "x_scale",
            "linear"
        )

    elif axis == "y":

        transformed = (
            float(cal.ay)
            * pixels
            + float(cal.by)
        )

        scale = getattr(
            cal,
            "y_scale",
            "linear"
        )

    else:

        raise ValueError(
            "axis must be 'x' or 'y'"
        )

    if scale == "linear":
        return transformed

    if scale == "log10":
        return np.power(
            10.0,
            transformed
        )

    if scale == "ln":
        return np.exp(
            transformed
        )

    raise ValueError(
        f"Unknown scale: {scale}"
    )


# =====================================================================
# DATA -> PIXEL
# =====================================================================

def _data_to_pixel(
    cal,
    values,
    axis,
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if axis == "x":

        a = float(cal.ax)
        b = float(cal.bx)

        scale = getattr(
            cal,
            "x_scale",
            "linear"
        )

    elif axis == "y":

        a = float(cal.ay)
        b = float(cal.by)

        scale = getattr(
            cal,
            "y_scale",
            "linear"
        )

    else:

        raise ValueError(
            "axis must be 'x' or 'y'"
        )

    if scale == "linear":

        transformed = values

    elif scale == "log10":

        if np.any(values <= 0):
            raise ValueError(
                "Log10 coordinates must be positive."
            )

        transformed = np.log10(
            values
        )

    elif scale == "ln":

        if np.any(values <= 0):
            raise ValueError(
                "Natural-log coordinates must be positive."
            )

        transformed = np.log(
            values
        )

    else:

        raise ValueError(
            f"Unknown scale: {scale}"
        )

    if abs(a) < 1e-12:

        raise ValueError(
            "Invalid calibration: zero scale."
        )

    return (
        transformed - b
    ) / a


# =====================================================================
# RESAMPLING
# =====================================================================

def _resample_points(
    points,
    n_output_points,
    adaptive_sampling=True,
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):

        return np.empty(
            (0, 2),
            dtype=np.float64
        )

    if len(points) == 0:
        return points

    if len(points) == 1:

        return np.repeat(
            points,
            max(
                1,
                n_output_points
            ),
            axis=0
        )

    # ================================================================
    # REMOVE NON-FINITE
    # ================================================================

    valid = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[
        valid
    ]

    if len(points) < 2:
        return points

    # ================================================================
    # SORT
    # ================================================================

    order = np.argsort(
        points[:, 0]
    )

    points = points[
        order
    ]

    x = points[:, 0]
    y = points[:, 1]

    # ================================================================
    # REMOVE DUPLICATE X
    # ================================================================

    unique_x = []
    unique_y = []

    i = 0

    while i < len(x):

        j = i + 1

        while (
            j < len(x)
            and abs(
                x[j] - x[i]
            ) < 1e-10
        ):

            j += 1

        unique_x.append(
            float(
                np.mean(
                    x[i:j]
                )
            )
        )

        unique_y.append(
            float(
                np.mean(
                    y[i:j]
                )
            )
        )

        i = j

    x = np.asarray(
        unique_x,
        dtype=np.float64
    )

    y = np.asarray(
        unique_y,
        dtype=np.float64
    )

    if len(x) < 2:

        return np.column_stack(
            (x, y)
        )

    # ================================================================
    # RESAMPLE
    # ================================================================

    n = max(
        2,
        int(n_output_points)
    )

    if len(x) <= n:

        return np.column_stack(
            (x, y)
        )

    target_x = np.linspace(
        x[0],
        x[-1],
        n
    )

    target_y = np.interp(
        target_x,
        x,
        y
    )

    return np.column_stack(
        (
            target_x,
            target_y
        )
    )


# =====================================================================
# MAIN EXTRACTION
# =====================================================================

def extract_curve(
    image: np.ndarray,
    *,
    manual_calibration: Optional[dict] = None,
    curve_color_hue: Optional[int] = None,
    n_output_points: int = 200,
    adaptive_sampling: bool = True,
    **kwargs,
) -> ExtractionResult:

    warnings = []

    # ================================================================
    # 1. VALIDATE IMAGE
    # ================================================================

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    image = np.asarray(
        image
    )

    if image.ndim not in (
        2,
        3
    ):

        raise ValueError(
            f"Unsupported image shape: "
            f"{image.shape}"
        )

    if image.size == 0:

        raise ValueError(
            "Input image is empty."
        )

    # ================================================================
    # 2. PREPROCESS
    # ================================================================

    prep = preprocess.preprocess(
        image
    )

    gray = prep["gray"]
    binary = prep["binary"]
    edges = prep["edges"]

    h, w = gray.shape[:2]

    # ================================================================
    # 3. AXIS ROTATION
    # ================================================================

    try:

        rotation = (
            axes.estimate_frame_rotation(
                edges
            )
        )

    except Exception:

        rotation = 0.0

    if abs(rotation) > 2.0:

        warnings.append(
            f"Chart rotation detected: "
            f"{rotation:.2f} degrees."
        )

    # ================================================================
    # 4. LOCATE AXES
    # ================================================================

    try:

        frame = axes.locate_axes(
            binary
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not detect chart axes: "
            f"{exc}"
        )

    x_axis_row = float(
        frame.x_axis.pixel_position
    )

    y_axis_col = float(
        frame.y_axis.pixel_position
    )

    # ================================================================
    # 5. GEOMETRIC TICKS
    # ================================================================

    try:

        x_ticks, y_ticks = (
            axes.detect_ticks(
                binary,
                frame
            )
        )

    except Exception as exc:

        x_ticks = []
        y_ticks = []

        warnings.append(
            "Geometric tick detection failed: "
            f"{exc}"
        )

    try:

        frame._x_tick_pixels = x_ticks
        frame._y_tick_pixels = y_ticks

    except Exception:

        pass

    # ================================================================
    # 6. OCR
    # ================================================================

    x_labels_raw = _read_x_labels(
        gray,
        frame
    )

    y_labels_raw = _read_y_labels(
        gray,
        frame
    )

    # ================================================================
    # 7. FILTER OCR
    # ================================================================

    x_labels, y_labels = (
        _filter_axis_labels(
            x_labels_raw,
            y_labels_raw,
            frame,
            w,
            h
        )
    )

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    print()
    print("==============================")
    print("VISIONPLOT OCR DIAGNOSTICS")
    print("==============================")

    print(
        "Detected x-axis:",
        x_axis_row
    )

    print(
        "Detected y-axis:",
        y_axis_col
    )

    print(
        "Geometric X ticks:",
        x_ticks
    )

    print(
        "Geometric Y ticks:",
        y_ticks
    )

    print(
        "Raw X OCR:",
        x_labels_raw
    )

    print(
        "Raw Y OCR:",
        y_labels_raw
    )

    print(
        "Filtered X OCR:",
        x_labels
    )

    print(
        "Filtered Y OCR:",
        y_labels
    )

    # ================================================================
    # 8. CALIBRATION
    # ================================================================

    cal = None

    try:

        cal = _build_calibration_from_labels(
            x_labels,
            y_labels,
            frame=frame,
            manual=manual_calibration
        )

    except Exception as exc:

        warnings.append(
            f"Calibration failed: {exc}"
        )

        print(
            "Calibration failed:",
            exc
        )

    if cal is None:

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=None,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    # ================================================================
    # 9. CALIBRATION DIAGNOSTICS
    # ================================================================

    print()
    print("==============================")
    print("CALIBRATION")
    print("==============================")

    print(
        "X:",
        f"{float(cal.ax):.10f} * pixel + "
        f"{float(cal.bx):.10f}"
    )

    print(
        "Y:",
        f"{float(cal.ay):.10f} * pixel + "
        f"{float(cal.by):.10f}"
    )

    print(
        "X scale:",
        getattr(
            cal,
            "x_scale",
            "linear"
        )
    )

    print(
        "Y scale:",
        getattr(
            cal,
            "y_scale",
            "linear"
        )
    )

    # ================================================================
    # 10. CONVERT IMAGE TO BGR
    # ================================================================

    if image.ndim == 2:

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    elif image.shape[2] == 4:

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    else:

        bgr = image.copy()

    # ================================================================
    # 11. CURVE SEARCH REGION
    # ================================================================
    #
    # IMPORTANT FIX
    #
    # DO NOT assume the curve is only:
    #
    #     right of Y-axis
    #     above X-axis
    #
    # Your graph contains negative X and negative Y.
    #
    # Therefore the entire image is used for curve detection.
    #
    # Calibration still uses the detected axes.
    #
    # ================================================================

    u_min = 0
    v_min = 0

    u_max = w - 1
    v_max = h - 1

    plot_bounds = (
        u_min,
        v_min,
        u_max,
        v_max
    )

    print(
        "Curve search bounds:",
        plot_bounds
    )

    # ================================================================
    # 12. DETECT CURVE COLOUR
    # ================================================================

    hue = curve_color_hue

    if hue is None:

        try:

            hue = (
                curve.auto_detect_curve_hue(
                    bgr,
                    plot_bounds
                )
            )

        except Exception as exc:

            hue = None

            warnings.append(
                "Automatic curve colour detection "
                f"failed: {exc}"
            )

    print(
        "Detected curve hue:",
        hue
    )

    # ================================================================
    # 13. CURVE MASK
    # ================================================================

    if hue is not None:

        try:

            curve_mask = (
                curve.isolate_color_curve(
                    bgr,
                    int(hue)
                )
            )

        except Exception as exc:

            warnings.append(
                "Colour curve isolation failed: "
                f"{exc}"
            )

            curve_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

    else:

        # ------------------------------------------------------------
        # Fallback
        # ------------------------------------------------------------

        curve_mask = (
            (binary > 0)
            .astype(np.uint8)
            * 255
        )

        warnings.append(
            "No curve hue detected; "
            "using binary foreground."
        )

    # ================================================================
    # 14. RESTRICT MASK TO FULL IMAGE
    # ================================================================

    plot_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    plot_mask[
        v_min:v_max + 1,
        u_min:u_max + 1
    ] = 255

    curve_mask = cv2.bitwise_and(
        curve_mask,
        plot_mask
    )

    # ================================================================
    # 15. REMOVE AXES / GRID
    # ================================================================

    try:

        curve_mask = (
            curve.remove_axes_and_grid(
                curve_mask,
                x_axis_row=int(
                    round(
                        x_axis_row
                    )
                ),
                y_axis_col=int(
                    round(
                        y_axis_col
                    )
                )
            )
        )

    except Exception as exc:

        warnings.append(
            "Axis/grid removal failed: "
            f"{exc}"
        )

    # ================================================================
    # 16. CHECK MASK
    # ================================================================

    mask_pixels = int(
        np.count_nonzero(
            curve_mask
        )
    )

    print(
        "Curve mask pixels:",
        mask_pixels
    )

    if mask_pixels == 0:

        warnings.append(
            "Curve mask is empty."
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    # ================================================================
    # 17. SKELETONISE
    # ================================================================

    try:

        skeleton = curve.skeletonise(
            curve_mask
        )

    except Exception as exc:

        warnings.append(
            "Skeletonisation failed: "
            f"{exc}"
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    # ================================================================
    # 18. PRUNE BRANCHES
    # ================================================================

    try:

        skeleton = (
            curve.prune_short_branches(
                skeleton,
                min_branch_len=6
            )
        )

    except Exception as exc:

        warnings.append(
            "Skeleton branch pruning failed: "
            f"{exc}"
        )

    # ================================================================
    # 19. EXTRACT PIXEL POINTS
    # ================================================================

    try:

        pixel_x, pixel_y, sigma_y = (
            curve.extract_points_by_column(
                skeleton,
                gray,
                plot_bounds
            )
        )

    except Exception as exc:

        warnings.append(
            "Curve point extraction failed: "
            f"{exc}"
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    pixel_x = np.asarray(
        pixel_x,
        dtype=np.float64
    )

    pixel_y = np.asarray(
        pixel_y,
        dtype=np.float64
    )

    print(
        "Extracted pixel columns:",
        len(pixel_x)
    )

    # ================================================================
    # 20. VALIDATE
    # ================================================================

    if len(pixel_x) == 0:

        warnings.append(
            "No curve pixels were extracted."
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    pixel_points = np.column_stack(
        (
            pixel_x,
            pixel_y
        )
    )

    valid = np.all(
        np.isfinite(
            pixel_points
        ),
        axis=1
    )

    pixel_points = pixel_points[
        valid
    ]

    # ================================================================
    # 21. PIXEL -> GRAPH
    # ================================================================

    try:

        data_x = _pixel_to_data(
            cal,
            pixel_points[:, 0],
            "x"
        )

        data_y = _pixel_to_data(
            cal,
            pixel_points[:, 1],
            "y"
        )

        data_points = np.column_stack(
            (
                data_x,
                data_y
            )
        )

    except Exception as exc:

        warnings.append(
            "Pixel-to-data conversion failed: "
            f"{exc}"
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=pixel_points,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    # ================================================================
    # 22. REMOVE NON-FINITE
    # ================================================================

    valid = np.all(
        np.isfinite(
            data_points
        ),
        axis=1
    )

    data_points = data_points[
        valid
    ]

    corresponding_pixels = (
        pixel_points[valid]
    )

    if len(data_points) == 0:

        warnings.append(
            "No finite graph coordinates remain."
        )

        return ExtractionResult(
            points=np.empty(
                (0, 2),
                dtype=np.float64
            ),
            calibration=cal,
            x_confidence=0.0,
            y_confidence=0.0,
            warnings=warnings,
            reprojected_pixels=None,
            frame=frame,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
        )

    # ================================================================
    # 23. SORT BY X
    # ================================================================

    order = np.argsort(
        data_points[:, 0]
    )

    data_points = (
        data_points[order]
    )

    corresponding_pixels = (
        corresponding_pixels[order]
    )

    # ================================================================
    # 24. REMOVE DUPLICATE X
    # ================================================================

    unique_points = []
    unique_pixels = []

    i = 0

    while i < len(data_points):

        x_value = data_points[
            i,
            0
        ]

        j = i + 1

        while (
            j < len(data_points)
            and
            abs(
                data_points[
                    j,
                    0
                ]
                - x_value
            ) < 1e-10
        ):

            j += 1

        unique_points.append(
            np.mean(
                data_points[i:j],
                axis=0
            )
        )

        unique_pixels.append(
            np.mean(
                corresponding_pixels[i:j],
                axis=0
            )
        )

        i = j

    data_points = np.asarray(
        unique_points,
        dtype=np.float64
    )

    corresponding_pixels = np.asarray(
        unique_pixels,
        dtype=np.float64
    )

    # ================================================================
    # 25. RESAMPLE
    # ================================================================

    sampled_points = (
        _resample_points(
            data_points,
            n_output_points=n_output_points,
            adaptive_sampling=adaptive_sampling
        )
    )

    # ================================================================
    # 26. REPROJECT DATA -> PIXELS
    # ================================================================

    try:

        reproj_x = _data_to_pixel(
            cal,
            sampled_points[:, 0],
            "x"
        )

        reproj_y = _data_to_pixel(
            cal,
            sampled_points[:, 1],
            "y"
        )

        reprojected_pixels = np.column_stack(
            (
                reproj_x,
                reproj_y
            )
        )

    except Exception as exc:

        warnings.append(
            "Back-projection failed: "
            f"{exc}"
        )

        reprojected_pixels = None

    # ================================================================
    # 27. CONFIDENCE
    # ================================================================

    #
    # This is based on number of usable OCR labels.
    # It is NOT a fabricated physical accuracy measurement.
    #

    x_confidence = float(
        min(
            1.0,
            len(x_labels) / 4.0
        )
    )

    y_confidence = float(
        min(
            1.0,
            len(y_labels) / 4.0
        )
    )

    # ================================================================
    # 28. FINAL DIAGNOSTICS
    # ================================================================

    print()
    print("==============================")
    print("EXTRACTION RESULT")
    print("==============================")

    print(
        "Pixel points:",
        len(pixel_points)
    )

    print(
        "Graph points:",
        len(sampled_points)
    )

    if len(sampled_points) > 0:

        print(
            "First graph point:",
            sampled_points[0]
        )

        print(
            "Last graph point:",
            sampled_points[-1]
        )

        print(
            "X range:",
            float(
                np.min(
                    sampled_points[:, 0]
                )
            ),
            "to",
            float(
                np.max(
                    sampled_points[:, 0]
                )
            )
        )

        print(
            "Y range:",
            float(
                np.min(
                    sampled_points[:, 1]
                )
            ),
            "to",
            float(
                np.max(
                    sampled_points[:, 1]
                )
            )
        )

    print(
        "X confidence:",
        x_confidence
    )

    print(
        "Y confidence:",
        y_confidence
    )

    print(
        "Warnings:",
        warnings
    )

    # ================================================================
    # 29. RETURN
    # ================================================================

    return ExtractionResult(

        points=np.asarray(
            sampled_points,
            dtype=np.float64
        ),

        calibration=cal,

        x_confidence=x_confidence,

        y_confidence=y_confidence,

        warnings=warnings,

        reprojected_pixels=reprojected_pixels,

        frame=frame,

        x_ticks=x_ticks,

        y_ticks=y_ticks,

        x_labels=x_labels,

        y_labels=y_labels,
    )