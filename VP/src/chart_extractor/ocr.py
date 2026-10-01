"""
Stage 3a: OCR of numeric tick labels.

VisionPlot OCR module
- Detects numeric labels on X and Y axes
- Automatically finds Tesseract
- Uses multiple preprocessing variants
- Uses multiple Tesseract PSM modes
- Handles decimal and negative values
- Supports per-tick OCR
"""

from __future__ import annotations

import os
import re
import shutil
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


# ============================================================
# TESSERACT SETUP
# ============================================================

def _find_tesseract() -> Optional[str]:
    """Find Tesseract executable."""

    # First try Windows PATH
    found = shutil.which("tesseract")

    if found:
        return found

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
        ),
        os.path.expanduser(
            r"~\Tesseract-OCR\tesseract.exe"
        ),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


if pytesseract is not None:

    _tesseract = _find_tesseract()

    if _tesseract:
        pytesseract.pytesseract.tesseract_cmd = _tesseract


# ============================================================
# NUMBER PARSING
# ============================================================

_NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w.])
    [-+]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][-+]?\d+)?
    (?![\w.])
    """,
    re.VERBOSE,
)


def _parse_number(text: str) -> Optional[float]:
    """
    Extract a numeric value from OCR text.
    """

    if not text:
        return None

    text = str(text).strip()

    # Common OCR mistakes
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        ",": ".",
    }

    cleaned = text

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    match = _NUMBER_PATTERN.search(cleaned)

    if not match:
        return None

    try:
        value = float(match.group())
    except ValueError:
        return None

    if not np.isfinite(value):
        return None

    return value


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _make_variants(image: np.ndarray) -> List[np.ndarray]:
    """
    Generate multiple OCR-friendly image variants.
    """

    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    # Upscale small chart labels
    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )

    variants = []

    # 1. Grayscale
    variants.append(gray)

    # 2. Histogram equalisation
    equalized = cv2.equalizeHist(gray)
    variants.append(equalized)

    # 3. Otsu binary
    _, otsu = cv2.threshold(
        equalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    variants.append(otsu)

    # 4. Inverted Otsu
    _, otsu_inv = cv2.threshold(
        equalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    variants.append(otsu_inv)

    # 5. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    variants.append(adaptive)

    # 6. Gaussian blur + Otsu
    blur = cv2.GaussianBlur(
        equalized,
        (3, 3),
        0,
    )

    _, blur_otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants.append(blur_otsu)

    return variants


# ============================================================
# SINGLE OCR
# ============================================================

def _ocr_single(
    image: np.ndarray,
    psm: int = 7,
) -> List[Tuple[str, float]]:
    """
    Run Tesseract on one image.

    Returns:
        [(text, confidence), ...]
    """

    if pytesseract is None:
        return []

    if image is None or image.size == 0:
        return []

    config = (
        f"--psm {psm} "
        "--oem 3 "
        "-c tessedit_char_whitelist=0123456789.-+eE"
    )

    try:

        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

    except Exception:
        return []

    results = []

    texts = data.get("text", [])
    confidences = data.get("conf", [])

    for text, conf in zip(
        texts,
        confidences,
    ):

        text = str(text).strip()

        if not text:
            continue

        try:
            confidence = float(conf)
        except (ValueError, TypeError):
            confidence = 0.0

        value = _parse_number(text)

        if value is not None:
            results.append(
                (
                    text,
                    confidence,
                )
            )

    return results


# ============================================================
# BEST OCR VALUE
# ============================================================

def _best_ocr_value(
    image: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Try multiple preprocessing variants and PSM modes.

    Returns:
        (value, confidence)
    """

    variants = _make_variants(image)

    if not variants:
        return None

    # Tesseract page segmentation modes
    psms = [
        7,   # single line
        8,   # single word
        6,   # block
        10,  # single character
    ]

    candidates = []

    for variant in variants:

        for psm in psms:

            results = _ocr_single(
                variant,
                psm=psm,
            )

            for text, confidence in results:

                value = _parse_number(text)

                if value is None:
                    continue

                candidates.append(
                    (
                        value,
                        confidence,
                    )
                )

    if not candidates:
        return None

    # Highest-confidence result
    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# OCR REGION
# ============================================================

def _ocr_region(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    OCR a rectangular region.

    Returns:
        (center_x, center_y, value, confidence)
    """

    h, w = image.shape[:2]

    x1 = max(
        0,
        min(w - 1, int(x1))
    )

    x2 = max(
        0,
        min(w, int(x2))
    )

    y1 = max(
        0,
        min(h - 1, int(y1))
    )

    y2 = max(
        0,
        min(h, int(y2))
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    result = _best_ocr_value(crop)

    if result is None:
        return None

    value, confidence = result

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    return (
        center_x,
        center_y,
        value,
        confidence,
    )


# ============================================================
# X AXIS GENERAL OCR
# ============================================================

def read_x_axis_labels(
    image: np.ndarray,
    frame=None,
    x_ticks=None,
) -> List[Tuple[float, float, float, float]]:
    """
    Detect numeric labels below the X axis.

    Compatible with:
        read_x_axis_labels(image)
        read_x_axis_labels(image, frame)
        read_x_axis_labels(image, frame, x_ticks)

    Returns:
        [(x, y, value, confidence), ...]
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Find X axis location
    # --------------------------------------------------------

    if frame is not None:

        try:
            axis_y = float(frame.x_axis)

        except Exception:

            try:
                axis_y = float(frame.x_axis_y)

            except Exception:
                axis_y = h * 0.75

    else:
        axis_y = h * 0.75

    # --------------------------------------------------------
    # Region below X axis
    # --------------------------------------------------------

    y_start = int(
        max(
            0,
            min(
                h - 1,
                axis_y + 5
            )
        )
    )

    y_end = int(
        min(
            h,
            y_start + h * 0.25
        )
    )

    if y_end <= y_start:

        y_start = int(
            h * 0.60
        )

        y_end = h

    region = image[
        y_start:y_end,
        :
    ]

    if region.size == 0:
        return []

    variants = _make_variants(region)

    detections = []

    for variant in variants:

        for psm in [6, 11, 12]:

            if pytesseract is None:
                continue

            config = (
                f"--psm {psm} "
                "--oem 3 "
                "-c tessedit_char_whitelist=0123456789.-+eE"
            )

            try:

                data = pytesseract.image_to_data(
                    variant,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )

            except Exception:
                continue

            texts = data.get(
                "text",
                []
            )

            confs = data.get(
                "conf",
                []
            )

            xs = data.get(
                "left",
                []
            )

            ys = data.get(
                "top",
                []
            )

            ws = data.get(
                "width",
                []
            )

            hs = data.get(
                "height",
                []
            )

            for (
                text,
                conf,
                xx,
                yy,
                ww,
                hh,
            ) in zip(
                texts,
                confs,
                xs,
                ys,
                ws,
                hs,
            ):

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                value = _parse_number(
                    text
                )

                if value is None:
                    continue

                try:
                    confidence = float(
                        conf
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    confidence = 0.0

                # Tesseract operated on 3x image
                center_x = (
                    xx / 3.0
                    + ww / 6.0
                )

                center_y = (
                    yy / 3.0
                    + hh / 6.0
                    + y_start
                )

                detections.append(
                    (
                        float(center_x),
                        float(center_y),
                        float(value),
                        float(confidence),
                    )
                )

    return _deduplicate_labels(
        detections
    )


# ============================================================
# X AXIS PER-TICK OCR
# ============================================================

def read_x_axis_labels_per_tick(
    image: np.ndarray,
    frame=None,
    x_ticks=None,
) -> List[Tuple[float, float, float, float]]:
    """
    OCR each X-axis tick individually.

    Compatible with:
        read_x_axis_labels_per_tick(image, frame, x_ticks)

    Returns:
        [(x, y, value, confidence), ...]
    """

    h, w = image.shape[:2]

    results = []

    if x_ticks is None:
        return results

    # --------------------------------------------------------
    # Find X axis
    # --------------------------------------------------------

    if frame is not None:

        try:
            axis_y = float(
                frame.x_axis
            )

        except Exception:

            try:
                axis_y = float(
                    frame.x_axis_y
                )

            except Exception:
                axis_y = h * 0.75

    else:
        axis_y = h * 0.75

    # --------------------------------------------------------
    # OCR around every tick
    # --------------------------------------------------------

    for tick in x_ticks:

        try:
            tx = float(tick)

        except (
            TypeError,
            ValueError,
        ):
            continue

        x1 = int(
            tx - 50
        )

        x2 = int(
            tx + 50
        )

        y1 = int(
            axis_y + 2
        )

        y2 = int(
            axis_y + min(
                90,
                h * 0.18
            )
        )

        result = _ocr_region(
            image,
            x1,
            y1,
            x2,
            y2,
        )

        if result is not None:
            results.append(
                result
            )

    return _deduplicate_labels(
        results
    )


# ============================================================
# Y AXIS GENERAL OCR
# ============================================================

def read_y_axis_labels(
    image: np.ndarray,
    frame=None,
    y_ticks=None,
) -> List[Tuple[float, float, float, float]]:
    """
    Detect numeric labels to the left of the Y axis.

    Compatible with:
        read_y_axis_labels(image)
        read_y_axis_labels(image, frame)
        read_y_axis_labels(image, frame, y_ticks)

    Returns:
        [(x, y, value, confidence), ...]
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Find Y axis
    # --------------------------------------------------------

    if frame is not None:

        try:
            axis_x = float(
                frame.y_axis
            )

        except Exception:

            try:
                axis_x = float(
                    frame.y_axis_x
                )

            except Exception:
                axis_x = w * 0.20

    else:
        axis_x = w * 0.20

    # --------------------------------------------------------
    # Region to left of Y axis
    # --------------------------------------------------------

    x_end = int(
        max(
            1,
            min(
                w,
                axis_x - 5
            )
        )
    )

    x_start = int(
        max(
            0,
            x_end - w * 0.30
        )
    )

    if x_end <= x_start:

        x_start = 0
        x_end = int(
            w * 0.35
        )

    region = image[
        :,
        x_start:x_end
    ]

    if region.size == 0:
        return []

    variants = _make_variants(
        region
    )

    detections = []

    # --------------------------------------------------------
    # Normal horizontal OCR
    # --------------------------------------------------------

    for variant in variants:

        for psm in [6, 11, 12]:

            if pytesseract is None:
                continue

            config = (
                f"--psm {psm} "
                "--oem 3 "
                "-c tessedit_char_whitelist=0123456789.-+eE"
            )

            try:

                data = pytesseract.image_to_data(
                    variant,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )

            except Exception:
                continue

            texts = data.get(
                "text",
                []
            )

            confs = data.get(
                "conf",
                []
            )

            xs = data.get(
                "left",
                []
            )

            ys = data.get(
                "top",
                []
            )

            ws = data.get(
                "width",
                []
            )

            hs = data.get(
                "height",
                []
            )

            for (
                text,
                conf,
                xx,
                yy,
                ww,
                hh,
            ) in zip(
                texts,
                confs,
                xs,
                ys,
                ws,
                hs,
            ):

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                value = _parse_number(
                    text
                )

                if value is None:
                    continue

                try:
                    confidence = float(
                        conf
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    confidence = 0.0

                center_x = (
                    xx / 3.0
                    + ww / 6.0
                    + x_start
                )

                center_y = (
                    yy / 3.0
                    + hh / 6.0
                )

                detections.append(
                    (
                        float(center_x),
                        float(center_y),
                        float(value),
                        float(confidence),
                    )
                )

    # --------------------------------------------------------
    # Rotated fallback
    # --------------------------------------------------------

    if len(detections) < 2:

        rotated = cv2.rotate(
            region,
            cv2.ROTATE_90_CLOCKWISE
        )

        rotated_variants = _make_variants(
            rotated
        )

        for variant in rotated_variants:

            for psm in [6, 7, 11, 12]:

                if pytesseract is None:
                    continue

                config = (
                    f"--psm {psm} "
                    "--oem 3 "
                    "-c tessedit_char_whitelist=0123456789.-+eE"
                )

                try:

                    data = pytesseract.image_to_data(
                        variant,
                        config=config,
                        output_type=pytesseract.Output.DICT,
                    )

                except Exception:
                    continue

                texts = data.get(
                    "text",
                    []
                )

                confs = data.get(
                    "conf",
                    []
                )

                xs = data.get(
                    "left",
                    []
                )

                ys = data.get(
                    "top",
                    []
                )

                ws = data.get(
                    "width",
                    []
                )

                hs = data.get(
                    "height",
                    []
                )

                for (
                    text,
                    conf,
                    xx,
                    yy,
                    ww,
                    hh,
                ) in zip(
                    texts,
                    confs,
                    xs,
                    ys,
                    ws,
                    hs,
                ):

                    text = str(
                        text
                    ).strip()

                    if not text:
                        continue

                    value = _parse_number(
                        text
                    )

                    if value is None:
                        continue

                    try:
                        confidence = float(
                            conf
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):
                        confidence = 0.0

                    rcx = (
                        xx / 3.0
                        + ww / 6.0
                    )

                    rcy = (
                        yy / 3.0
                        + hh / 6.0
                    )

                    # Reverse clockwise rotation
                    original_x = (
                        rcy
                        + x_start
                    )

                    original_y = (
                        region.shape[1]
                        - rcx
                    )

                    detections.append(
                        (
                            float(original_x),
                            float(original_y),
                            float(value),
                            float(confidence),
                        )
                    )

    return _deduplicate_labels(
        detections
    )


# ============================================================
# Y AXIS PER-TICK OCR
# ============================================================

def read_y_axis_labels_per_tick(
    image: np.ndarray,
    frame=None,
    y_ticks=None,
) -> List[Tuple[float, float, float, float]]:
    """
    OCR each Y-axis tick individually.

    Compatible with:
        read_y_axis_labels_per_tick(image, frame, y_ticks)

    Returns:
        [(x, y, value, confidence), ...]
    """

    h, w = image.shape[:2]

    results = []

    if y_ticks is None:
        return results

    # --------------------------------------------------------
    # Find Y axis
    # --------------------------------------------------------

    if frame is not None:

        try:
            axis_x = float(
                frame.y_axis
            )

        except Exception:

            try:
                axis_x = float(
                    frame.y_axis_x
                )

            except Exception:
                axis_x = w * 0.20

    else:
        axis_x = w * 0.20

    # --------------------------------------------------------
    # First attempt: normal crop
    # --------------------------------------------------------

    for tick in y_ticks:

        try:
            ty = float(tick)

        except (
            TypeError,
            ValueError,
        ):
            continue

        x1 = int(
            axis_x - min(
                110,
                w * 0.20
            )
        )

        x2 = int(
            axis_x - 2
        )

        y1 = int(
            ty - 35
        )

        y2 = int(
            ty + 35
        )

        result = _ocr_region(
            image,
            x1,
            y1,
            x2,
            y2,
        )

        if result is not None:
            results.append(
                result
            )

    # --------------------------------------------------------
    # Wider fallback
    # --------------------------------------------------------

    if len(results) < 2:

        results = []

        for tick in y_ticks:

            try:
                ty = float(tick)

            except (
                TypeError,
                ValueError,
            ):
                continue

            x1 = int(
                axis_x - min(
                    180,
                    w * 0.32
                )
            )

            x2 = int(
                axis_x - 1
            )

            y1 = int(
                ty - 45
            )

            y2 = int(
                ty + 45
            )

            result = _ocr_region(
                image,
                x1,
                y1,
                x2,
                y2,
            )

            if result is not None:
                results.append(
                    result
                )

    return _deduplicate_labels(
        results
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate_labels(
    labels: List[
        Tuple[
            float,
            float,
            float,
            float
        ]
    ],
) -> List[
    Tuple[
        float,
        float,
        float,
        float
    ]
]:
    """
    Remove duplicate OCR detections.
    """

    if not labels:
        return []

    # Highest confidence first
    labels = sorted(
        labels,
        key=lambda item: item[3],
        reverse=True,
    )

    kept = []

    for candidate in labels:

        cx, cy, value, confidence = candidate

        duplicate = False

        for existing in kept:

            ex, ey, ev, ec = existing

            # Same location
            if (
                abs(cx - ex) < 25
                and
                abs(cy - ey) < 25
            ):

                # Same numeric value
                if abs(value - ev) < 1e-9:

                    duplicate = True
                    break

                # Almost same location
                if (
                    abs(cx - ex) < 12
                    and
                    abs(cy - ey) < 12
                ):

                    duplicate = True
                    break

        if not duplicate:
            kept.append(
                candidate
            )

    # Sort spatially
    kept.sort(
        key=lambda item: (
            item[1],
            item[0]
        )
    )

    return kept


# ============================================================
# TESSERACT STATUS
# ============================================================

def tesseract_available() -> bool:
    """
    Check whether Tesseract is available.
    """

    if pytesseract is None:
        return False

    try:

        pytesseract.get_tesseract_version()

        return True

    except Exception:

        return False