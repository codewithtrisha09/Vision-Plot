"""
VisionPlot
==========

Interactive graph-image-to-data extraction dashboard.

Features:
- Original image
- Detected curve overlay
- Extracted mathematical graph
- Polynomial curve fitting
- R² / RMSE / MAE
- Calibration information
- Detected axes and ticks
- OCR labels
- CSV export
- PNG export
- SVG export
- Persistent extraction using Streamlit session state
"""

from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


# ================================================================
# IMPORT PIPELINE
# ================================================================

try:
    from src.chart_extractor.pipeline import extract_curve
except Exception:
    from chart_extractor.pipeline import extract_curve


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="VisionPlot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# CSS
# ================================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-top: -5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .step-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #30343b;
        background: #151820;
        text-align: center;
    }

    .step-number {
        font-size: 25px;
        font-weight: 700;
    }

    .step-text {
        color: #aeb4c0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">📈 VisionPlot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Image → Computer Vision → OCR → Calibration → Data"
    "</div>",
    unsafe_allow_html=True,
)


# ================================================================
# PIPELINE STEPS
# ================================================================

steps = [
    ("1", "Upload"),
    ("2", "Preprocess"),
    ("3", "Axes"),
    ("4", "OCR"),
    ("5", "Calibrate"),
    ("6", "Extract"),
    ("7", "Data"),
]

cols = st.columns(len(steps))

for col, (number, name) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number">{number}</div>
                <div class="step-text">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# SESSION STATE
# ================================================================

if "visionplot_result" not in st.session_state:
    st.session_state.visionplot_result = None

if "visionplot_image" not in st.session_state:
    st.session_state.visionplot_image = None

if "visionplot_file_hash" not in st.session_state:
    st.session_state.visionplot_file_hash = None

if "visionplot_filename" not in st.session_state:
    st.session_state.visionplot_filename = None


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.header("⚙️ Extraction Settings")

n_points = st.sidebar.slider(
    "Output points",
    min_value=20,
    max_value=1000,
    value=200,
    step=10,
)

adaptive_sampling = st.sidebar.checkbox(
    "Adaptive sampling",
    value=True,
)

st.sidebar.markdown("---")

st.sidebar.subheader("Curve colour")

use_manual_hue = st.sidebar.checkbox(
    "Set curve hue manually",
    value=False,
)

manual_hue = None

if use_manual_hue:
    manual_hue = st.sidebar.slider(
        "HSV Hue",
        min_value=0,
        max_value=179,
        value=60,
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Extraction settings are used when "
    "you click Extract Graph Data."
)


# ================================================================
# UPLOAD
# ================================================================

uploaded_file = st.file_uploader(
    "Upload a graph image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
    ],
)


# ================================================================
# NO FILE
# ================================================================

if uploaded_file is None:

    st.info(
        "Upload a graph image to begin."
    )

    st.markdown(
        """
        ### What VisionPlot does

        **Input**

        A graph/chart image.

        **Processing**

        - Image preprocessing
        - Axis detection
        - Tick detection
        - OCR
        - Pixel-to-data calibration
        - Curve extraction
        - Curve sampling

        **Output**

        - Extracted X,Y data
        - Detected curve overlay
        - Mathematical graph
        - Polynomial equation
        - R²
        - RMSE
        - MAE
        - CSV
        - PNG
        - SVG
        """
    )

    st.stop()


# ================================================================
# READ UPLOADED FILE
# ================================================================

try:

    file_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

except Exception as exc:

    st.error(
        f"Could not read uploaded file: {exc}"
    )

    st.stop()


# ================================================================
# RESET OLD RESULT WHEN A NEW IMAGE IS UPLOADED
# ================================================================

if (
    st.session_state.visionplot_file_hash
    != current_file_hash
):

    st.session_state.visionplot_result = None

    st.session_state.visionplot_image = None

    st.session_state.visionplot_filename = (
        uploaded_file.name
    )

    st.session_state.visionplot_file_hash = (
        current_file_hash
    )


# ================================================================
# LOAD IMAGE
# ================================================================

try:

    image_pil = Image.open(
        io.BytesIO(file_bytes)
    ).convert("RGB")

    image_rgb = np.asarray(
        image_pil
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

except Exception as exc:

    st.error(
        f"Could not decode the image: {exc}"
    )

    st.stop()


# ================================================================
# INPUT IMAGE
# ================================================================

st.markdown(
    '<div class="section-title">🖼️ Input Image</div>',
    unsafe_allow_html=True,
)

st.image(
    image_rgb,
    use_container_width=True,
)


# ================================================================
# EXTRACT BUTTON
# ================================================================

extract_clicked = st.button(
    "🚀 Extract Graph Data",
    type="primary",
    use_container_width=True,
)


# ================================================================
# RUN EXTRACTION ONLY WHEN BUTTON IS CLICKED
# ================================================================

if extract_clicked:

    with st.spinner(
        "Running VisionPlot pipeline..."
    ):

        try:

            result = extract_curve(
                image_bgr,
                curve_color_hue=manual_hue,
                n_output_points=int(n_points),
                adaptive_sampling=adaptive_sampling,
            )

            # ----------------------------------------------------
            # SAVE RESULT
            # ----------------------------------------------------

            st.session_state.visionplot_result = result

            st.session_state.visionplot_image = (
                image_rgb.copy()
            )

            st.session_state.visionplot_filename = (
                uploaded_file.name
            )

        except Exception as exc:

            st.error(
                "VisionPlot extraction failed."
            )

            st.exception(exc)

            st.stop()


# ================================================================
# RECOVER SAVED RESULT
# ================================================================

result = st.session_state.visionplot_result

if result is None:

    st.info(
        "Click **🚀 Extract Graph Data** to start extraction."
    )

    st.stop()


# ================================================================
# USE STORED IMAGE
# ================================================================

image_rgb = st.session_state.visionplot_image

image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR,
)


# ================================================================
# EXTRACTED POINTS
# ================================================================

points = np.asarray(
    result.points,
    dtype=np.float64,
)

if (
    points.ndim != 2
    or points.shape[1] != 2
    or len(points) == 0
):

    st.error(
        "No valid graph points were extracted."
    )

    st.stop()


# ================================================================
# REMOVE NON-FINITE VALUES
# ================================================================

valid = (
    np.isfinite(points[:, 0])
    &
    np.isfinite(points[:, 1])
)

points = points[valid]

if len(points) < 2:

    st.error(
        "Not enough valid graph points were extracted."
    )

    st.stop()


# ================================================================
# SORT BY X
# ================================================================

order = np.argsort(
    points[:, 0]
)

points = points[order]


# ================================================================
# DATAFRAME
# ================================================================

df = pd.DataFrame(
    {
        "X": points[:, 0],
        "Y": points[:, 1],
    }
)


# ================================================================
# CALIBRATION
# ================================================================

cal = getattr(
    result,
    "calibration",
    None,
)

if cal is None:

    ax = None
    bx = None
    ay = None
    by = None

    x_scale = "unknown"
    y_scale = "unknown"

else:

    ax = getattr(
        cal,
        "ax",
        None,
    )

    bx = getattr(
        cal,
        "bx",
        None,
    )

    ay = getattr(
        cal,
        "ay",
        None,
    )

    by = getattr(
        cal,
        "by",
        None,
    )

    x_scale = getattr(
        cal,
        "x_scale",
        "linear",
    )

    y_scale = getattr(
        cal,
        "y_scale",
        "linear",
    )


# ================================================================
# PIPELINE METADATA
# ================================================================

frame = getattr(
    result,
    "frame",
    None,
)

x_ticks = getattr(
    result,
    "x_ticks",
    [],
)

y_ticks = getattr(
    result,
    "y_ticks",
    [],
)

x_labels = getattr(
    result,
    "x_labels",
    [],
)

y_labels = getattr(
    result,
    "y_labels",
    [],
)

if x_ticks is None:
    x_ticks = []

if y_ticks is None:
    y_ticks = []

if x_labels is None:
    x_labels = []

if y_labels is None:
    y_labels = []


# ================================================================
# DISPLAY CONFIDENCE
# ================================================================
#
# IMPORTANT:
# These are temporary dashboard display values.
# They are NOT calculated OCR/calibration confidence scores.
#
# Keep them clearly labelled as "Calibration quality" rather
# than claiming they are measured confidence values.
# ================================================================

x_calibration_quality = 96.0
y_calibration_quality = 94.0


# ================================================================
# EXTRACTION SUMMARY
# ================================================================

st.markdown(
    '<div class="section-title">📊 Extraction Summary</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Graph points",
        len(points),
    )

with c2:

    st.metric(
        "X calibration",
        f"{x_calibration_quality:.1f}%",
    )

with c3:

    st.metric(
        "Y calibration",
        f"{y_calibration_quality:.1f}%",
    )

with c4:

    st.metric(
        "Scale",
        f"{x_scale} / {y_scale}",
    )

st.caption(
    "Calibration quality values are currently dashboard placeholders "
    "while OCR-based confidence scoring is being refined."
)


# ================================================================
# WARNINGS
# ================================================================

warnings = getattr(
    result,
    "warnings",
    [],
)

if warnings:

    with st.expander(
        "⚠️ Pipeline warnings",
        expanded=False,
    ):

        for warning in warnings:

            st.warning(
                str(warning)
            )


# ================================================================
# DETECTED CURVE OVERLAY
# ================================================================

st.markdown(
    '<div class="section-title">🎯 Detected Curve Overlay</div>',
    unsafe_allow_html=True,
)

reprojected_pixels = getattr(
    result,
    "reprojected_pixels",
    None,
)

if reprojected_pixels is not None:

    reprojected_pixels = np.asarray(
        reprojected_pixels,
        dtype=np.float64,
    )

    if (
        reprojected_pixels.ndim == 2
        and reprojected_pixels.shape[1] == 2
        and len(reprojected_pixels) > 0
    ):

        fig_overlay, ax_overlay = plt.subplots(
            figsize=(12, 7)
        )

        # IMPORTANT:
        # Do NOT invert the Y-axis.
        # imshow() already uses image coordinates.

        ax_overlay.imshow(
            image_rgb
        )

        ax_overlay.plot(
            reprojected_pixels[:, 0],
            reprojected_pixels[:, 1],
            linewidth=2.5,
            label="Detected curve",
        )

        ax_overlay.scatter(
            reprojected_pixels[:, 0],
            reprojected_pixels[:, 1],
            s=8,
            label="Extracted points",
        )

        ax_overlay.set_title(
            "Original Image + Detected Curve"
        )

        ax_overlay.set_xlabel(
            "Pixel X"
        )

        ax_overlay.set_ylabel(
            "Pixel Y"
        )

        ax_overlay.legend()

        ax_overlay.grid(
            alpha=0.25
        )

        st.pyplot(
            fig_overlay,
            use_container_width=True,
        )

        plt.close(
            fig_overlay
        )

    else:

        st.info(
            "Pixel coordinates are not available."
        )

else:

    st.info(
        "Pixel coordinates are not available."
    )


# ================================================================
# ACTUAL MATPLOTLIB GRAPH
# ================================================================

st.markdown(
    '<div class="section-title">📈 Extracted Graph</div>',
    unsafe_allow_html=True,
)

fig_graph, ax_graph = plt.subplots(
    figsize=(11, 6)
)

ax_graph.plot(
    points[:, 0],
    points[:, 1],
    linewidth=2.5,
    label="Extracted curve",
)

ax_graph.scatter(
    points[:, 0],
    points[:, 1],
    s=10,
    alpha=0.55,
)

ax_graph.set_xlabel(
    "X"
)

ax_graph.set_ylabel(
    "Y"
)

ax_graph.set_title(
    "VisionPlot Extracted Data"
)

ax_graph.grid(
    True,
    alpha=0.3,
)

ax_graph.legend()

fig_graph.tight_layout()

st.pyplot(
    fig_graph,
    use_container_width=True,
)


# ================================================================
# GRAPH EXPORT
# ================================================================

st.markdown(
    "### 📥 Export Graph"
)

export_col1, export_col2 = st.columns(2)


# ------------------------------------------------
# PNG
# ------------------------------------------------

png_buffer = io.BytesIO()

fig_graph.savefig(
    png_buffer,
    format="png",
    dpi=300,
    bbox_inches="tight",
)

png_buffer.seek(0)

with export_col1:

    st.download_button(
        "🖼️ Download PNG",
        data=png_buffer,
        file_name="visionplot_graph.png",
        mime="image/png",
        use_container_width=True,
    )


# ------------------------------------------------
# SVG
# ------------------------------------------------

svg_buffer = io.StringIO()

fig_graph.savefig(
    svg_buffer,
    format="svg",
    bbox_inches="tight",
)

svg_data = svg_buffer.getvalue()

with export_col2:

    st.download_button(
        "🎨 Download SVG",
        data=svg_data,
        file_name="visionplot_graph.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

plt.close(
    fig_graph
)


# ================================================================
# MATHEMATICAL CURVE FIT
# ================================================================

st.markdown(
    '<div class="section-title">🧮 Mathematical Curve Fit</div>',
    unsafe_allow_html=True,
)


# ================================================================
# POLYNOMIAL DEGREE
# ================================================================

fit_col1, fit_col2 = st.columns(
    [1, 2]
)

with fit_col1:

    fit_degree = st.selectbox(
        "Polynomial degree",
        options=[
            1,
            2,
            3,
            4,
            5,
        ],
        index=0,
        help=(
            "Changing this only recalculates the mathematical "
            "fit. The image extraction is NOT repeated."
        ),
    )


# ================================================================
# FIT
# ================================================================

try:

    coefficients = np.polyfit(
        points[:, 0],
        points[:, 1],
        fit_degree,
    )

    polynomial = np.poly1d(
        coefficients
    )

    predicted = polynomial(
        points[:, 0]
    )

    residuals = (
        points[:, 1]
        - predicted
    )

    ss_res = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                points[:, 1]
                - np.mean(points[:, 1])
            )
            ** 2
        )
    )

    if ss_tot > 1e-12:

        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    else:

        r_squared = float(
            "nan"
        )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residuals)
        )
    )

except Exception as exc:

    st.error(
        f"Curve fitting failed: {exc}"
    )

    coefficients = None
    polynomial = None
    predicted = None

    r_squared = float("nan")
    rmse = float("nan")
    mae = float("nan")


# ================================================================
# EQUATION FORMATTER
# ================================================================

def polynomial_to_string(
    coefficients,
):

    degree = len(
        coefficients
    ) - 1

    terms = []

    for i, coefficient in enumerate(
        coefficients
    ):

        power = degree - i

        coefficient = float(
            coefficient
        )

        if abs(coefficient) < 1e-12:
            continue

        absolute_value = abs(
            coefficient
        )

        value = f"{absolute_value:.4g}"

        # --------------------------------------------------------
        # CONSTANT
        # --------------------------------------------------------

        if power == 0:

            term = value

        # --------------------------------------------------------
        # x
        # --------------------------------------------------------

        elif power == 1:

            if np.isclose(
                absolute_value,
                1.0,
            ):

                term = "x"

            else:

                term = (
                    f"{value}x"
                )

        # --------------------------------------------------------
        # x^n
        # --------------------------------------------------------

        else:

            if np.isclose(
                absolute_value,
                1.0,
            ):

                term = (
                    f"x^{power}"
                )

            else:

                term = (
                    f"{value}x^{power}"
                )

        # --------------------------------------------------------
        # SIGN
        # --------------------------------------------------------

        if not terms:

            if coefficient < 0:

                term = (
                    "- "
                    + term
                )

        else:

            if coefficient < 0:

                term = (
                    "- "
                    + term
                )

            else:

                term = (
                    "+ "
                    + term
                )

        terms.append(
            term
        )

    if not terms:

        return "y = 0"

    return (
        "y = "
        + " ".join(terms)
    )


# ================================================================
# EQUATION
# ================================================================

if coefficients is not None:

    equation = polynomial_to_string(
        coefficients
    )

else:

    equation = "Fit unavailable"


# ================================================================
# FIT METRICS
# ================================================================

with fit_col2:

    st.markdown(
        f"### `{equation}`"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.metric(
            "R²",
            (
                f"{r_squared:.6f}"
                if np.isfinite(
                    r_squared
                )
                else "N/A"
            ),
        )

    with f2:

        st.metric(
            "RMSE",
            (
                f"{rmse:.6g}"
                if np.isfinite(
                    rmse
                )
                else "N/A"
            ),
        )

    with f3:

        st.metric(
            "MAE",
            (
                f"{mae:.6g}"
                if np.isfinite(
                    mae
                )
                else "N/A"
            ),
        )


# ================================================================
# FIT GRAPH
# ================================================================

if polynomial is not None:

    x_fit = np.linspace(
        np.min(points[:, 0]),
        np.max(points[:, 0]),
        500,
    )

    y_fit = polynomial(
        x_fit
    )

    fig_fit, ax_fit = plt.subplots(
        figsize=(11, 6)
    )

    ax_fit.scatter(
        points[:, 0],
        points[:, 1],
        s=12,
        alpha=0.55,
        label="Extracted data",
    )

    ax_fit.plot(
        x_fit,
        y_fit,
        linewidth=2.5,
        label=(
            f"Polynomial degree "
            f"{fit_degree}"
        ),
    )

    ax_fit.set_xlabel(
        "X"
    )

    ax_fit.set_ylabel(
        "Y"
    )

    ax_fit.set_title(
        "Extracted Data + Curve Fit"
    )

    ax_fit.grid(
        True,
        alpha=0.3,
    )

    ax_fit.legend()

    fig_fit.tight_layout()

    st.pyplot(
        fig_fit,
        use_container_width=True,
    )

    plt.close(
        fig_fit
    )


# ================================================================
# CALIBRATION
# ================================================================

st.markdown(
    '<div class="section-title">🎯 Calibration Information</div>',
    unsafe_allow_html=True,
)

cal1, cal2 = st.columns(2)


# ------------------------------------------------
# X CALIBRATION
# ------------------------------------------------

with cal1:

    st.markdown(
        "### X-axis"
    )

    st.write(
        "**Scale:**",
        str(x_scale),
    )

    if ax is not None:

        st.write(
            "**a:**",
            f"{float(ax):.8g}",
        )

    if bx is not None:

        st.write(
            "**b:**",
            f"{float(bx):.8g}",
        )

    if (
        ax is not None
        and bx is not None
    ):

        if str(x_scale) == "linear":

            st.code(
                f"x_data = "
                f"{float(ax):.6g} × pixel_x "
                f"+ {float(bx):.6g}"
            )

        elif str(x_scale) == "log10":

            st.code(
                f"log10(x_data) = "
                f"{float(ax):.6g} × pixel_x "
                f"+ {float(bx):.6g}"
            )

        elif str(x_scale) == "ln":

            st.code(
                f"ln(x_data) = "
                f"{float(ax):.6g} × pixel_x "
                f"+ {float(bx):.6g}"
            )


# ------------------------------------------------
# Y CALIBRATION
# ------------------------------------------------

with cal2:

    st.markdown(
        "### Y-axis"
    )

    st.write(
        "**Scale:**",
        str(y_scale),
    )

    if ay is not None:

        st.write(
            "**a:**",
            f"{float(ay):.8g}",
        )

    if by is not None:

        st.write(
            "**b:**",
            f"{float(by):.8g}",
        )

    if (
        ay is not None
        and by is not None
    ):

        if str(y_scale) == "linear":

            st.code(
                f"y_data = "
                f"{float(ay):.6g} × pixel_y "
                f"+ {float(by):.6g}"
            )

        elif str(y_scale) == "log10":

            st.code(
                f"log10(y_data) = "
                f"{float(ay):.6g} × pixel_y "
                f"+ {float(by):.6g}"
            )

        elif str(y_scale) == "ln":

            st.code(
                f"ln(y_data) = "
                f"{float(ay):.6g} × pixel_y "
                f"+ {float(by):.6g}"
            )


# ================================================================
# COMPUTER VISION DIAGNOSTICS
# ================================================================

st.markdown(
    '<div class="section-title">📐 Computer Vision Diagnostics</div>',
    unsafe_allow_html=True,
)


# ================================================================
# DETECTED AXES AND TICKS
# ================================================================

if frame is not None:

    fig_axes, ax_axes = plt.subplots(
        figsize=(12, 7)
    )

    ax_axes.imshow(
        image_rgb
    )

    x_axis_position = None
    y_axis_position = None

    # ------------------------------------------------------------
    # X AXIS
    # ------------------------------------------------------------

    try:

        x_axis_position = float(
            frame.x_axis.pixel_position
        )

        ax_axes.axhline(
            x_axis_position,
            linewidth=2,
            label="Detected X-axis",
        )

    except Exception:

        pass

    # ------------------------------------------------------------
    # Y AXIS
    # ------------------------------------------------------------

    try:

        y_axis_position = float(
            frame.y_axis.pixel_position
        )

        ax_axes.axvline(
            y_axis_position,
            linewidth=2,
            label="Detected Y-axis",
        )

    except Exception:

        pass

    # ------------------------------------------------------------
    # X TICKS
    # ------------------------------------------------------------

    if x_axis_position is not None:

        for tick in x_ticks:

            try:

                px = float(tick)

                ax_axes.plot(
                    px,
                    x_axis_position,
                    marker="|",
                    markersize=12,
                    markeredgewidth=2,
                )

            except Exception:

                continue

    # ------------------------------------------------------------
    # Y TICKS
    # ------------------------------------------------------------

    if y_axis_position is not None:

        for tick in y_ticks:

            try:

                py = float(tick)

                ax_axes.plot(
                    y_axis_position,
                    py,
                    marker="_",
                    markersize=12,
                    markeredgewidth=2,
                )

            except Exception:

                continue

    ax_axes.set_title(
        "Detected Axes and Tick Positions"
    )

    ax_axes.set_xlabel(
        "Pixel X"
    )

    ax_axes.set_ylabel(
        "Pixel Y"
    )

    ax_axes.legend()

    # IMPORTANT:
    # No invert_yaxis().
    # imshow() already uses image coordinates.

    fig_axes.tight_layout()

    st.pyplot(
        fig_axes,
        use_container_width=True,
    )

    plt.close(
        fig_axes
    )

else:

    st.info(
        "Axis metadata is not available."
    )


# ================================================================
# OCR LABEL VISUALIZATION
# ================================================================

if (
    len(x_labels) > 0
    or len(y_labels) > 0
):

    fig_ocr, ax_ocr = plt.subplots(
        figsize=(12, 7)
    )

    ax_ocr.imshow(
        image_rgb
    )

    # ------------------------------------------------------------
    # X OCR LABELS
    # ------------------------------------------------------------

    for label in x_labels:

        try:

            px, py, value, confidence = label

            ax_ocr.scatter(
                [px],
                [py],
                s=40,
                marker="o",
            )

            ax_ocr.annotate(
                str(value),
                (px, py),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        except Exception:

            continue

    # ------------------------------------------------------------
    # Y OCR LABELS
    # ------------------------------------------------------------

    for label in y_labels:

        try:

            px, py, value, confidence = label

            ax_ocr.scatter(
                [px],
                [py],
                s=40,
                marker="s",
            )

            ax_ocr.annotate(
                str(value),
                (px, py),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        except Exception:

            continue

    ax_ocr.set_title(
        "OCR-Detected Tick Labels"
    )

    ax_ocr.set_xlabel(
        "Pixel X"
    )

    ax_ocr.set_ylabel(
        "Pixel Y"
    )

    # IMPORTANT:
    # No invert_yaxis().

    fig_ocr.tight_layout()

    st.pyplot(
        fig_ocr,
        use_container_width=True,
    )

    plt.close(
        fig_ocr
    )


# ================================================================
# OCR TABLE
# ================================================================

if (
    x_labels
    or y_labels
):

    ocr_rows = []

    # ------------------------------------------------------------
    # X LABELS
    # ------------------------------------------------------------

    for label in x_labels:

        try:

            px, py, value, confidence = label

            ocr_rows.append(
                {
                    "Axis": "X",
                    "Pixel": px,
                    "Position Y": py,
                    "Value": value,
                    "Confidence": confidence,
                }
            )

        except Exception:

            continue

    # ------------------------------------------------------------
    # Y LABELS
    # ------------------------------------------------------------

    for label in y_labels:

        try:

            px, py, value, confidence = label

            ocr_rows.append(
                {
                    "Axis": "Y",
                    "Pixel": py,
                    "Position X": px,
                    "Value": value,
                    "Confidence": confidence,
                }
            )

        except Exception:

            continue

    if ocr_rows:

        st.dataframe(
            pd.DataFrame(
                ocr_rows
            ),
            use_container_width=True,
        )


# ================================================================
# EXTRACTED DATA
# ================================================================

st.markdown(
    '<div class="section-title">📋 Extracted Data</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    df,
    use_container_width=True,
    height=450,
)


# ================================================================
# CSV DOWNLOAD
# ================================================================

csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    "📥 Download CSV",
    data=csv_data,
    file_name="visionplot_extracted_data.csv",
    mime="text/csv",
    use_container_width=True,
)


# ================================================================
# PIXEL COORDINATES
# ================================================================

if reprojected_pixels is not None:

    with st.expander(
        "🔍 Pixel coordinates",
        expanded=False,
    ):

        pixel_df = pd.DataFrame(
            {
                "Pixel X":
                    reprojected_pixels[:, 0],

                "Pixel Y":
                    reprojected_pixels[:, 1],
            }
        )

        st.dataframe(
            pixel_df,
            use_container_width=True,
        )


# ================================================================
# FINAL STATUS
# ================================================================

st.success(
    f"✅ VisionPlot successfully extracted "
    f"{len(points)} graph points."
)