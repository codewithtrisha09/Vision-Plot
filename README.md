\# VisionPlot



\### Computer Vision-Based Chart Data Extraction



VisionPlot is an end-to-end computer vision system that extracts numerical data and plotted curves from chart images.



The system takes a chart image as input, detects its axes and tick marks, recognizes numerical labels using OCR, calibrates pixel coordinates to graph coordinates, extracts the plotted curve, and reconstructs the data as numerical `(x, y)` points.



It also provides interactive visualization, curve fitting, error metrics, and data export through a Streamlit dashboard.



\---



\## Overview



Charts and graphs often contain valuable numerical information that is difficult to reuse when the original dataset is unavailable.



VisionPlot addresses this problem by converting a chart image into structured numerical data.



\### Pipeline



```text

Chart Image

&#x20;    │

&#x20;    ▼

Image Preprocessing

&#x20;    │

&#x20;    ├── Grayscale Conversion

&#x20;    ├── Denoising

&#x20;    ├── Illumination Normalization

&#x20;    ├── Thresholding

&#x20;    └── Edge Detection

&#x20;    │

&#x20;    ▼

Axis Detection

&#x20;    │

&#x20;    ├── Morphological Processing

&#x20;    ├── Hough Line Detection

&#x20;    └── Tick Detection

&#x20;    │

&#x20;    ▼

OCR

&#x20;    │

&#x20;    ├── Tesseract OCR

&#x20;    ├── Numeric Label Detection

&#x20;    └── Label Filtering

&#x20;    │

&#x20;    ▼

Coordinate Calibration

&#x20;    │

&#x20;    ├── Pixel → Data Coordinate Mapping

&#x20;    └── RANSAC-Based Robust Fitting

&#x20;    │

&#x20;    ▼

Curve Extraction

&#x20;    │

&#x20;    ├── Color Segmentation

&#x20;    ├── Axis/Grid Removal

&#x20;    ├── Skeletonization

&#x20;    └── Curve Point Extraction

&#x20;    │

&#x20;    ▼

Numerical Dataset

&#x20;    │

&#x20;    ├── CSV Export

&#x20;    ├── Graph Reconstruction

&#x20;    └── Curve Fitting

````



\---



\## Key Features



\* Upload chart images through an interactive Streamlit interface

\* Automatic grayscale conversion and preprocessing

\* Edge-preserving image denoising

\* Illumination normalization

\* Otsu thresholding

\* Canny edge detection

\* Automatic horizontal and vertical axis detection

\* Hough-based line detection

\* Tick detection

\* Numerical tick-label recognition using Tesseract OCR

\* Automatic filtering of OCR detections based on axis geometry

\* Pixel-to-data coordinate calibration

\* Robust RANSAC-based calibration

\* Automatic plotted-curve color detection

\* HSV-based curve segmentation

\* Removal of axes and grid lines

\* Connected-component filtering

\* Skeletonization/thinning of the detected curve

\* Curve point extraction

\* Data resampling

\* Reconstruction of the extracted graph

\* Polynomial curve fitting

\* R², RMSE and MAE evaluation

\* CSV data export

\* PNG/SVG graph export

\* Visualization of detected axes, OCR labels and extracted curve

\* Interactive dashboard for the complete extraction pipeline



\---



\## Computer Vision Techniques



VisionPlot combines several image-processing and computer-vision techniques:



| Technique                  | Purpose                                                                |

| -------------------------- | ---------------------------------------------------------------------- |

| Grayscale Conversion       | Simplifies image processing                                            |

| Guided/Bilateral Filtering | Reduces noise while preserving important structures                    |

| Median Filtering           | Background/illumination processing                                     |

| Illumination Normalization | Reduces uneven background intensity                                    |

| Otsu Thresholding          | Converts the image into a binary representation                        |

| Canny Edge Detection       | Detects strong image boundaries                                        |

| Morphological Processing   | Detects and processes axis/grid structures                             |

| Hough Line Detection       | Identifies long horizontal and vertical lines                          |

| ROI Processing             | Restricts processing to the graph region                               |

| HSV Color Segmentation     | Separates the plotted curve from the background                        |

| Connected Components       | Removes small unwanted regions                                         |

| Skeletonization            | Reduces the curve to a one-pixel-wide representation                   |

| Branch Pruning             | Removes small skeleton branches                                        |

| OCR                        | Recognizes numerical tick labels                                       |

| Geometric Calibration      | Maps pixel coordinates to graph coordinates                            |

| RANSAC                     | Provides robust coordinate fitting in the presence of noisy detections |

| Reprojection               | Validates the pixel-to-data transformation                             |



\---



\## System Architecture



```text

&#x20;                  ┌─────────────────────┐

&#x20;                  │     Chart Image      │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                  ┌─────────────────────┐

&#x20;                  │   Preprocessing     │

&#x20;                  │ OpenCV + NumPy      │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                  ┌─────────────────────┐

&#x20;                  │    Axis Detection   │

&#x20;                  │ Hough + Morphology  │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                   ┌─────────┴─────────┐

&#x20;                   ▼                   ▼

&#x20;            ┌─────────────┐     ┌─────────────┐

&#x20;            │ Tick/Axis   │     │     OCR     │

&#x20;            │ Detection   │     │ Tesseract   │

&#x20;            └──────┬──────┘     └──────┬──────┘

&#x20;                   │                   │

&#x20;                   └─────────┬─────────┘

&#x20;                             ▼

&#x20;                  ┌─────────────────────┐

&#x20;                  │    Calibration      │

&#x20;                  │ Pixel → Data Space  │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                  ┌─────────────────────┐

&#x20;                  │   Curve Detection   │

&#x20;                  │ HSV + Skeletonize   │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                  ┌─────────────────────┐

&#x20;                  │  Numerical Dataset  │

&#x20;                  │      (x, y)         │

&#x20;                  └──────────┬──────────┘

&#x20;                             │

&#x20;                ┌────────────┼────────────┐

&#x20;                ▼            ▼            ▼

&#x20;            CSV Data    Graph Plot    Curve Fit

```



\---



\## Example Workflow



\### 1. Upload a Chart



The user uploads an image containing a plotted graph.



\### 2. Preprocess the Image



The image is converted to grayscale and processed using denoising, illumination normalization, thresholding and edge detection.



\### 3. Detect Axes



The system identifies the horizontal and vertical axes using morphological operations and Hough line detection.



\### 4. Detect and Recognize Tick Labels



Tick positions are detected and numerical labels are recognized using Tesseract OCR.



The OCR results are spatially filtered so that X-axis labels and Y-axis labels are associated with the correct axis.



\### 5. Calibrate Coordinates



Pixel coordinates are mapped to graph coordinates using recognized tick values.



For example:



```text

Pixel coordinate → Graph coordinate



(614, 717) → (0, 0)

(840, 717) → (2, 0)

(614, 534) → (0, 2)

```



This allows the extracted curve to be represented in the original mathematical coordinate system.



\### 6. Extract the Curve



VisionPlot detects the plotted curve using color-based segmentation.



Axes and grid lines are removed before skeletonization and curve-point extraction.



\### 7. Reconstruct the Data



The detected pixels are converted into numerical `(x, y)` coordinates.



The resulting data can be:



\* visualized as a graph

\* exported as CSV

\* fitted using a polynomial model

\* evaluated using statistical error metrics



\---



\## Curve Fitting



VisionPlot can fit a polynomial model to the extracted data.



For example:



```text

y = 0.0764x² + 0.4729x + 0.9746

```



The fitted model can be evaluated using:



\### R²



Measures how well the fitted model explains the extracted data.



\### RMSE



Measures the root mean squared difference between the extracted values and fitted values.



\### MAE



Measures the mean absolute difference between the extracted values and fitted values.



These metrics provide a quantitative way to evaluate the reconstructed curve.



> The fitted polynomial is an approximation of the extracted points and does not necessarily represent the original equation used to generate the chart.



\---



\## Project Structure



```text

Vision-Plot/

│

├── VP/

│   ├── app.py

│   │

│   └── src/

│       └── chart\_extractor/

│           ├── \_\_init\_\_.py

│           ├── axes.py

│           ├── calibration.py

│           ├── curve.py

│           ├── ocr.py

│           ├── pipeline.py

│           ├── preprocess.py

│           └── types.py

│

├── README.md

├── requirements.txt

├── .gitignore

└── ...

```



\---



\## Tech Stack



\### Programming Language



\* Python



\### Computer Vision



\* OpenCV

\* NumPy



\### OCR



\* Tesseract OCR

\* pytesseract



\### Data Processing



\* Pandas

\* NumPy



\### Visualization



\* Matplotlib

\* Streamlit



\### Image Processing



\* OpenCV

\* Pillow



\---



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/codewithtrisha09/Vision-Plot.git

cd Vision-Plot

```



\### 2. Create a virtual environment



\#### Windows



```powershell

python -m venv venv

```



Activate it:



```powershell

.\\venv\\Scripts\\activate

```



\### 3. Install Python dependencies



```powershell

pip install -r requirements.txt

```



\### 4. Install Tesseract OCR



VisionPlot uses Tesseract for recognizing numerical tick labels.



After installing Tesseract, make sure the executable is available to `pytesseract`.



On Windows, a common installation location is:



```text

C:\\Program Files\\Tesseract-OCR\\tesseract.exe

```



VisionPlot can automatically detect common Windows Tesseract installation paths.



\---



\## Running the Application



From the project root:



```powershell

streamlit run .\\VP\\app.py

```



The Streamlit dashboard will open in your browser.



\---



\## Output



VisionPlot produces several useful outputs from a chart image:



\### 1. Detected Curve Overlay



The original image is displayed with the detected curve overlaid.



\### 2. Reconstructed Graph



The extracted numerical points are plotted using the calibrated graph coordinates.



\### 3. Axis and Tick Visualization



The detected X and Y axes and tick positions can be inspected.



\### 4. OCR Results



The numerical labels recognized by Tesseract are displayed.



\### 5. Numerical Dataset



The extracted points are provided as `(x, y)` values.



\### 6. CSV Export



The extracted data can be downloaded as a CSV file.



\### 7. Graph Export



The reconstructed graph can be exported as PNG/SVG.



\### 8. Curve Equation



A polynomial approximation can be generated from the extracted data.



\---



\## Example Result



For a chart with an X-axis approximately ranging from `-5` to `5` and a Y-axis ranging from `0` to `8`, VisionPlot can reconstruct the plotted curve into numerical coordinates such as:



```text

X          Y

\-4.9765    0.1726

\-4.9265    0.1845

\-4.8764    0.1845

...

```



The extracted points can then be plotted independently of the original image.



\---



\## Why VisionPlot?



A chart image contains information in pixel form rather than directly accessible numerical form.



VisionPlot bridges the gap between:



```text

Image

&#x20; ↓

Pixels

&#x20; ↓

Computer Vision

&#x20; ↓

OCR + Calibration

&#x20; ↓

Numerical Coordinates

&#x20; ↓

Structured Data

```



This makes chart images usable as a source for further numerical analysis and visualization.



\---



\## Limitations



Current extraction performance can depend on the quality and structure of the input chart.



Challenges include:



\* Low-resolution images

\* Overlapping curves

\* Multiple plotted curves

\* Curves with colors similar to the background

\* Heavy grid lines

\* Irregular or rotated charts

\* Non-numeric axis labels

\* Logarithmic or non-linear axes

\* Complex chart layouts



The current implementation is primarily designed for charts containing a clearly distinguishable plotted curve and numerical axes.



\---



\## Future Improvements



Potential extensions include:



\* Support for multiple curves in a single chart

\* Improved automatic curve/legend association

\* Better handling of logarithmic axes

\* Improved OCR confidence scoring

\* Automatic model selection for curve fitting

\* Support for scatter plots and bar charts

\* Improved handling of rotated charts

\* Deep-learning-based curve segmentation

\* Automatic chart type classification

\* Interactive manual correction of detected axes and OCR labels

\* Export to Excel and other structured formats



\---



\## Project Goals



VisionPlot was developed to explore the combination of:



\* Computer vision

\* Image processing

\* OCR

\* Geometric transformations

\* Numerical data extraction

\* Data visualization



The project demonstrates how an image-processing pipeline can transform an unstructured chart image into structured numerical data.



\---



\## Author



\*\*Trisha Shetty\*\*



B.Tech Computer Science and Engineering

Artificial Intelligence \& Machine Learning



MIT Manipal



\---



\## License



This project is currently intended for academic and educational use.



````



