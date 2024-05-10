## Road Marking Detection

We use classical CV techniques to detect road markings given a RGB image. Here are the steps followed to generate the binary mask:
- Convert RGB image to grayscale image, apply contrast and smoothening filters.
- Apply Canny Edge Detection.
- Detect Region of Interest (ROI) using mask and get canny output of the mask.
- Apply Hough Transform to detect lines.
- Get contours from canny and hough output.
- Get the final mask by taking a union of contours from Canny and Hough.
- Apply super-resolution to the binary mask.

---

### [**Installation**](#) <a name="install"></a>

**1.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv py_env
$ source py_env/bin/activate
```

**2.** Install necessary packages:

``` shell
$ cd Road_Marking_Detection/
$ pip install -r requirements.txt
```

## Usage

```
python main.py -h

usage: main.py [-h] [--data_path] [--output_path] [--canny_lower_threshold] [--canny_upper_threshold]
               [--hough_threshold] [--hough_min_length] [--hough_max_gap]
usage options:
  --help                show this help message and exit
  --data_path           the path pointing the input road marking images
  --output_path         output directory for the binary mask
  --canny_lower_threshold  lower threshold value in Hysteresis Thresholding
  --canny_upper_threshold  upper threshold value in Hysteresis Thresholding
  --hough_threshold     hough threshold value to detect lines
  --hough_min_length    minimum length of line
  --hough_max_gap       maximum gap used to combine lines
```

