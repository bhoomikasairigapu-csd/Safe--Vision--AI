# Safe-Vision-AI# SAFE VISION AI

### Intelligent Road Safety and Traffic Monitoring System

## About the Project

SAFE VISION AI is a road safety and traffic monitoring project that uses Deep Learning and YOLO-based object detection to analyze traffic images and videos. It includes a Streamlit dashboard for presenting detection results.

## Objectives

* Explore and prepare traffic image datasets.
* Train and evaluate object detection models.
* Detect supported traffic objects using YOLO.
* Display detection results through a dashboard.

## Technologies Used

* Python
* YOLO / Ultralytics
* OpenCV
* Streamlit
* Jupyter Notebook

## Project Structure

* `dashboard.py` — Streamlit dashboard
* `01_dataset_inspection.ipynb` — Dataset inspection
* `02_data_cleaning.ipynb` — Dataset cleaning
* `requirements.txt` — Python dependencies
* `prepared_data/` — Prepared dataset configuration
* `runs/` — Model training and prediction outputs

## Installation

1. Install Python.
2. Clone the repository:
   `git clone https://github.com/bhoomikasairigapu-csd/Safe--Vision--AI.git`
3. Open the project folder:
   `cd Safe--Vision--AI`
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start the dashboard:
   `streamlit run dashboard.py`

## Dataset and Model

Datasets and trained model files may need to be downloaded or configured separately. Ensure the required dataset paths and model weights are available before running the application.

## Limitations

Detection performance depends on the trained model, dataset quality, and configuration. Real-time tracking, number plate recognition, and automated traffic fines should be considered future enhancements unless implemented and tested.

## Future Enhancements

* Improve traffic object and violation detection.
* Explore number plate recognition.
* Investigate emergency vehicle alerts.
* Evaluate real-time camera integration.

## Author

SAFE VISION AI Project Team
