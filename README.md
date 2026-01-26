# LSPIV - Large Scale Particle Image Velocimetry

A Python-based tool for analyzing surface water flow velocities using video imagery. Includes both a command-line interface and a web application built with Streamlit.

## Features

- **Video-based flow analysis** - Upload videos of water surfaces to measure flow velocities
- **Interactive ROI selection** - Define regions of interest for analysis
- **Multiple visualizations** - Streamlines, heatmaps, quiver plots
- **Real-time progress tracking** - Monitor processing status
- **Export capabilities** - Download results as images or raw data

## Requirements

- Python 3.12+
- Ubuntu/Debian (or compatible Linux distribution)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/LSPIV.git
cd LSPIV
```

### 2. Install Python venv (if not already installed)

```bash
sudo apt install python3.12-venv
```

### 3. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r webapp/requirements.txt
```

## Running the Web Application

### Start the Streamlit app

```bash
streamlit run webapp/app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web App

1. **Upload Video** - Select a video file (MP4, AVI, MOV, MKV)
2. **Define ROI** - Enter coordinates for the 4 corners of your region of interest
3. **Set Parameters** - Adjust analysis parameters in the sidebar:
   - FPS (frame rate)
   - ROI Width in meters (for velocity scaling)
   - PIV Window Size
   - Overlap
   - Max Frames to analyze
4. **Process** - Click "Start Analysis" and wait for processing
5. **View Results** - Explore visualizations in the tabs:
   - Streamlines Overlay
   - Speed Heatmap
   - Quiver Plot
   - Streamlines Plot

## Running the Command-Line Version

```bash
cd V8_5_3
python main.py
```

Edit the parameters at the top of `main.py`:
```python
VIDEO_PATH = "Videos/your_video.mp4"
FPS = 60
ROI_WIDTH_METERS = 100.0
MAX_PIV_FRAMES = 10
```

## Project Structure

```
LSPIV/
├── webapp/                     # Streamlit web application
│   ├── app.py                  # Main application
│   ├── components/             # UI components
│   │   ├── video_upload.py
│   │   ├── roi_canvas.py
│   │   ├── parameter_panel.py
│   │   └── results_display.py
│   ├── utils/
│   │   └── session_state.py
│   └── requirements.txt
├── V8_5_3/                     # Core LSPIV library
│   ├── core/
│   │   └── pipeline.py         # Unified processing pipeline
│   ├── data_io/
│   │   └── video_loader.py
│   ├── image_processing/
│   │   ├── frame_warper.py
│   │   ├── roi_selector.py
│   │   └── transforms.py
│   ├── piv/
│   │   ├── piv_core.py         # PIV computation
│   │   ├── parallel_piv.py     # Parallel processing
│   │   └── scaling.py
│   ├── visualization/
│   │   └── plotter.py
│   └── main.py
└── README.md
```

## Quick Start (Copy-Paste)

```bash
# One-time setup
sudo apt install python3.12-venv
git clone https://github.com/yourusername/LSPIV.git
cd LSPIV
python3 -m venv venv
source venv/bin/activate
pip install -r webapp/requirements.txt

# Run the app (every time)
source venv/bin/activate
streamlit run webapp/app.py
```

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### "externally-managed-environment" error
Make sure you're using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### "No module named 'venv'" error
Install the venv package:
```bash
sudo apt install python3.12-venv
```

### Streamlit not found
Make sure the virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r webapp/requirements.txt
```

## License

MIT License
