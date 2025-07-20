# Texture-Based Relative Depth Estimation

This project performs relative depth prediction using patch-wise ranking from a **single image**, leveraging a tiny CNN trained on self-supervised texture strength signals.

## 📁 Project Structure

texture_depth_project/
├── data/
│   └── jcsmr.jpg               # Your input image
├── dataset.py                 # Dataset creation from image patches
├── loss.py                    # Ranking loss for relative depth
├── model.py                   # CNN model
├── texture_utils.py           # Sobel filter & texture logic
├── train.py                   # Training loop
├── visualize.py               # Visualization of results
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # Project guide

## 🚀 How to Run

1. Place your image inside the `data/` folder (e.g., `data/jcsmr.jpg`)
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the full pipeline:
    ```bash
    python main.py
    ```

## 🧠 Method Summary

- Extracts two random patches from the image
- Compares texture strength using Sobel filters
- Assigns relative label (+1/-1)
- Trains CNN to learn patch-wise depth using ranking loss
- Predicts full image depth map and texture map

## 🖼 Output

- Relative Depth Heatmap (colormap: `plasma`)
- Low Texture Classifier Heatmap (colormap: `hot`)
