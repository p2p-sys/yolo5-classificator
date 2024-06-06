# Object Detection and Classification using YOLOv5

This program is designed to perform object detection and classification using YOLOv5 models. It processes images, detects objects, and classifies them into predefined categories. The detected objects are then used to match and label the images with corresponding room types based on their content.

## Features

- Detects and classifies objects using YOLOv5 models.
- Supports two YOLOv5 models: COCO and Objects365.
- Matches detected objects to predefined room types.
- Option to label images with room types.
- Option to delete processed images after processing.

## Room types
- Bedrooms
- Living Rooms
- Children's Rooms
- Bathrooms
- Toilets
- Garages
- Cosmetology Rooms
- Hospitals
- Classrooms
- Flower Shops
- Offices (Lobby)
- Cafes
- Storage Rooms
- Outdoors
- Gazebos
- Offices
- Verandas/Patios
- Kitchens  

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/p2p-sys/yolo5-classificator.git
   cd yolo5-classificator

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

4. Configuration
Open config.py and configure the paths and parameters as needed

6. Download Models
The program automatically downloads the necessary YOLOv5 models if they are not present in the models directory.

7. To run the program, execute the following command:
   ```bash
     python3 classificator.py

The program will process images from the specified directory, detect objects, classify them, and save the results in the output directory. By default, processed images will be deleted after processing. To keep the processed images, modify the delete_processed parameter in the filter_and_process_images function call within classificator.py:

License
This project is licensed under the MIT License. See the LICENSE file for details.

The YOLOv5 models used in this program are licensed under the GNU General Public License v3.0.

Acknowledgements
Ultralytics YOLOv5 for the object detection models.
All contributors to the open-source libraries used in this project.
