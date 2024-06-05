import os
from datetime import date
import cv2
import requests
import yolov5
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from config import *
import torch
    

# List of objects for Objects365 and COCO datasets
objects365 = ['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup',
              'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
              'Book',
              'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag',
              'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Monitor/TV',
              'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers',
              'Bicycle',
              'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Basket', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird',
              'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck',
              'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning',
              'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife',
              'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Balloon', 'Tripod', 'Dog', 'Spoon',
              'Clock',
              'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish',
              'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan',
              'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer',
              'Skiboard',
              'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign',
              'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck',
              'Baseball Bat',
              'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard',
              'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite',
              'Strawberry',
              'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
              'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors',
              'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra',
              'Grape',
              'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
              'Billiards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber',
              'Cigar/Cigarette',
              'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extension Cord', 'Tong', 'Tennis Racket',
              'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine',
              'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine',
              'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hot-air balloon',
              'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice',
              'Wallet/Purse',
              'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball',
              'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush',
              'Penguin',
              'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts',
              'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
              'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers',
              'CD',
              'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroom', 'Screwdriver', 'Soap', 'Recorder',
              'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measure/Ruler', 'Pig', 'Showerhead', 'Globe',
              'Chips',
              'Steak', 'Crosswalk Sign', 'Stapler', 'Camel', 'Formula 1', 'Pomegranate', 'Dishwasher', 'Crab',
              'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
              'Butterfly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer',
              'Egg tart',
              'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target',
              'French',
              'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus',
              'Barbell',
              'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle',
              'Cosmetics Brush/Eyeliner Pencil',
              'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling',
              'Table Tennis']

coco80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']

# Combine and sort the object lists
items = []
for item in coco80:
    items.append(item.lower())

for item in objects365:
    items.append(item.lower())

combined_objects = list(set(items))
combined_objects.sort()

# Define the weights for room objects
room_objects_weights = {
    'Bedroom': {
        'bed': (0.4, 0.2), 'pillow': (0.2, 0.1), 'nightstand': (0.1, 0.1),
        'lamp': (0.1, 0.1), 'chair': (0.05, 0.1), 'desk': (0.05, 0.1),
        'sneakers': (0.1, 0.1)
    },
    'Living room': {
        'couch': (0.5, 0.3), 'monitor/tv': (0.2, 0.2), 'coffee table': (0.2, 0.2),
        'chair': (0.1, 0.1), 'lamp': (0.1, 0.1), 'book': (0.1, 0.1), 'pillow': (0.2, 0.1),
        'vase': (0.2, 0.2), 'picture/frame': (0.1, 0.1)
    },
    'Children\'s room': {
        'bed': (0.4, 0.2), 'stuffed toy': (0.3, 0.2), 'desk': (0.1, 0.1),
        'chair': (0.1, 0.1), 'teddy bear': (0.2, 0.2),
        'book': (0.2, 0.2), 'lamp': (0.1, 0.1)
    },
    'Bathroom': {
        'sink': (0.4, 0.2), 'toilet': (0.3, 0.2), 'bathtub': (0.2, 0.2),
        'mirror': (0.1, 0.1), 'towel': (0.1, 0.1), 'showerhead': (0.2, 0.2),
        'soap': (0.1, 0.1), 'brush': (0.1, 0.1)
    },
    'Toilet': {
        'toilet': (0.5, 0.3), 'toilet paper': (0.17, 0.1), 'sink': (0.3, 0.2),
        'mirror': (0.2, 0.1), 'trash bin can': (0.17, 0.1), 'soap': (0.08, 0.1),
        'urinal': (0.08, 0.1)
    },
    'Garage': {
        'car': (0.6, 0.4), 'bicycle': (0.2, 0.2), 'motorcycle': (0.2, 0.2),
        'van': (0.3, 0.3), 'heavy truck': (0.3, 0.3)
    },
    'Cosmetology room': {
        'chair': (0.3, 0.2), 'bed': (0.3, 0.2), 'lamp': (0.2, 0.2), 'mirror': (0.2, 0.2),
        'cosmetics': (0.2, 0.2), 'towel': (0.1, 0.1), 'brush': (0.2, 0.2),
        'scissors': (0.1, 0.1)
    },
    'Hospital': {
        'bed': (0.3, 0.3), 'monitor/tv': (0.2, 0.2), 'sink': (0.3, 0.2),
        'toilet': (0.2, 0.1), 'trolley': (0.1, 0.1), 'trash bin can': (0.05, 0.1),
        'wheelchair': (0.3, 0.3), 'gloves': (0.2, 0.2)
    },
    'Classroom': {
        'desk': (0.3, 0.2), 'chair': (0.2, 0.2), 'blackboard/whiteboard': (0.3, 0.2),
        'picture/frame': (0.2, 0.1), 'projector': (0.2, 0.2), 'book': (0.2, 0.2),
        'notepaper': (0.1, 0.1), 'pencil case': (0.1, 0.1)
    },
    'Flower shop': {
        'flower': (0.4, 0.2), 'barrel/bucket': (0.3, 0.2), 'potted plant': (0.2, 0.2),
        'vase': (0.2, 0.2), 'scissors': (0.1, 0.1), 'gloves': (0.1, 0.1)
    },
    'Office (lobby)': {
        'desk': (0.2, 0.2), 'chair': (0.2, 0.2), 'potted plant': (0.2, 0.2),
        'book': (0.1, 0.1), 'handbag/satchel': (0.1, 0.1), 'person': (0.1, 0.1),
        'cell phone': (0.05, 0.1), 'couch': (0.3, 0.2), 'carpet': (0.3, 0.2),
        'vase': (0.3, 0.2), 'clock': (0.2, 0.2)
    },
    'Cafe': {
        'person': (0.4, 0.2), 'desk': (0.2, 0.2), 'chair': (0.3, 0.2),
        'lamp': (0.2, 0.2), 'plate': (0.2, 0.2), 'bowl/basin': (0.1, 0.1),
        'couch': (0.2, 0.2), 'bench': (0.1, 0.1), 'sink': (0.2, 0.2),
        'wine glass': (0.1, 0.1), 'coffee machine': (0.2, 0.2)
    },
    'Storage room': {
        'refrigerator': (0.2, 0.2), 'bottle': (0.2, 0.2), 'storage box': (0.3, 0.2),
        'barrel/bucket': (0.2, 0.2), 'cup': (0.1, 0.1), 'trash bin can': (0.2, 0.2),
        'bathtub': (0.2, 0.2), 'monitor/tv': (0.1, 0.1), 'towel': (0.1, 0.1)
    },
    'Outdoor': {
        'traffic cone': (0.3, 0.2), 'street lights': (0.3, 0.2), 'traffic light': (0.2, 0.2),
        'bicycle': (0.2, 0.2), 'bus': (0.2, 0.2), 'car': (0.2, 0.2),
        'truck': (0.2, 0.2), 'motorcycle': (0.2, 0.2), 'scooter': (0.2, 0.2),
        'skateboard': (0.2, 0.2), 'dog': (0.2, 0.2), 'cat': (0.2, 0.2),
        'horse': (0.2, 0.2), 'zebra': (0.2, 0.2), 'elephant': (0.2, 0.2),
        'helicopter': (0.2, 0.2), 'airplane': (0.2, 0.2), 'balloon': (0.2, 0.2),
        'ship': (0.2, 0.2), 'boat': (0.2, 0.2), 'surfboard': (0.2, 0.2),
        'lifesaver': (0.2, 0.2), 'flag': (0.2, 0.2), 'umbrella': (0.2, 0.2),
        'bench': (0.2, 0.2), 'bird': (0.2, 0.2), 'wild bird': (0.2, 0.2),
        'potted plant': (0.2, 0.2), 'lantern': (0.2, 0.2)
    },
    'Gazebo': {
        'car': (0.3, 0.2), 'chair': (0.3, 0.2), 'desk': (0.2, 0.2),
        'person': (0.1, 0.1), 'bench': (0.1, 0.1)
    },
    'Office': {
        'computer box': (0.3, 0.2), 'laptop': (0.3, 0.2), 'chair': (0.2, 0.2),
        'desk': (0.2, 0.2), 'monitor/tv': (0.2, 0.2), 'telephone': (0.1, 0.1),
        'printer': (0.1, 0.1), 'keyboard': (0.1, 0.1), 'mouse': (0.1, 0.1),
        'book': (0.1, 0.1), 'cell phone': (0.1, 0.1), 'mirror': (0.05, 0.1), 'radiator': (0.05, 0.1),
        'router/modem': (0.1, 0.1), 'clock': (0.1, 0.1), 'folder': (0.1, 0.1)
    },
    'Veranda/patio': {
        'bench': (0.3, 0.2), 'chair': (0.2, 0.2), 'pillow': (0.2, 0.2),
        'stroller': (0.2, 0.2), 'umbrella': (0.1, 0.1)
    },
    'Kitchen': {
        'oven': (0.5, 0.2), 'refrigerator': (0.3, 0.2), 'sink': (0.2, 0.2),
        'dining table': (0.3, 0.2), 'cabinet/shelf': (0.2, 0.2), 'chair': (0.1, 0.1),
        'microwave': (0.2, 0.1), 'toaster': (0.1, 0.1), 'kettle': (0.1, 0.1),
        'plate': (0.1, 0.1), 'cup': (0.1, 0.1), 'fork': (0.1, 0.1), 'knife': (0.1, 0.1)
    }
}


# Function to calculate the room match score
def calculate_room_match(detected_objects, room_objects_weights):
    room_scores = {}
    for room, objects_weights in room_objects_weights.items():
        room_score = 0
        for obj, (weight, volume_weight) in objects_weights.items():
            obj_lower = obj.lower()
            contribution = min(weight * detected_objects.get(obj_lower, 0), weight)
            volume_contribution = detected_objects.get(f'{obj_lower}_volume', 0) * volume_weight
            room_score += contribution + volume_contribution
        total_weight = sum(weight + volume_weight for weight, volume_weight in objects_weights.values())
        room_scores[room] = (room_score / total_weight) * 100  # Percentage match
    sorted_rooms = sorted(room_scores.items(), key=lambda x: x[1], reverse=True)
    filtered_rooms = [(room, score) for room, score in sorted_rooms if score >= 10]
    return filtered_rooms[:5]


# Function to add room labels to the image
def add_room_labels(image, room_scores):
    if not room_scores:
        return image

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

    if debug:
        y_offset = 10
        for room, score in room_scores:
            text = f"{room}: {score:.1f}%"
            draw.text((11, y_offset + 1), text, font=font, fill=(0, 0, 0))
            draw.text((9, y_offset + 1), text, font=font, fill=(0, 0, 0))
            draw.text((11, y_offset - 1), text, font=font, fill=(0, 0, 0))
            draw.text((9, y_offset - 1), text, font=font, fill=(0, 0, 0))
            draw.text((10, y_offset), text, font=font, fill=(255, 255, 255))
            y_offset += 30

    else:
        # Get the room with the maximum score
        max_room, max_score = room_scores[0]
        text = f"{max_room}: {max_score:.1f}%"

        draw.text((11, 11), text, font=font, fill=(0, 0, 0))
        draw.text((9, 11), text, font=font, fill=(0, 0, 0))
        draw.text((11, 9), text, font=font, fill=(0, 0, 0))
        draw.text((9, 9), text, font=font, fill=(0, 0, 0))
        draw.text((10, 10), text, font=font, fill=(255, 255, 255))

    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image


# Function to draw detected objects on the image
def draw_detected_objects(image, results, class_names, obj365=False):
    for *xyxy, conf, cls in results:
        if conf < tolerance:
            continue

        if obj365:
            label = f"{class_names[int(cls)]} {conf:.2f} (365)"
            color = (128, 0, 128)  # Purple for Objects365
        else:
            label = f"{class_names[int(cls)]} {conf:.2f}"
            color = (255, 0, 0)  # Blue for other models

        plot_one_box(xyxy, image, label=label, color=color, line_thickness=2)

    return image


# Function to plot a bounding box on the image
def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def filter_and_process_images(model_objects365, model_coco, path_name, result_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_objects365.to(device)
    model_coco.to(device)

    items = [os.path.join(root, file) for root, _, files in os.walk(path_name) for file in files if
             file.lower().endswith('.jpg')]
    total = len(items)

    for item in tqdm(items, total=total):
        try:
            img = cv2.imread(item)
            if img is None:
                continue  

            results_objects365 = model_objects365(item, size=1280, device=device)
            results_coco = model_coco(item, size=1280, device=device)

            img_with_labels = cv2.imread(item)

            detected_objects = {}

            for *xyxy, conf, cls in results_objects365.xyxy[0].to(device):
                if conf < tolerance:
                    continue
                obj_name = model_objects365.names[int(cls)]
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / (img_with_labels.shape[0] * img_with_labels.shape[1])
                if obj_name in detected_objects:
                    detected_objects[obj_name] += conf.item()
                    detected_objects[f'{obj_name}_volume'] += area
                else:
                    detected_objects[obj_name] = conf.item()
                    detected_objects[f'{obj_name}_volume'] = area

            for *xyxy, conf, cls in results_coco.xyxy[0].to(device):
                if conf < tolerance:
                    continue
                obj_name = model_coco.names[int(cls)]
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / (img_with_labels.shape[0] * img_with_labels.shape[1])
                if obj_name in detected_objects:
                    detected_objects[obj_name] += conf.item()
                    detected_objects[f'{obj_name}_volume'] += area
                else:
                    detected_objects[obj_name] = conf.item()
                    detected_objects[f'{obj_name}_volume'] = area

            if objects_labels:
                # Objects365
                img_with_labels = draw_detected_objects(img_with_labels, results_objects365.xyxy[0],
                                                        model_objects365.names,
                                                        obj365=True)

                # COCO
                img_with_labels = draw_detected_objects(img_with_labels, results_coco.xyxy[0], model_coco.names)

            room_scores = calculate_room_match(detected_objects, room_objects_weights)

            if room_scores:
                max_room, max_score = room_scores[0]
            else:
                max_room = "Other"

            if max_room in class_white_list:
                if class_labels:
                    img_with_labels = add_room_labels(img_with_labels, room_scores)

                room_dir = os.path.join(result_dir, max_room + 's')
                if not os.path.exists(room_dir):
                    os.makedirs(room_dir)

                filename = os.path.basename(item)
                cv2.imwrite(f"{room_dir}/{filename}", img_with_labels)

            if delete_processed_file:
                os.remove(item)

        except Exception as e:
            # print(f"Error processing file {item}: {e}")
            pass


# Function to download a file
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


if __name__ == '__main__':

    # Path to the models directory
    models_dir = os.path.join(os.getcwd(), 'models')
    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Full paths to the model files
    model_coco_path = os.path.join(models_dir, model_coco_filename)
    model_365_path = os.path.join(models_dir, model_365_filename)

    # Check and download the models
    if not os.path.exists(model_coco_path):
        print(f"Downloading COCO model to {model_coco_path}...")
        download_file(model_coco_url, model_coco_path)
        print(f"COCO model downloaded successfully.")

    if not os.path.exists(model_365_path):
        print(f"Downloading 365 model to {model_365_path}...")
        download_file(model_365_url, model_365_path)
        print(f"365 model downloaded successfully.")

    if debug:
        used_objects = set()
        for room, objects_weights in room_objects_weights.items():
            used_objects.update(objects_weights.keys())

        unused_objects = set(combined_objects) - used_objects
        new_objects = used_objects - set(combined_objects)

        print(f"Unused: {unused_objects}")

        print(f"Bad: {new_objects}")

    model_objects365 = yolov5.load(model_365_path)
    # set model parameters
    model_objects365.conf = 0.25  # NMS confidence threshold
    model_objects365.iou = 0.45  # NMS IoU threshold
    model_objects365.agnostic = False  # NMS class-agnostic
    model_objects365.multi_label = False  # NMS multiple labels per box
    model_objects365.max_det = 1000  # maximum number of detections per image

    model_coco = yolov5.load(model_coco_path)
    # set model parameters
    model_coco.conf = 0.25  # NMS confidence threshold
    model_coco.iou = 0.45  # NMS IoU threshold
    model_coco.agnostic = False  # NMS class-agnostic
    model_coco.multi_label = False  # NMS multiple labels per box
    model_coco.max_det = 1000  # maximum number of detections per image

    today = date.today()
    result_dir = f'{results_dir}/{today}'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    try:
        filter_and_process_images(model_objects365, model_coco, images_dir, result_dir)
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
