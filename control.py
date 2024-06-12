from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import Counter
model_path = "runs/detect/best1.pt"
model = YOLO(model_path)


def is_image(input):
    try:
        # Try loading the input as an image
        img = cv2.imread(input)
        if img is not None:
            return True  # Input is an image path
    except:
        pass

    return False  # Input is not an image path


def make_bounding_box(img_path):
    if not is_image(img_path):
        image = img_path
    else:
        image = cv2.imread(img_path)
    results = model(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Extract the coordinates
            # print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2})" )
            # Draw the bounding box and label on the image
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
    
    return image_rgb

def count_classes_in_image(image_path):
    

    # Load the image
    if not is_image(image_path):
        image = image_path
    else:
        image = cv2.imread(image_path)

    # Perform inference
    results = model(image)


    # Extract class IDs
    class_ids = []
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[-1].item())  # Assuming the class ID is the last element
            class_ids.append(class_id)

    # Count the number of instances of each class
    class_counts = Counter(class_ids)

    # Map class IDs to class names
    class_names = {cls_id: model.names[cls_id] for cls_id in class_counts.keys()}
    class_counts_named = {class_names[cls_id]: count for cls_id, count in class_counts.items()}

    return class_counts_named

weights = {
        'ambulance': 5, 'army vehicle': 5, 'auto rickshaw': 2, 'bicycle': 1,
        'bus': 10, 'car': 3, 'garbagevan': 7, 'human hauler': 6, 'minibus': 8,
        'minivan': 4, 'motorbike': 1, 'pickup': 4, 'policecar': 5, 'rickshaw': 2,
        'scooter': 1, 'suv': 4, 'taxi': 3, 'three wheelers -CNG-': 3, 'truck': 9,
        'van': 4, 'wheelbarrow': 1
    }

def calculate_weight(img):
    class_count=count_classes_in_image(img)
    weighted_traffic=0
    for cls_id, count in class_count.items():
        class_name = cls_id
        weight = weights.get(class_name, 1)  # Default weight is 1 if not specified
        weighted_traffic += (count * weight)
        
    return weighted_traffic
    
    
def process_video_frames(video_path):
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():    # check whether its opened or not 
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:     # frame detection ck
            break
        cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)  # Create window with resizable option

        img = make_bounding_box(frame)
        weight= calculate_weight(frame)
        print(weight)
        # Display the frame
        cv2.imshow('Video Frame', img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

video = "video/2103099-uhd_3840_2160_30fps.mp4"
process_video_frames(video)
























