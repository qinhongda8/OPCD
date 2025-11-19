import cv2
from segment_anything import sam_model_registry
import time
import os
import glob
import json
import numpy as np
import cv2
import shutil
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

images_dir = "./N_to_A/images"
labels_dir = "./N_to_A/labels"

output_masks_dir     = "./N_to_A_out/masks"
output_overlays_dir  = "./overlays_N_to_A_out"
output_json_dir      = "./N_to_A_out/json"
output_images_dir    = "./N_to_A_out/new_images"
output_labels_dir    = "./N_to_A_out/new_labels"

for d in [output_masks_dir, output_overlays_dir, output_json_dir, output_images_dir, output_labels_dir]:
    os.makedirs(d, exist_ok=True)

yolo2nor = {
    0: 15,   # bus    → 15
    1: 18,   # bicycle→ 18
    2: 13,   # car    → 13
    3: 17,   # motor  → 17
    4: 11,   # person → 11
    5: 12,   # rider  → 12
    6: 14    # truck  → 14
}

id_to_cat = {
    15: "bus",   
    18: "bicycle",   
    13: "car",   
    17: "motor",   
    11: "person",  
    12: "rider",   
    14: "truck"    
}

default_attributes = {"occluded": False, "truncated": False, "trafficLightColor": "none"}

a_colors = {
    15: (0, 60, 100),    # bus
    18: (119, 11, 32),   # bicycle
    13: (0, 0, 142),     # car
    17: (0, 0, 230),     # motorcycle
    11: (220, 20, 60),   # person
    12: (255, 0, 0),     # rider
    14: (0, 0, 70)       # truck
}

def apply_color_mask(image, mask, color, color_dark=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - color_dark) + color_dark * color[c],
                                  image[:, :, c])
    return image

def yolo_to_pixel_box(line, img_width, img_height):
    cls, xc, yc, w, h = map(float, line.split())
    x1 = (xc - w / 2) * img_width
    y1 = (yc - h / 2) * img_height
    x2 = (xc + w / 2) * img_width
    y2 = (yc + h / 2) * img_height
    return int(cls), [x1, y1, x2, y2]

def extract_poly_from_mask(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    poly = []
    for pt in approx:
        pt = pt[0]
        poly.append([float(pt[0]), float(pt[1]), "L"])
    return poly

def process_image(img_path, yolo_txt_path, predictor):
    image_origin = cv2.imread(img_path)
    if image_origin is None:
        raise ValueError("cannt read: " + img_path)
    image_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    
    predictor.set_image(image_rgb)
    
    objects = [] 
    instance_mask = np.zeros((H, W), dtype=np.uint8)  
    instance_overlay = image_origin.copy() 
    obj_id = 0

    with open(yolo_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yolo_cls, box = yolo_to_pixel_box(line, W, H)
            yolo_cls = yolo2nor.get(yolo_cls, 0) 
            a_id = yolo_cls
            detection_obj = {
                "category": id_to_cat.get(a_id, "unknown"),
                "id": obj_id,
                "attributes": default_attributes,
                "box2d": {
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3]
                }
            }
            objects.append(detection_obj)
            obj_id += 1
        
            box_np = np.array(box)[None, :]  # shape (1,4)
            masks, scores, logits = predictor.predict(box=box_np, multimask_output=True)
            if len(masks) == 0:
                continue
            mask = masks[0]  
    
            instance_mask[mask == 1] = a_id
            
            poly = extract_poly_from_mask(mask)
            if poly is not None:
                segmentation_obj = {
                    "category": a_id,  
                    "id": obj_id,
                    "attributes": {},
                    "poly2d": [poly]
                }
                objects.append(segmentation_obj)
                obj_id += 1
            
            color = a_colors.get(a_id, (0, 0, 0))
            instance_overlay = apply_color_mask(instance_overlay, mask, color, color_dark=0.5)
    
    name = os.path.splitext(os.path.basename(img_path))[0]
    json_data = {
        "name": name,
        "frames": [
            {
                "timestamp": 0,
                "objects": objects
            }
        ],
        "attributes": {}
    }
    return json_data, instance_mask, instance_overlay


sam_checkpoint = "./sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda:0"  # select device
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
times = [] 

image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
i = 0
for img_file in tqdm(image_files):
    i = i + 1
    if i > 1000:
        break
    base = os.path.splitext(os.path.basename(img_file))[0]
    yolo_file = os.path.join(labels_dir, f"{base}.txt")
    if not os.path.exists(yolo_file):
        print(f"pass {base}: cannt find label")
        continue
    start_time = time.time()

    json_data, inst_mask, inst_overlay = process_image(img_file, yolo_file, predictor)
    
    elapsed = time.time() - start_time
    times.append((base, elapsed))
    print(f"processed : {base}, time {elapsed:.3f} s")

    json_out_path = os.path.join(output_json_dir, f"{base}.json")
    with open(json_out_path, "w") as fp:
        json.dump(json_data, fp, indent=4)
    

    mask_out_path = os.path.join(output_masks_dir, f"{base}_mask.png")
    cv2.imwrite(mask_out_path, inst_mask)

    overlay_out_path = os.path.join(output_overlays_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_out_path, inst_overlay)
    

    new_img_path = os.path.join(output_images_dir, os.path.basename(img_file))
    shutil.copy(img_file, new_img_path)
    

    new_label_path = os.path.join(output_labels_dir, os.path.basename(yolo_file))
    shutil.copy(yolo_file, new_label_path)
    
    print(f"processed: {base}")

print("all done")

