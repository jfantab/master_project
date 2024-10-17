

def xywh2xyxy(bbox):    
    x1, y1, w, h = bbox
    
    x2, y2 = x1 + w, y1 + h

    return (x1, y1, x2, y2)

def extract_soda10m_image_id(filepath):
    parts = filepath.split('_')
    image_id = int(parts[2].lstrip('0'))
    return image_id

def extract_soda10m_class_name(classes, class_id):
    for cls in classes:
        if cls["id"] == class_id:
            return cls["supercategory"]