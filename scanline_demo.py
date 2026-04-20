import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches


# calculate th difference between angles a and b
def angle_diff(a, b):
    # return np.arctan2((np.sin(a)+np.sin(-b)), (np.cos(a)+np.cos(-b)))*2
    return np.arctan2((np.sin(a - b)), (np.cos(a - b)))


# simple color classification
class ColorClassifier:
    def __init__(self, bW, bB, bO, cC, cW):
        self.brightnessConeRadiusWhite = bW  # 70
        self.brightnessConeRadiusBlack = bB  # 15
        self.brightnessConeOffset = bO  # 20
        self.colorAngleCenter = cC  # -1.25 # yellow
        self.colorAngleWith = cW  # 0.1

    # return true if the (y,u,v) have a low color part, i.e., white, gray, black
    def no_color(self, y, u, v):
        brightness_alpha = (self.brightnessConeRadiusWhite - self.brightnessConeRadiusBlack) / (
                    255.0 - self.brightnessConeOffset)
        chroma_threshold = np.clip(
            self.brightnessConeRadiusBlack + brightness_alpha * (y - self.brightnessConeOffset),
            self.brightnessConeRadiusBlack, 255)
        chroma = np.hypot(u - 128, v - 128)
        return np.less(chroma, chroma_threshold)

    def is_chroma(self, y, u, v):
        color_angle = np.arctan2(v - 128, u - 128)
        diff = angle_diff(color_angle, self.colorAngleCenter)
        return np.abs(diff) < self.colorAngleWith

    def is_color(self, y, u, v):
        return np.logical_and(np.logical_not(self.no_color(y, u, v)), self.is_chroma(y, u, v))


def load_image(path):
    img = Image.open(path)
    ycbcr = img.convert('YCbCr')

    width = ycbcr.size[0]
    height = ycbcr.size[1]
    size = (height, width)

    # separate channels
    img_y = np.array(list(ycbcr.getdata(band=0)))
    img_u = np.array(list(ycbcr.getdata(band=1)))
    img_v = np.array(list(ycbcr.getdata(band=2)))

    img_y = np.reshape(img_y, size)
    img_u = np.reshape(img_u, size)
    img_v = np.reshape(img_v, size)

    return img, img_y, img_u, img_v


def detect_ball_candidates(image, is_green, step_y=10, step_x=10, min_gap_w=30, max_gap_w=150): #min=20; max = 100
    """
    Scans the image for gaps in green field and returns candidate bounding boxes.
    
    Args:
        image: The input RGB/BGR image.
        is_green: Function that returns True if a pixel is green.
        step_y: Vertical distance between scanlines (skipping rows for speed).
        step_x: Horizontal distance between vertical scanlines (skipping columns for speed).
        min_gap_w: Minimum width of a non-green segment to be considered.
        max_gap_w: Maximum width of a non-green segment (to filter out robots/walls).
    """
    height, width = image.shape[0], image.shape[1]
    print('Image shape:', image.shape)
    print('Image height:', height)

    candidates = []

    # 1. Scan horizontal lines (left to right)
    for y in range(0, height, step_y):
        in_gap = False
        start_x = 0
        
        for x in range(width):
            pixel_is_green = is_green[y, x]
            
            if not pixel_is_green and not in_gap:
                # Started a non-green segment
                start_x = x
                in_gap = True
            elif pixel_is_green and in_gap:
                # Ended a non-green segment
                gap_width = x - start_x
                if min_gap_w <= gap_width <= max_gap_w:
                    candidates.append({'y': y, 'x1': start_x, 'x2': x, 'type': 'horizontal'})
                in_gap = False

    # 2. Scan vertical lines (top to bottom)
    for x in range(0, width, step_x):
        in_gap = False
        start_y = 0
        
        for y in range(height):
            pixel_is_green = is_green[y, x]
            
            if not pixel_is_green and not in_gap:
                # Started a non-green segment
                start_y = y
                in_gap = True
            elif pixel_is_green and in_gap:
                # Ended a non-green segment
                gap_height = y - start_y
                if min_gap_w <= gap_height <= max_gap_w:
                    candidates.append({'x': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                in_gap = False
                    
    return cluster_candidates(candidates)

def cluster_candidates(segments, proximity=20):
    """
    Groups nearby scanline segments into bounding boxes.
    """
    clusters = []
    for seg in segments:
        matched = False
        
        for cluster in clusters:
            # Check for overlap between segments
            if seg['type'] == 'horizontal' and cluster.get('type') == 'horizontal':

                # Two horizontal segments: check vertical and horizontal proximity
                if abs(seg['y'] - cluster['y_max']) <= proximity and \
                    not (seg['x2'] < cluster['x1'] or seg['x1'] > cluster['x2']):
                    # Update cluster bounds
                    cluster['x1'] = min(cluster['x1'], seg['x1'])
                    cluster['x2'] = max(cluster['x2'], seg['x2'])
                    cluster['y_max'] = seg['y']
                    matched = True
                    break
                    
            elif seg['type'] == 'vertical' and cluster.get('type') == 'vertical':
                # Two vertical segments: check horizontal and vertical proximity
                if abs(seg['x'] - cluster['x_max']) <= proximity and \
                    not (seg['y2'] < cluster['y1'] or seg['y1'] > cluster['y2']):
                    # Update cluster bounds
                    cluster['y1'] = min(cluster['y1'], seg['y1'])
                    cluster['y2'] = max(cluster['y2'], seg['y2'])
                    cluster['x_max'] = seg['x']
                    matched = True
                    break
                    
            elif seg['type'] != cluster.get('type'):
                # Mixed horizontal and vertical: check if they intersect/overlap
                if cluster.get('type') == 'horizontal':
                    h_seg = cluster
                    v_seg = seg
                else:
                    h_seg = seg
                    v_seg = cluster
                # Get vertical x coordinate (handle both raw segment and cluster formats)
                v_x = v_seg.get('x', v_seg.get('x1', 0))
                # Get horizontal y coordinate
                h_y = h_seg.get('y', h_seg.get('y1', 0))

                # Check if vertical segment intersects with horizontal segment's region
                h_x1 = h_seg.get('x1', 0)
                h_x2 = h_seg.get('x2', h_x1)
                v_y1 = v_seg.get('y1', 0)
                v_y2 = v_seg.get('y2', v_y1)
                
                if h_x1 <= v_x <= h_x2 and v_y1 <= h_y <= v_y2:
                    # Merge: expand both bounds
                    h_seg['x1'] = min(h_x1, v_x)
                    h_seg['x2'] = max(h_x2, v_x)
                    h_seg['y1'] = min(h_seg.get('y1', h_y), v_y1)
                    h_seg['y2'] = max(h_seg.get('y2', h_y), v_y2)
                    h_seg['type'] = 'merged'
                    matched = True
                    break
        
        if not matched:
            if seg['type'] == 'horizontal':
                clusters.append({
                    'x1': seg['x1'], 
                    'y1': seg['y'], 
                    'x2': seg['x2'], 
                    'y_max': seg['y'],
                    'y2': seg['y'],
                    'type': 'horizontal'
                })
            else:  # vertical
                clusters.append({
                    'x': seg['x'],
                    'x_max': seg['x'],
                    'y1': seg['y1'], 
                    'y2': seg['y2'],
                    'x1': seg['x'],
                    'x2': seg['x'],
                    'type': 'vertical'
                })

    # Convert to (x, y, w, h) format
    bboxes = []
    for c in clusters:
        x1 = c.get('x1', c.get('x', 0))
        x2 = c.get('x2', c.get('x', 0))
        y1 = c.get('y1', c.get('y', 0))
        y2 = c.get('y2', c.get('y_max', 0))
        
        # Ensure valid dimensions
        if x2 > x1 and y2 > y1:
            width = x2 - x1
            height = y2 - y1
            
            # Filter candidates: balls are roughly square and not too elongated
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
            
            # Keep candidates that are not too elongated (filters out field lines)
            # Balls should have aspect ratio close to 1, lines have ratio >> 1
            if aspect_ratio < 1.5:  
                bboxes.append((x1, y1, width, height))
    
            
    # Convert to (x, y, w, h) format for classifiers
    return bboxes

def visualize_candidates(image, candidate_bboxes):
    """
    Draws bounding boxes on the image for debugging.
    
    Args:
        image: The original image (numpy array).
        candidate_bboxes: List of (x, y, w, h) tuples.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.imshow(image)

    for (x, y, w, h) in candidate_bboxes:
        # Create a Rectangle patch
        # We use a bright color (like magenta or cyan) to stand out against green
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2, 
            edgecolor='#FF00FF', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Optional: Add a label
        ax.text(x, y - 5, 'Candidate', color='#FF00FF', fontsize=8, weight='bold')

    plt.title(f"Detected {len(candidate_bboxes)} Potential Ball Regions")
    plt.axis('off')
    plt.show()

# Example usage:
# def my_green_classifier(pixel): ... return True/False
# candidates = detect_ball_candidates(frame, my_green_classifier)
if __name__ == "__main__":
    (img, img_y, img_u, img_v) = load_image("example_frame_rc25.png")

    classifier_green = ColorClassifier(55, 10, 40, np.radians(210), np.radians(25))  # green
    img_green = classifier_green.is_color(img_y, img_u, img_v)

    #print(img_green[0,0])

    candidates = detect_ball_candidates(np.array(img), img_green)
    print(candidates)

    visualize_candidates(np.array(img), candidates)