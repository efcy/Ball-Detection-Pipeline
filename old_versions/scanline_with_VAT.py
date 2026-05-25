import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from pathlib import Path
from vaapi.client import Vaapi
from itertools import islice
import os
import requests

"""
Constants
"""
camera = "TOP"
log_id = 41 
OPENING_ANGLE_DIAGONAL_DEG = 72.6  # from NaoTHSoccer\Config\platform\Nao6\CameraMatrixTop.ini 67.4 (same for bottom)

class CameraInfo:
    width: int
    height: int
    focal_length: float          # in pixels
    optical_center_x: float
    optical_center_y: float
    opening_angle_height: float  # vertical FOV in radians


def make_camera_info(opening_angle_diagonal_deg: float, width=640, height=480):
    """Exact translation of CameraInfo.cpp getter functions."""
    opening_angle_diagonal = np.radians(opening_angle_diagonal_deg)
    
    # getFocalLength()
    half_diag = 0.5 * np.hypot(width, height)
    focal_length = half_diag / np.tan(0.5 * opening_angle_diagonal)
    
    # getOpeningAngleHeight()
    opening_angle_height = 2.0 * np.arctan2(float(height), focal_length * 2.0)
    
    # getOpticalCenterX/Y() — note: integer division, same as C++
    optical_center_x = float(width  // 2)   # = 320.0
    optical_center_y = float(height // 2)   # = 240.0

    return CameraInfo(
        width=width,
        height=height,
        focal_length=focal_length,
        optical_center_x=optical_center_x,
        optical_center_y=optical_center_y,
        opening_angle_height=opening_angle_height
    )


"""
Init API's
"""
v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

def parse_camera_matrix(representation_data: dict) -> np.ndarray:
    """
    T is camera position in world, relative to robot hips.
    R rows are the camera axes expressed in world frame.
    
    To use for projection (world→camera) we need the inverse:
        R_inv = R^T  (rotation inverse)
        T_inv = -R^T @ T

    """
    pose = representation_data["pose"]
    rot  = pose["rotation"]
    t    = pose["translation"]

    R = np.array([
        [rot[0]["x"], rot[0]["y"], rot[0]["z"]],
        [rot[1]["x"], rot[1]["y"], rot[1]["z"]],
        [rot[2]["x"], rot[2]["y"], rot[2]["z"]],
    ])
    T = np.array([t["x"], t["y"], t["z"]])  # in mm, camera pos in world

    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = T
    return M

def camera_matrix_to_world_to_cam(M: np.ndarray) -> np.ndarray:
    """
    Converts the stored pose (cam in world) to world→camera transform,
    which is what the geometry functions need.
    
    If M = [R | T] is cam-in-world, then world→cam is its inverse:
        R_wc = R^T
        T_wc = -R^T @ T
    """
    R = M[:3, :3]
    T = M[:3,  3]
    
    R_inv = R.T
    T_inv = -R.T @ T
    
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3,  3] = T_inv
    return M_inv


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


def detect_ball_candidates(image, is_green, step_y=10, step_x=10, min_gap_w=20, max_gap_w=100): #min=20; max = 100
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

    candidates = []
    gap_segments = [] 

    scanlines_h = list(range(0, height, step_y))
    scanlines_v = list(range(0, width, step_x))

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
                    gap_segments.append({'x1': start_x, 'x2': x, 'y1': y, 'y2': y, 'type': 'horizontal'})
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
                    gap_segments.append({'x1': x, 'x2': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                in_gap = False
                    
    return cluster_candidates(candidates), scanlines_h, scanlines_v, gap_segments

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
                    cluster['x1'] = min(cluster['x1'], seg['x1'])
                    cluster['x2'] = max(cluster['x2'], seg['x2'])
                    cluster['y_min'] = min(cluster['y_min'], seg['y']) 
                    cluster['y_max'] = max(cluster['y_max'], seg['y'])
                    matched = True
                    break
                    
            elif seg['type'] == 'vertical' and cluster.get('type') == 'vertical':
                if abs(seg['x'] - cluster['x_max']) <= proximity and \
                    not (seg['y2'] < cluster['y1'] or seg['y1'] > cluster['y2']):
                    # Update cluster bounds
                    cluster['y1'] = min(cluster['y1'], seg['y1'])
                    cluster['y2'] = max(cluster['y2'], seg['y2'])
                    cluster['x_min'] = min(cluster['x_min'], seg['x']) 
                    cluster['x_max'] = max(cluster['x_max'], seg['x']) 
                    matched = True
                    break
                    
            elif seg['type'] != cluster.get('type'):
                if cluster.get('type') == 'horizontal':
                    h_cluster = cluster
                    v_seg_data = seg
                else:
                    h_cluster = cluster
                    v_seg_data = seg  # cluster is always the thing we modify

                v_x = v_seg_data.get('x', v_seg_data.get('x1', 0))
                h_x1 = h_cluster.get('x1', 0)
                h_x2 = h_cluster.get('x2', h_x1)
                v_y1 = v_seg_data.get('y1', 0)
                v_y2 = v_seg_data.get('y2', v_y1)
                h_y_min = h_cluster.get('y_min', h_cluster.get('y1', h_cluster.get('y', 0)))
                h_y_max = h_cluster.get('y_max', h_cluster.get('y2', h_y_min))

                PROX = 15
                if (h_x1 - PROX) <= v_x <= (h_x2 + PROX) and \
                (v_y1 - PROX) <= h_y_max and (v_y2 + PROX) >= h_y_min:
                    h_cluster['x1'] = min(h_x1, v_x)
                    h_cluster['x2'] = max(h_x2, v_x)
                    h_cluster['y1'] = min(h_y_min, v_y1)
                    h_cluster['y2'] = max(h_y_max, v_y2)
                    h_cluster['y_min'] = h_cluster['y1']
                    h_cluster['y_max'] = h_cluster['y2']
                    h_cluster['type'] = 'merged'
                    matched = True
                    break
        
        if not matched:
            if seg['type'] == 'horizontal':
                clusters.append({
                    'x1': seg['x1'], 
                    'y1': seg['y'], 
                    'x2': seg['x2'], 
                    'y_min': seg['y'],
                    'y_max': seg['y'],
                    'y2': seg['y'],
                    'type': 'horizontal'
                })
            else:  # vertical
                clusters.append({
                    'x': seg['x'],
                    'x_min': seg['x'],
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
        if c['type'] == 'horizontal':
            x1 = c['x1']
            x2 = c['x2']
            y1 = c['y_min']
            y2 = c['y_max']
        elif c['type'] == 'vertical':
            x1 = c['x_min']  
            x2 = c['x_max']
            y1 = c['y1']
            y2 = c['y2']
        else:  # merged
            x1 = c.get('x1', 0)
            x2 = c.get('x2', 0)
            y1 = c.get('y1', 0)
            y2 = c.get('y2', 0)
            
        width = x2 - x1
        height = y2 - y1

        # if height or width is zero (single-axis cluster)
        # synthesize the missing dimension from the known one

        if width == 0 or height == 0:
            continue
        if x2 > x1 and y2 > y1:
            # Filter candidates: balls are roughly square and not too elongated
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
            # Keep candidates that are not too elongated (filters out field lines)
            # Balls should have aspect ratio close to 1, lines have ratio >> 1
            if aspect_ratio < 10:
                bboxes.append((x1, y1, width, height))

    # Convert to (x, y, w, h) format for classifiers
    return bboxes

def visualize_candidates(image, candidate_bboxes, scanlines_h=None, scanlines_v=None,  gap_segments=None, step_y=10, step_x=10):
    """
    Draws bounding boxes and scanlines on the image for debugging.
    Shows the direction of each scanline type.
    
    Args:
        image: The original image (numpy array).
        candidate_bboxes: List of (x, y, w, h) tuples.
        scanlines_h: Optional list of horizontal scanline y-coordinates.
        scanlines_v: Optional list of vertical scanline x-coordinates.
        step_y: Vertical step used in scanning.
        step_x: Horizontal step used in scanning.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    ax.imshow(image)
    height, width = image.shape[0], image.shape[1]
    
    # Draw horizontal scanlines (scanned left to right)
    #if scanlines_h is None:
    #    scanlines_h = list(range(0, height, step_y))
    if scanlines_h is not None:
        for y in scanlines_h:
            ax.plot([0, width], [y, y], color='cyan', linewidth=0.8, alpha=0.4, linestyle='--')

        # Add direction indicators for horizontal scanlines (arrows pointing right)
        arrow_spacing = max(1, len(scanlines_h) // 5)  # Show ~5 arrows to avoid clutter
        for i, y in enumerate(scanlines_h):
            if i % arrow_spacing == 0:
                ax.annotate('', xy=(width-20, y), xytext=(width-60, y),
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=2, alpha=0.8))
        
    # Draw vertical scanlines 
    #if scanlines_v is None:
    #    scanlines_v = list(range(0, width, step_x))
    if scanlines_v is not None:
        for x in scanlines_v:
            ax.plot([x, x], [0, height], color='lime', linewidth=0.8, alpha=0.4, linestyle='--')    

        # Add direction indicators for vertical scanlines
        arrow_spacing = max(1, len(scanlines_v) // 5)  # Show ~5 arrows to avoid clutter
        for i, x in enumerate(scanlines_v):
            if i % arrow_spacing == 0:
                ax.annotate('', xy=(x, height-20), xytext=(x, height-60),
                        arrowprops=dict(arrowstyle='->', color='lime', lw=2, alpha=0.8))
    # Draw raw non-green gap segments
    if gap_segments:
        for seg in gap_segments:
            if seg['type'] == 'horizontal':
                ax.plot([seg['x1'], seg['x2']], [seg['y1'], seg['y1']], 
                        color='orange', linewidth=2.5, alpha=0.8, solid_capstyle='round')
            else:  # vertical
                ax.plot([seg['x1'], seg['x1']], [seg['y1'], seg['y2']], 
                        color='yellow', linewidth=2.5, alpha=0.8, solid_capstyle='round')

    
    # Draw detected candidate bounding boxes
    for (x, y, w, h) in candidate_bboxes:
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2.5, 
            edgecolor='#FF00FF', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(x, y - 8, 'Ball Candidate', color='#FF00FF', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', linewidth=2, label='Horizontal scanlines (L→R)', linestyle='--'),
        Line2D([0], [0], color='lime', linewidth=2, label='Vertical scanlines (T→B)', linestyle='--'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='#FF00FF', 
               markersize=10, linewidth=2.5, label='Ball candidates'),
        Line2D([0], [0], color='orange', linewidth=2, label='Horizontal Raw non-green gap segments', linestyle='--'),
        Line2D([0], [0], color='yellow', linewidth=2, label='Vertical Raw non-green gap segments', linestyle='--'),
        
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.title(f"Ball Detection: {len(candidate_bboxes)} Candidates | Scanlines with Direction Indicators", 
             fontsize=13, weight='bold')
    ax.set_xlabel(f"Horizontal scanlines every {step_y} pixels | Vertical scanlines every {step_x} pixels", 
                 fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
        
    CAMERA_INFO_TOP    = make_camera_info(opening_angle_diagonal_deg = 72.6)
    CAMERA_INFO_BOTTOM = make_camera_info(opening_angle_diagonal_deg = 72.6)  # same hardware

    # Quick check — print what this gives:
    cam = CAMERA_INFO_TOP
    print(f"focal_length:         {cam.focal_length:.4f} px")
    print(f"opening_angle_height: {np.degrees(cam.opening_angle_height):.4f} deg")
    print(f"optical_center:       {cam.optical_center_x}, {cam.optical_center_y}")
    # focal_length:         572.pppp px  
    # opening_angle_height: ~44.pp deg
    # optical_center:       320.0, 240.0

    frame_data = {}
    image_obj_list = v_client.image.list(log=log_id, camera=camera)
    first_batch = list(islice(image_obj_list, 5))

    for img_obj in first_batch:
        img_url = "https://logs.berlin-united.com/" + img_obj.image_url
        frame_id = img_obj.frame.id
        
        response = requests.get(img_url)
        if response.status_code == 200:
            file_path = Path("./test_images") / f"{frame_id}.jpg"
            with open(file_path, "wb") as f:
                f.write(response.content)

        if camera == "TOP":
            cm_list = v_client.cameramatrixtop.list(frame=frame_id)
        else:
            cm_list = v_client.cameramatrix.list(frame=frame_id)

        cm_list_all = list(islice(cm_list, 3))
        print(dir(v_client))
        if len(cm_list_all) == 0:
            print(" No camera matrix found for this frame!", frame_id)            
        else:
            first = cm_list_all[0]
            rep = first.representation_data
            print(f"representation_data: {rep}")

        cam_matrix = None
        for cm in islice(cm_list, 1):  
            cam_pose   = parse_camera_matrix(cm.representation_data)      # cam in world
            cam_matrix = camera_matrix_to_world_to_cam(cam_pose)          # world→cam (for geometry)

            # Quick confirmation print:
            R = cam_pose[:3, :3]
            T = cam_pose[:3,  3]
            print(f"Camera world pos: x={T[0]:.1f}  y={T[1]:.1f}  z={T[2]:.1f}mm")
            
            # Test ball radius at a few rows
            for test_y in [100, 200, 300, 400]:
                r = estimated_ball_radius_px(cam_matrix, CAMERA_INFO_TOP, 50.0, 320, test_y)
                print(f"  row {test_y}: expected ball radius = {r:.1f}px")

        # save both together 
        frame_data[frame_id] = {
            "image_path": file_path,
            "cam_matrix": cam_matrix,
        }      

    # Images from Labor C:\Users\anina\Documents\Studium_Berlin\Study\RoboCup2026\naoth-deeplearning\balldetection2026\patch_based_training\data\TOP\images\
#     for path in Path("test_images").iterdir():
#         if path.is_file():
#             (img, img_y, img_u, img_v) = load_image(path)

#             classifier_green = ColorClassifier(55, 10, 40, np.radians(210), np.radians(25))  # green
#             img_green = classifier_green.is_color(img_y, img_u, img_v)

#             #print(img_green[0,0])

# #            candidates, scanlines_h, scanlines_v, gap_segments = detect_ball_candidates(np.array(img), img_green)            
#             candidates, _, _, gap_segments = detect_ball_candidates(np.array(img), img_green)            
           
#             visualize_candidates(np.array(img), candidates,  
#                      gap_segments=gap_segments)