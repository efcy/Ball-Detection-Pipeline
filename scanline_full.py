import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
from vaapi.client import Vaapi
from itertools import islice
import os
import requests

"""
Constants
"""
camera = "TOP"
log_id =675
OPENING_ANGLE_DIAGONAL_DEG = 72.6  # from NaoTHSoccer\Config\platform\Nao6\CameraMatrixTop.ini; same for bottom camera
BALL_RADIUS = 70.0  # mm, RoboCup standard ball size after 2026

# CameraInfo
@dataclass
class CameraInfo:
    width: int
    height: int
    focal_length: float          # in pixels
    optical_center_x: float
    optical_center_y: float
    opening_angle_height: float  # vertical FOV in radians

# CameraInfo.cpp getter functions translated to Python
def make_camera_info(opening_angle_diagonal_deg: float, width=640, height=480) -> CameraInfo:
    """
    Exact translation of CameraInfo.cpp getter functions.
    Derives focal length and vertical FOV from the diagonal FOV angle.
    """
    opening_angle_diagonal = np.radians(opening_angle_diagonal_deg)

    # getFocalLength(): half-diagonal in pixels / tan(half FOV angle)
    half_diag = 0.5 * np.hypot(width, height)
    focal_length = half_diag / np.tan(0.5 * opening_angle_diagonal)

    # getOpeningAngleHeight(): vertical FOV from focal length
    opening_angle_height = 2.0 * np.arctan2(float(height), focal_length * 2.0)

    # getOpticalCenterX/Y(): integer division, same as C++
    optical_center_x = float(width  // 2)   # = 320.0
    optical_center_y = float(height // 2)   # = 240.0

    return CameraInfo(
        width=width,
        height=height,
        focal_length=focal_length,
        optical_center_x=optical_center_x,
        optical_center_y=optical_center_y,
        opening_angle_height=opening_angle_height,
    )


# Camera Matrix parsing and inversion
# cpp Serializer<CameraMatrix>
def parse_camera_matrix(representation_data: dict) -> np.ndarray:
    """
    Parses the API representation_data into a 4x4 cam-in-world pose matrix.

    M = [R | T] where:
      R  — camera axes expressed in robot/world frame (rows = x, y, z camera axes)
      T  — camera position in robot/world frame (mm)

    Usage:
      world_point = M @ cam_point     (cam → world)
      cam_point   = M_inv @ world_pt  (world → cam, use camera_matrix_to_world_to_cam)
    """
    pose = representation_data["pose"]
    rot  = pose["rotation"]
    t    = pose["translation"]

    R = np.array([
        [rot[0]["x"], rot[0]["y"], rot[0]["z"]],
        [rot[1]["x"], rot[1]["y"], rot[1]["z"]],
        [rot[2]["x"], rot[2]["y"], rot[2]["z"]],
    ])
    T = np.array([t["x"], t["y"], t["z"]])  # camera position in world (mm)

    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = T
    return M

# cpp Pose3D::invert() for rigid body transform
def camera_matrix_to_world_to_cam(M: np.ndarray) -> np.ndarray:
    """
    Inverts a cam-in-world pose into a world-to-camera transform.

    Since M is a rigid body transform (orthonormal R):
      R_inv = R^T
      T_inv = -R^T @ T
    """
    R = M[:3, :3]
    T = M[:3,  3]

    R_inv = R.T
    T_inv = -R.T @ T

    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3,  3] = T_inv
    return M_inv



# Camera geometry — pixel field projection
def image_pixel_to_camera_coords(cam_info: CameraInfo, img_x: float, img_y: float) -> np.ndarray:
    """
    Converts a pixel (img_x, img_y) into a direction ray in camera space.
    Direct translation of imagePixelToCameraCoords() from C++.

    Camera convention: X = forward (optical axis), Y = left, Z = up.
    focal_length is the X component — it sets the "depth" of the virtual image plane.
    optical_center offsets map pixel origin (top-left) to camera center.
    """
    x = cam_info.focal_length
    y = cam_info.optical_center_x - 0.5 - img_x   # pixel-right  → cam-left  (flip)
    z = cam_info.optical_center_y - 0.5 - img_y   # pixel-down   → cam-up    (flip)
    return np.array([x, y, z])

#CameraGeometry.cpp CameraGeometry::imagePixelToFieldCoord() !CHECK!
def image_pixel_to_field_coord(
    cam_pose_world: np.ndarray,
    cam_info: CameraInfo,
    img_x: float,
    img_y: float,
    object_height: float,
) -> np.ndarray | None:
    """
    Projects a pixel back onto the horizontal field plane at a given height above ground.
    Returns (x, y) in robot/world coordinates, or None if the projection is impossible.

    Impossible cases (mirrors the C++ epsilon guard):
      - Ray is horizontal (pixel_vec_world[2] ≈ 0) → never reaches target height
      - Ray points away from target height (signs differ) → looking above horizon
        when target is below camera, or vice versa
    """
    epsilon = 1e-13

    R = cam_pose_world[:3, :3]   # camera axes in world frame
    T = cam_pose_world[:3,  3]   # camera position in world (mm)

    # Build the ray direction: pixel → camera space → rotate into world space
    pixel_vec_cam   = image_pixel_to_camera_coords(cam_info, img_x, img_y)
    pixel_vec_world = R @ pixel_vec_cam

    # Vertical gap between camera and target plane (negative when camera is above target)
    height_diff = object_height - T[2]

    # Guard: pixel_vec_world[2] is the ray's vertical component.
    # The product must be positive: both pointing same direction toward target.
    if pixel_vec_world[2] * height_diff < epsilon:
        return None

    # Parameter t: how far along the ray to reach object_height
    # world_point = cam_pos + t * ray_direction → solve for t from Z component
    t = height_diff / pixel_vec_world[2]

    field_x = T[0] + t * pixel_vec_world[0]
    field_y = T[1] + t * pixel_vec_world[1]
    return np.array([field_x, field_y])

#CameraGeometry.cpp estimateBallRadiusInPixels() !CHECK!
def estimated_ball_radius_px(
    cam_pose_world: np.ndarray,
    cam_info: CameraInfo,
    ball_radius_mm: float,
    img_x: float,
    img_y: float,
) -> float:
    """
    Estimates the expected ball radius in pixels for a ball seen at pixel (img_x, img_y).
    Returns -1.0 if the projection is geometrically impossible.

    Steps:
      1. Project pixel → field plane at height = ball_radius_mm
         → gives horizontal ball position (x, y) on the field
      2. Reconstruct full 3D ball center: (field_x, field_y, ball_radius_mm)
         z = ball_radius because ball rests on ground → center is one radius above ground
      3. Transform ball center into camera space → air distance from lens to ball center
      4. Angular diameter → pixel radius:
         alpha = atan2(r, d)   [half-angle subtended by ball]
         pixel_r = alpha / vertical_FOV * image_height
    """
    # Step 1: project pixel onto field plane at ball center height
    point_on_field = image_pixel_to_field_coord(
        cam_pose_world, cam_info, img_x, img_y, ball_radius_mm
    )
    if point_on_field is None:
        return -1.0

    # Step 2: full 3D ball center in world coordinates (homogeneous)
    ball_center_world = np.array([
        point_on_field[0],
        point_on_field[1],
        ball_radius_mm,     # z = one radius above ground
        1.0,                # homogeneous coordinate
    ])

    # Step 3: transform into camera space, get straight-line distance
    cam_matrix_world_to_cam = camera_matrix_to_world_to_cam(cam_pose_world)
    ball_in_cam = cam_matrix_world_to_cam @ ball_center_world
    cam_ball_distance = np.linalg.norm(ball_in_cam[:3])

    # Guard: camera must be outside the ball (distance > radius)
    # If inside, the tangent-line angular formula below breaks down
    if cam_ball_distance <= ball_radius_mm:
        return -1.0

    # Step 4: angular half-diameter → pixels
    alpha = np.arctan2(ball_radius_mm, cam_ball_distance)
    return alpha / cam_info.opening_angle_height * cam_info.height


# ==============================================================================
# Color classifier
# ==============================================================================

def angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


class ColorClassifier:
    def __init__(self, bW, bB, bO, cC, cW):
        self.brightnessConeRadiusWhite = bW   # e.g. 70
        self.brightnessConeRadiusBlack = bB   # e.g. 15
        self.brightnessConeOffset      = bO   # e.g. 20
        self.colorAngleCenter          = cC   # e.g. -1.25 (yellow)
        self.colorAngleWith            = cW   # e.g. 0.1

    def no_color(self, y, u, v):
        """Returns True where pixel has low saturation (white / gray / black)."""
        brightness_alpha = (self.brightnessConeRadiusWhite - self.brightnessConeRadiusBlack) / (
            255.0 - self.brightnessConeOffset)
        chroma_threshold = np.clip(
            self.brightnessConeRadiusBlack + brightness_alpha * (y - self.brightnessConeOffset),
            self.brightnessConeRadiusBlack, 255)
        chroma = np.hypot(u - 128, v - 128)
        return np.less(chroma, chroma_threshold)

    def is_chroma(self, y, u, v):
        """Returns True where pixel hue matches the configured color angle."""
        color_angle = np.arctan2(v - 128, u - 128)
        diff = angle_diff(color_angle, self.colorAngleCenter)
        return np.abs(diff) < self.colorAngleWith

    def is_color(self, y, u, v):
        return np.logical_and(np.logical_not(self.no_color(y, u, v)), self.is_chroma(y, u, v))


# ==============================================================================
# Image loading
# ==============================================================================

def load_image(path):
    img   = Image.open(path)
    ycbcr = img.convert('YCbCr')
    width, height = ycbcr.size
    size = (height, width)

    img_y = np.reshape(np.array(list(ycbcr.getdata(band=0))), size)
    img_u = np.reshape(np.array(list(ycbcr.getdata(band=1))), size)
    img_v = np.reshape(np.array(list(ycbcr.getdata(band=2))), size)

    return img, img_y, img_u, img_v

def find_field_top_row(is_green: np.ndarray, min_green_fraction: float = 0.05) -> int:
    """
    Returns the first row where green pixels exceed min_green_fraction of row width.
    This is the upper boundary of the field — scanlines should start here.
    """
    height, width = is_green.shape[:2]
    for y in range(height):
        green_count = np.sum(is_green[y, :])
        if green_count / width >= min_green_fraction:
            return y
    return 0  # no green found, fall back to full image scan

# ==============================================================================
# Ball candidate detection — fixed step version (original)
# ==============================================================================

def detect_ball_candidates(image, is_green, step_y=10, step_x=10, min_gap_w=20, max_gap_w=100):
    """
    Scans the image for gaps in green field and returns candidate bounding boxes.

    Args:
        image:     Input image (numpy array).
        is_green:  Boolean mask — True where pixel is classified as green.
        step_y:    Vertical distance between horizontal scanlines.
        step_x:    Horizontal distance between vertical scanlines.
        min_gap_w: Minimum non-green segment width to consider.
        max_gap_w: Maximum non-green segment width (filters out robots/walls).
    """
    height, width = image.shape[0], image.shape[1]
    candidates  = []
    gap_segments = []

    scanlines_h = list(range(0, height, step_y))
    scanlines_v = list(range(0, width,  step_x))

    # Horizontal scanlines (left → right)
    for y in range(0, height, step_y):
        in_gap  = False
        start_x = 0
        for x in range(width):
            pixel_is_green = is_green[y, x]
            if not pixel_is_green and not in_gap:
                start_x = x
                in_gap  = True
            elif pixel_is_green and in_gap:
                gap_width = x - start_x
                if min_gap_w <= gap_width <= max_gap_w:
                    candidates.append({'y': y, 'x1': start_x, 'x2': x, 'type': 'horizontal'})
                    gap_segments.append({'x1': start_x, 'x2': x, 'y1': y, 'y2': y, 'type': 'horizontal'})
                in_gap = False

    # Vertical scanlines (top → bottom)
    for x in range(0, width, step_x):
        in_gap  = False
        start_y = 0
        for y in range(height):
            pixel_is_green = is_green[y, x]
            if not pixel_is_green and not in_gap:
                start_y = y
                in_gap  = True
            elif pixel_is_green and in_gap:
                gap_height = y - start_y
                if min_gap_w <= gap_height <= max_gap_w:
                    candidates.append({'x': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                    gap_segments.append({'x1': x, 'x2': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                in_gap = False

    return cluster_candidates(candidates), scanlines_h, scanlines_v, gap_segments


# ==============================================================================
# Ball candidate detection — adaptive step version (geometry-aware)
# ==============================================================================

def detect_ball_candidates_adaptive(
    image,
    is_green,
    cam_pose_world: np.ndarray,
    cam_info: CameraInfo,
    ball_radius_mm: float = BALL_RADIUS,
    step_scale: float = 0.35,        # scanline step = step_scale × expected ball diameter
    size_tolerance: float = 0.5,    # accept patches within ±50% of expected radius
):
    """
    Geometry-aware version: scanline density and patch filtering both adapt to
    the expected ball size at each image row.

    For each row we compute estimated_ball_radius_px() at the image center column.
    This gives a per-row expected radius r_px, then:
      - vertical step between scanlines ≈ step_scale × 2 × r_px
        (denser near the bottom where balls are large and close)
      - patches whose radius falls outside [r_px*(1-tol), r_px*(1+tol)] are rejected
        (removes field lines, robot legs, and other false positives)

    Args:
        image:          Input image (numpy array).
        is_green:       Boolean mask — True where pixel is green.
        cam_pose_world: 4×4 cam-in-world matrix from parse_camera_matrix().
        cam_info:       CameraInfo for this camera.
        ball_radius_mm: Physical ball radius in mm.
        step_scale:     Controls scanline density. 0.5 → one scanline per ball-radius height.
        size_tolerance: Fraction of expected radius to accept (0.5 = ±50%).
    """
    height, width = image.shape[:2]
    cx = width // 2   # use image center column for radius estimates

    # --- Build per-row expected ball radius lookup (geometry only, cheap) ---
    row_expected_radius = {}
    for y in range(height):
        r = estimated_ball_radius_px(cam_pose_world, cam_info, ball_radius_mm, cx, y)
        row_expected_radius[y] = r   # -1.0 means above horizon / invalid

    # --- Adaptive horizontal scanlines ---
    # Jump by the expected ball diameter at each row so we never skip over a ball
    field_top = find_field_top_row(is_green, min_green_fraction=0.05)
    for y in range(field_top, height):
        r = estimated_ball_radius_px(cam_pose_world, cam_info, ball_radius_mm, cx, y)
        row_expected_radius[y] = r
        
    scanlines_h = []
    y = field_top         
    while y < height:
        scanlines_h.append(y)
        r    = row_expected_radius.get(y, -1.0)
        step = max(2, int(step_scale * 2 * r)) if r > 0 else 10
        y   += step

    # --- Adaptive vertical scanlines ---
    # Use the median expected radius in the lower half of the image as a guide
    scanlines_v = []
    x = 0
    while x < width:
        scanlines_v.append(x)
        representative_y = field_top + int((height - field_top) * 0.5)
        r_col = estimated_ball_radius_px(cam_pose_world, cam_info, ball_radius_mm, x, representative_y)
        step = max(2, int(step_scale * 2 * r_col)) if r_col > 0 else 10
        x += step

    # --- Derive min/max gap sizes from geometry ---
    # Smallest balls appear at the top of the field (far away), largest near the bottom
    r_far  = max(row_expected_radius.get(scanlines_h[0],  5.0), 3.0)
    r_near = max(row_expected_radius.get(scanlines_h[-1], 60.0), r_far + 5.0)
    r_mid = float(np.median(list(v for v in row_expected_radius.values() if v > 0))) if row_expected_radius else 15
    cluster_proximity = max(5, int(r_mid * 0.6)) 

    min_gap = max(3,  int(r_far  * (1.0 - size_tolerance) * 2))
    max_gap = max(80, int(r_near * (1.0 + size_tolerance) * 2))

    # --- Run the core scan using our adaptive scanlines ---
    candidates_raw, _, _, gap_segments = _scan_on_given_lines(
        image, is_green,
        scanlines_h=scanlines_h,
        scanlines_v=scanlines_v,
        min_gap_w=min_gap,
        max_gap_w=max_gap,
        field_top=field_top, 
        proximity=cluster_proximity,
        
    )

    # --- Filter patches by geometric size expectation ---
    filtered = []
    for (x, y, w, h) in candidates_raw:
        patch_center_x = x + w // 2
        patch_center_y = y + h // 2
        r_expected = estimated_ball_radius_px(
            cam_pose_world, cam_info, ball_radius_mm,
            patch_center_x, patch_center_y
        )
        if r_expected <= 0:
            continue

        aspect = w / h if h > 0 else 999
        halves = (
            [(x, y, w//2, h), (x + w//2, y, w//2, h)]
            if aspect > 1.8 else
            [(x, y, w, h)]
        )

        for (hx, hy, hw, hh) in halves:
            patch_radius = (hw + hh) / 4.0
            lo = r_expected * (1.0 - size_tolerance)
            hi = r_expected * (1.0 + size_tolerance)
            if lo <= patch_radius <= hi:
                filtered.append((hx, hy, hw, hh))
        print(f"  patch ({x},{y}) w={w} h={h} r_patch={patch_radius:.1f} "
            f"r_exp={r_expected:.1f} lo={lo:.1f} hi={hi:.1f} "
            f"→ {'PASS' if lo <= patch_radius <= hi else 'REJECT'}")
        
    return filtered, scanlines_h, scanlines_v, gap_segments


def _scan_on_given_lines(image, is_green, scanlines_h, scanlines_v, min_gap_w, max_gap_w, field_top: int = 0, proximity: int = 15):
    """
    Internal helper: runs the gap-scan on pre-computed scanline positions.
    Mirrors detect_ball_candidates() but accepts explicit line lists instead of steps.
    """
    height, width = image.shape[:2]
    candidates   = []
    gap_segments = []

    for y in scanlines_h:
        if y >= height:
            continue
        in_gap  = False
        start_x = 0
        for x in range(width):
            green = is_green[y, x]
            if not green and not in_gap:
                start_x = x
                in_gap  = True
            elif green and in_gap:
                gw = x - start_x
                if min_gap_w <= gw <= max_gap_w:
                    candidates.append({'y': y, 'x1': start_x, 'x2': x, 'type': 'horizontal'})
                    gap_segments.append({'x1': start_x, 'x2': x, 'y1': y, 'y2': y, 'type': 'horizontal'})
                in_gap = False

    for x in scanlines_v:
        if x >= width:
            continue
        in_gap  = False
        start_y = 0
        for y in range(field_top, height):
            green = is_green[y, x]
            if not green and not in_gap:
                start_y = y
                in_gap  = True
            elif green and in_gap:
                gh = y - start_y
                if min_gap_w <= gh <= max_gap_w:
                    candidates.append({'x': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                    gap_segments.append({'x1': x, 'x2': x, 'y1': start_y, 'y2': y, 'type': 'vertical'})
                in_gap = False

    return cluster_candidates(candidates, proximity=proximity), scanlines_h, scanlines_v, gap_segments


# ==============================================================================
# Clustering
# ==============================================================================

def cluster_candidates(segments, proximity=15):
    """
    Groups nearby scanline segments into bounding boxes (x, y, w, h).
    """
    clusters = []
    for seg in segments:
        matched = False
        for cluster in clusters:
            if seg['type'] == 'horizontal' and cluster.get('type') == 'horizontal':
                if abs(seg['y'] - cluster['y_max']) <= proximity and \
                        not (seg['x2'] < cluster['x1'] or seg['x1'] > cluster['x2']):
                    cluster['x1']   = min(cluster['x1'],   seg['x1'])
                    cluster['x2']   = max(cluster['x2'],   seg['x2'])
                    cluster['y_min'] = min(cluster['y_min'], seg['y'])
                    cluster['y_max'] = max(cluster['y_max'], seg['y'])
                    matched = True
                    break

            elif seg['type'] == 'vertical' and cluster.get('type') == 'vertical':
                if abs(seg['x'] - cluster['x_max']) <= proximity and \
                        not (seg['y2'] < cluster['y1'] or seg['y1'] > cluster['y2']):
                    cluster['y1']   = min(cluster['y1'],   seg['y1'])
                    cluster['y2']   = max(cluster['y2'],   seg['y2'])
                    cluster['x_min'] = min(cluster['x_min'], seg['x'])
                    cluster['x_max'] = max(cluster['x_max'], seg['x'])
                    matched = True
                    break

            elif seg['type'] != cluster.get('type'):
                if cluster.get('type') == 'horizontal':
                    h_cluster  = cluster
                    v_seg_data = seg
                else:
                    h_cluster  = cluster
                    v_seg_data = seg

                v_x    = v_seg_data.get('x',    v_seg_data.get('x1', 0))
                h_x1   = h_cluster.get('x1', 0)
                h_x2   = h_cluster.get('x2', h_x1)
                v_y1   = v_seg_data.get('y1', 0)
                v_y2   = v_seg_data.get('y2', v_y1)
                h_y_min = h_cluster.get('y_min', h_cluster.get('y1', h_cluster.get('y', 0)))
                h_y_max = h_cluster.get('y_max', h_cluster.get('y2', h_y_min))

                
                if (h_x1 - proximity) <= v_x <= (h_x2 + proximity) and \
                        (v_y1 - proximity) <= h_y_max and (v_y2 + proximity) >= h_y_min:
                    h_cluster['x1']   = min(h_x1, v_x)
                    h_cluster['x2']   = max(h_x2, v_x)
                    h_cluster['y1']   = min(h_y_min, v_y1)
                    h_cluster['y2']   = max(h_y_max, v_y2)
                    h_cluster['y_min'] = h_cluster['y1']
                    h_cluster['y_max'] = h_cluster['y2']
                    h_cluster['type']  = 'merged'
                    matched = True
                    break

        if not matched:
            if seg['type'] == 'horizontal':
                clusters.append({'x1': seg['x1'], 'y1': seg['y'], 'x2': seg['x2'],
                                  'y_min': seg['y'], 'y_max': seg['y'], 'y2': seg['y'],
                                  'type': 'horizontal'})
            else:
                clusters.append({'x': seg['x'], 'x_min': seg['x'], 'x_max': seg['x'],
                                  'y1': seg['y1'], 'y2': seg['y2'],
                                  'x1': seg['x'], 'x2': seg['x'], 'type': 'vertical'})

    bboxes = []
    for c in clusters:
        if c['type'] == 'horizontal':
            x1, x2, y1, y2 = c['x1'], c['x2'], c['y_min'], c['y_max']
        elif c['type'] == 'vertical':
            x1, x2, y1, y2 = c['x_min'], c['x_max'], c['y1'], c['y2']
        else:
            x1, x2, y1, y2 = c.get('x1', 0), c.get('x2', 0), c.get('y1', 0), c.get('y2', 0)

        w = x2 - x1
        h = y2 - y1

        if w == 0 or h == 0:
            continue
        if x2 > x1 and y2 > y1:
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 3:
                bboxes.append((x1, y1, w, h))

    return bboxes


# Visualisation
def visualize_candidates(image, candidate_bboxes, scanlines_h=None, scanlines_v=None,
                          gap_segments=None, step_y=10, step_x=10,
                          save_path: Path | None = None):
    """
    Draws bounding boxes, scanlines, and gap segments on the image for debugging.
    If save_path is given the figure is saved there instead of being shown.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(image)
    height, width = image.shape[0], image.shape[1]

    if scanlines_h is not None:
        for y in scanlines_h:
            ax.plot([0, width], [y, y], color='cyan', linewidth=0.8, alpha=0.4, linestyle='--')
    #     arrow_spacing = max(1, len(scanlines_h) // 5)
    #     for i, y in enumerate(scanlines_h):
    #         if i % arrow_spacing == 0:
    #             ax.annotate('', xy=(width - 20, y), xytext=(width - 60, y),
    #                         arrowprops=dict(arrowstyle='->', color='cyan', lw=2, alpha=0.8))

    if scanlines_v is not None:
        for x in scanlines_v:
            ax.plot([x, x], [0, height], color='lime', linewidth=0.8, alpha=0.4, linestyle='--')
    #     arrow_spacing = max(1, len(scanlines_v) // 5)
    #     for i, x in enumerate(scanlines_v):
    #         if i % arrow_spacing == 0:
    #             ax.annotate('', xy=(x, height - 20), xytext=(x, height - 60),
    #                         arrowprops=dict(arrowstyle='->', color='lime', lw=2, alpha=0.8))

    if gap_segments:
        for seg in gap_segments:
            if seg['type'] == 'horizontal':
                ax.plot([seg['x1'], seg['x2']], [seg['y1'], seg['y1']],
                        color='orange', linewidth=2.5, alpha=0.8, solid_capstyle='round')
            else:
                ax.plot([seg['x1'], seg['x1']], [seg['y1'], seg['y2']],
                        color='yellow', linewidth=2.5, alpha=0.8, solid_capstyle='round')

    for (x, y, w, h) in candidate_bboxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=2.5,
                                  edgecolor='#FF00FF', facecolor='none')
        ax.add_patch(rect)
        

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan',   linewidth=2, linestyle='--', label='Horizontal scanlines (L→R)'),
        Line2D([0], [0], color='lime',   linewidth=2, linestyle='--', label='Vertical scanlines (T→B)'),
        Line2D([0], [0], marker='s',     color='w',  markerfacecolor='none',
               markeredgecolor='#FF00FF', markersize=10, linewidth=2.5, label='Ball candidates'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Horizontal non-green gaps'),
        Line2D([0], [0], color='yellow', linewidth=2, linestyle='--', label='Vertical non-green gaps'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    plt.title(f"Ball Detection: {len(candidate_bboxes)} Candidates | Adaptive Scanlines", fontsize=13, weight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved visualisation → {save_path}")
    else:
        plt.show()

    plt.close(fig)  


# ==============================================================================
# API init
# ==============================================================================

v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    CAMERA_INFO_TOP    = make_camera_info(opening_angle_diagonal_deg=OPENING_ANGLE_DIAGONAL_DEG)
    CAMERA_INFO_BOTTOM = make_camera_info(opening_angle_diagonal_deg=OPENING_ANGLE_DIAGONAL_DEG)

    cam = CAMERA_INFO_TOP
    # print(f"focal_length:         {cam.focal_length:.4f} px")
    # print(f"opening_angle_height: {np.degrees(cam.opening_angle_height):.4f} deg")
    # print(f"optical_center:       {cam.optical_center_x}, {cam.optical_center_y}")

    # ------------------------------------------------------------------
    # Fetch images + camera matrices
    # ------------------------------------------------------------------
    frame_data    = {}
    image_obj_list = v_client.image.list(log=log_id, camera=camera)
    first_batch   = list(islice(image_obj_list, 10))

    Path("./test_images").mkdir(exist_ok=True)

    for img_obj in first_batch:
        img_url  = "https://logs.berlin-united.com/" + img_obj.image_url
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
        if len(cm_list_all) == 0:
            print(f"No camera matrix found for frame {frame_id}!")
            frame_data[frame_id] = {"image_path": file_path, "cam_pose_world": None}
            continue

        first = cm_list_all[0]
        #print(f"representation_data: {first.representation_data}")

        cam_pose_world = parse_camera_matrix(first.representation_data)
        T = cam_pose_world[:3, 3]
        frame_data[frame_id] = {"image_path": file_path, "cam_pose_world": cam_pose_world}

    # ------------------------------------------------------------------
    # Run adaptive ball detection on each frame
    # ------------------------------------------------------------------
    #classifier_green = ColorClassifier(55, 10, 40, np.radians(210), np.radians(25))
    classifier_green = ColorClassifier(50, 3, 40, np.radians(-132.0), np.radians(34.9))
    #55, 10, 40, np.radians(210), np.radians(25)
    VIS_DIR = Path("./visualizations")
    VIS_DIR.mkdir(exist_ok=True)

    for frame_id, data in frame_data.items():
        if data["cam_pose_world"] is None:
            print(f"Skipping frame {frame_id} — no camera matrix")
            continue

        img, img_y, img_u, img_v = load_image(data["image_path"])
        img_green = classifier_green.is_color(img_y, img_u, img_v)

        candidates, scanlines_h, scanlines_v, gap_segments = detect_ball_candidates_adaptive(
            np.array(img),
            img_green,
            cam_pose_world=data["cam_pose_world"],
            cam_info=CAMERA_INFO_TOP,
            ball_radius_mm=70.0,
            step_scale=0.5,       # one scanline per ball-radius height
            size_tolerance=0.25,   # accept patches within ±50% of expected radius
        )

        print(f"Frame {frame_id}: {len(candidates)} candidates after adaptive filtering")

        visualize_candidates(
            np.array(img), candidates,
            scanlines_h=scanlines_h,
            scanlines_v=scanlines_v,
            gap_segments=gap_segments,
            save_path=VIS_DIR / f"frame_{frame_id}.png",
        )