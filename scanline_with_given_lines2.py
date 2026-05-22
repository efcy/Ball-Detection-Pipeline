"""
Ball detection for NaoTH robot soccer — folder + API-camera-matrix mode.

Structure
---------
  CameraInfo          — camera intrinsics dataclass + factory
  CameraGeometry      — pixel↔world projection helpers (static methods)
  ColorClassifier     — YCbCr green / chroma classifier
  AnnotationParser    — parse ball / field-border / lines from JSON
  MaskBuilder         — rasterise polygon & line annotations → boolean masks
  BallDetector        — scanline-gap detector (field-border-first)
  Visualizer          — focused debug visualisation (only inside-field content)
  FrameProcessor      — orchestrates one frame end-to-end
  main                — CLI entry point
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OPENING_ANGLE_DIAGONAL_DEG = 72.6   # NaoTHSoccer/Config/platform/Nao6/CameraMatrixTop.ini
BALL_RADIUS_MM              = 70.0  # RoboCup standard (post-2026)
DEFAULT_IMAGE_SIZE          = (640, 480)


# ─────────────────────────────────────────────────────────────────────────────
# CameraInfo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraInfo:
    width:                int
    height:               int
    focal_length:         float
    optical_center_x:     float
    optical_center_y:     float
    opening_angle_height: float  # vertical FOV in radians

    @staticmethod
    def from_diagonal_fov(
        opening_angle_diagonal_deg: float,
        width: int = DEFAULT_IMAGE_SIZE[0],
        height: int = DEFAULT_IMAGE_SIZE[1],
    ) -> "CameraInfo":
        """
        Derives focal length and vertical FOV from the diagonal FOV angle.
        Config source: NaoTHSoccer/Config/platform/Nao6/CameraMatrixTop.ini
        C++: getFocalLength(), getOpeningAngleHeight(), getOpticalCenterX/Y() — CameraInfo.cpp
        """
        opening_angle_diagonal = np.radians(opening_angle_diagonal_deg)

        # getFocalLength(): half-diagonal in pixels / tan(half FOV angle)
        half_diag    = 0.5 * np.hypot(width, height)
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


# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry
# ─────────────────────────────────────────────────────────────────────────────

class CameraGeometry:
    """
    All methods are stateless; pass cam_pose_world (4×4 camera-to-world matrix)
    and a CameraInfo explicitly.
    """

    @staticmethod
    def parse_pose_matrix(representation_data: dict) -> np.ndarray:
        """
        Parses the API representation_data into a 4x4 cam-in-world pose matrix.
        C++: Serializer<CameraMatrix>

        M = [R | T] where:
          R  — camera axes expressed in robot/world frame (rows = x, y, z camera axes)
          T  — camera position in robot/world frame (mm)

        Usage:
          world_point = M @ cam_point     (cam → world)
          cam_point   = M_inv @ world_pt  (world → cam, use world_to_cam())
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

    @staticmethod
    def world_to_cam(cam_pose_world: np.ndarray) -> np.ndarray:
        """
        Inverts a cam-in-world pose into a world-to-camera transform.
        C++: Pose3D::invert() — rigid body transform inversion

        Since M is a rigid body transform (orthonormal R):
          R_inv = R^T
          T_inv = -R^T @ T
        """
        R = cam_pose_world[:3, :3]
        T = cam_pose_world[:3,  3]

        R_inv = R.T
        T_inv = -R.T @ T

        M_inv = np.eye(4)
        M_inv[:3, :3] = R_inv
        M_inv[:3,  3] = T_inv
        return M_inv

    @staticmethod
    def pixel_to_cam_ray(cam_info: CameraInfo, img_x: float, img_y: float) -> np.ndarray:
        """
        Converts a pixel (img_x, img_y) into a direction ray in camera space.
        C++: CameraGeometry::imagePixelToCameraCoords()

        Camera convention: X = forward (optical axis), Y = left, Z = up.
        focal_length is the X component — it sets the "depth" of the virtual image plane.
        optical_center offsets map pixel origin (top-left) to camera center.
        """
        x = cam_info.focal_length
        y = cam_info.optical_center_x - 0.5 - img_x
        z = cam_info.optical_center_y - 0.5 - img_y
        return np.array([x, y, z])

    @staticmethod
    def pixel_to_field(
        cam_pose_world: np.ndarray,
        cam_info: CameraInfo,
        img_x: float,
        img_y: float,
        object_height: float = 0.0,
    ) -> np.ndarray | None:
        """
        Projects a pixel back onto the horizontal field plane at a given height above ground.
        Returns (x, y) in robot/world coordinates, or None if the projection is impossible.
        C++: CameraGeometry::imagePixelToFieldCoord()  !CHECK!

        Impossible cases (mirrors the C++ epsilon guard):
          - Ray is horizontal (pixel_vec_world[2] ≈ 0) → never reaches target height
          - Ray points away from target height (signs differ) → looking above horizon
            when target is below camera, or vice versa
        """
        epsilon = 1e-13

        R = cam_pose_world[:3, :3]   # camera axes in world frame
        T = cam_pose_world[:3,  3]   # camera position in world (mm)

        # Build the ray direction: pixel → camera space → rotate into world space
        pixel_vec_cam   = CameraGeometry.pixel_to_cam_ray(cam_info, img_x, img_y)
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

    @staticmethod
    def expected_ball_radius_px(
        cam_pose_world: np.ndarray,
        cam_info: CameraInfo,
        ball_radius_mm: float,
        img_x: float,
        img_y: float,
    ) -> float:
        """
        Estimates the expected ball radius in pixels for a ball seen at pixel (img_x, img_y).
        Returns -1.0 if the projection is geometrically impossible.
        C++: CameraGeometry::estimateBallRadiusInPixels()  !CHECK!

        Steps:
          1. Project pixel to field plane at height = ball_radius_mm,
             gives horizontal ball position (x, y) on the field
          2. Reconstruct full 3D ball center: (field_x, field_y, ball_radius_mm)
             z = ball_radius because ball rests on ground and center is one radius above ground
          3. Transform ball center into camera space (air distance from lens to ball center)
          4. Angular diameter to pixel radius
        """
        # Step 1: project pixel onto field plane at ball center height
        point_on_field = CameraGeometry.pixel_to_field(
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
        ball_in_cam      = CameraGeometry.world_to_cam(cam_pose_world) @ ball_center_world
        cam_ball_distance = np.linalg.norm(ball_in_cam[:3])

        # Guard: camera must be outside the ball (distance > radius)
        # If inside, the tangent-line angular formula below breaks down
        if cam_ball_distance <= ball_radius_mm:
            return -1.0

        # Step 4: angular half-diameter → pixels
        alpha = np.arctan2(ball_radius_mm, cam_ball_distance)
        return alpha / cam_info.opening_angle_height * cam_info.height


# ─────────────────────────────────────────────────────────────────────────────
# Color classifier
# ─────────────────────────────────────────────────────────────────────────────

class ColorClassifier:
    """
    YCbCr-based colour classifier.  Construct with tuned parameters; call
    is_green() on numpy arrays of Y / Cb / Cr channel values.
    """

    def __init__(
        self,
        brightness_cone_radius_white: float,
        brightness_cone_radius_black: float,
        brightness_cone_offset:       float,
        color_angle_center:           float,
        color_angle_width:            float,
    ):
        self.bcr_white  = brightness_cone_radius_white
        self.bcr_black  = brightness_cone_radius_black
        self.bc_offset  = brightness_cone_offset
        self.ca_center  = color_angle_center
        self.ca_width   = color_angle_width

    # ── private helpers ───────────────────────────────────────────────────────

    def _is_achromatic(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        alpha     = (self.bcr_white - self.bcr_black) / (255.0 - self.bc_offset)
        threshold = np.clip(
            self.bcr_black + alpha * (y - self.bc_offset),
            self.bcr_black, 255,
        )
        return np.hypot(u - 128, v - 128) < threshold

    def _is_target_hue(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        angle = np.arctan2(v - 128, u - 128)
        diff  = np.arctan2(np.sin(angle - self.ca_center), np.cos(angle - self.ca_center))
        return np.abs(diff) < self.ca_width

    # ── public ───────────────────────────────────────────────────────────────

    def is_green(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return ~self._is_achromatic(y, u, v) & self._is_target_hue(u, v)

    @staticmethod
    def default_green() -> "ColorClassifier":
        return ColorClassifier(50, 3, 40, np.radians(-132.0), np.radians(34.9))


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parser
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationParser:
    """Parse Label Studio annotations (single item per call)."""

    @staticmethod
    def ball(annotation: dict) -> tuple[float, float, float, float] | None:
        if annotation.get("type") != "rectanglelabels":
            return None

        value  = annotation.get("value", {})
        labels = value.get("rectanglelabels", [])

        if not labels or labels[0].lower() != "ball":
            return None

        return (
            float(value["x"]),
            float(value["y"]),
            float(value["width"]),
            float(value["height"]),
        )

    @staticmethod
    def field_border(annotation: dict) -> list[tuple[float, float]] | None:
        if annotation.get("type") != "polygonlabels":
            return None

        value  = annotation.get("value", {})
        labels = value.get("polygonlabels", [])

        if not labels or labels[0] != "Field Border":
            return None

        pts = value.get("points", [])
        if len(pts) < 3:
            return None

        return [(float(x), float(y)) for x, y in pts]

    @staticmethod
    def lines(annotation: dict) -> list[list[tuple[float, float]]]:
        if annotation.get("type") != "polygonlabels":
            return []

        value  = annotation.get("value", {})
        labels = value.get("polygonlabels", [])

        if not labels or labels[0] != "Line":
            return []

        pts = value.get("points", [])
        if len(pts) < 2:
            return []

        return [[(float(x), float(y)) for x, y in pts]]


# ─────────────────────────────────────────────────────────────────────────────
# Mask builder
# ─────────────────────────────────────────────────────────────────────────────

class MaskBuilder:
    """Convert annotation geometry to boolean pixel masks."""

    @staticmethod
    def field_border(shape: tuple, polygon: list[tuple[float, float]]) -> np.ndarray:
        """True inside the field-border polygon."""
        h, w   = shape[:2]
        canvas = Image.new("L", (w, h), 0)

        polygon_px = [
            (x / 100.0 * w, y / 100.0 * h)
            for x, y in polygon
        ]

        print("POLYGON PX SAMPLE:", polygon_px[:5])

        ImageDraw.Draw(canvas).polygon(polygon_px, fill=255)
        return np.array(canvas) > 0

    @staticmethod
    def lines(
        shape: tuple,
        lines: list[list[tuple[float, float]]],
        thickness_px: int = 8,
    ) -> np.ndarray:
        """True where annotated field lines are (rasterised with given thickness)."""
        h, w   = shape[:2]
        canvas = Image.new("L", (w, h), 0)
        draw   = ImageDraw.Draw(canvas)

        for line in lines:
            line_px = [
                (x / 100.0 * w, y / 100.0 * h)
                for x, y in line
            ]
            if len(line_px) >= 3:
                draw.polygon(line_px, fill=255)

        return np.array(canvas) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ball detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    candidates:   list[tuple[int, int, int, int]]   # (x, y, w, h) in image coords
    scanlines_h:  list[int]                          # y values of horizontal scanlines
    scanlines_v:  list[int]                          # x values of vertical scanlines
    gap_segments: list[dict]                         # raw gap segments (for debug)

    green_mask:        np.ndarray | None = None
    valid_field_mask:  np.ndarray | None = None
    field_border_mask: np.ndarray | None = None
    line_mask:         np.ndarray | None = None

    roi:            tuple[int, int, int, int] | None = None
    raw_candidates: list[tuple[int, int, int, int]] | None = None
    row_radius:     dict[int, float] | None = None


class BallDetector:
    """
    Geometry-aware scanline-gap ball detector.

    Detection pipeline
    ------------------
    1. Build a *valid_field_mask* mask: green pixels UNION line pixels (inside field).
       Lines are treated as passable so they don't generate spurious gaps.
    2. Restrict all processing to the bounding box of the field-border polygon.
    3. Compute per-row expected ball radius using camera geometry.
    4. Space scanlines adaptively — closer rows need fewer, more-spread lines.
    5. Find non-green gaps of the expected size on each scanline.
    6. Cluster overlapping gaps into bounding boxes.
    7. Filter by expected radius at the candidate centre; split wide boxes.
    """

    def __init__(
        self,
        cam_pose_world: np.ndarray,
        cam_info:       CameraInfo,
        ball_radius_mm: float = BALL_RADIUS_MM,
        step_scale:     float = 0.5,
        size_tolerance: float = 0.76,
    ):
        self.cam_pose = cam_pose_world
        self.cam_info = cam_info
        self.ball_r   = ball_radius_mm
        self.step     = step_scale
        self.tol      = size_tolerance

    def detect(
        self,
        green_mask:        np.ndarray,
        field_border_mask: np.ndarray | None,
        line_mask:         np.ndarray | None,
    ) -> DetectionResult:
        h, w = green_mask.shape[:2]

        # Step 1 — valid_field_mask: green OR (line AND inside field)
        valid_field_mask = green_mask.copy()
        if field_border_mask is not None:
            valid_field_mask &= field_border_mask  # ignore green outside field boundary
        if line_mask is not None:
            inside_lines      = line_mask & (field_border_mask if field_border_mask is not None else True)
            valid_field_mask |= inside_lines       # treat lines as passable (like green)

        # Step 2 — ROI from field border bounding box
        roi        = self._roi_from_mask(field_border_mask, w, h)
        x0, x1, y0, y1 = roi

        # Step 3 — expected radius per row (in full-image coords)
        cx_full    = x0 + (x1 - x0) // 2
        row_radius = {
            y: CameraGeometry.expected_ball_radius_px(
                self.cam_pose, self.cam_info, self.ball_r, cx_full, y
            )
            for y in range(y0, y1)
        }

        # Step 4 — adaptive scanlines (full-image coords, inside ROI)
        field_top_y = self._first_green_row(valid_field_mask[y0:y1, x0:x1], y0)
        
        scanlines_h = self._hlines(valid_field_mask, field_border_mask, row_radius, field_top_y, y0, y1, x0, x1)
        scanlines_v = self._vlines(valid_field_mask, field_border_mask, row_radius, field_top_y, y0, y1, x0, x1)

        # Step 5 — gap scan + clustering
        radius_values = [v for v in row_radius.values() if v > 0]
        r_far   = max(row_radius.get(scanlines_h[0]  if scanlines_h else y0,  5.0), 3.0)
        r_near  = max(row_radius.get(scanlines_h[-1] if scanlines_h else y1, 60.0), r_far + 5.0)
        r_mid   = float(np.median(radius_values)) if radius_values else 15.0
        min_gap = max(3,  int(r_far  * (1 - self.tol) * 2))
        max_gap = max(80, int(r_near * (1 + self.tol) * 2))
        cluster_proximity = max(5, int(r_mid * 0.6))

        raw_candidates, gap_segments = self._scan_and_cluster(
            valid_field_mask, scanlines_h, scanlines_v,
            min_gap, max_gap, field_top_y, cluster_proximity,
        )

        # Step 6 — filter by expected radius; optionally split wide boxes
        candidates = self._filter(raw_candidates, field_border_mask, row_radius)

        return DetectionResult(
            candidates=candidates,
            scanlines_h=scanlines_h,
            scanlines_v=scanlines_v,
            gap_segments=gap_segments,
            green_mask=green_mask,
            valid_field_mask=valid_field_mask,
            field_border_mask=field_border_mask,
            line_mask=line_mask,
            roi=roi,
            raw_candidates=raw_candidates,
            row_radius=row_radius,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _roi_from_mask(mask: np.ndarray | None, w: int, h: int) -> tuple[int, int, int, int]:
        if mask is None:
            return 0, w, 0, h
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0, w, 0, h
        return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1

    @staticmethod
    def _first_green_row(green_crop: np.ndarray, y0: int, min_frac: float = 0.05) -> int:
        _, w = green_crop.shape[:2]
        for i, row in enumerate(green_crop):
            if row.sum() / w >= min_frac:
                return y0 + i
        return y0

    def _hlines(self, valid_field_mask, fb_mask, row_radius, field_top, y0, y1, x0, x1):
        lines = []
        y = field_top
        last_valid_r = 10.0
        while y < y1:
            row_slice = fb_mask[y, x0:x1] if fb_mask is not None else None
            if row_slice is None or row_slice.mean() > 0.05:
                lines.append(y)
            r    = row_radius.get(y, -1.0)
            
            if r <= 0:
                r = last_valid_r
            else:
                last_valid_r = r

            step = max(2, int(self.step * 2 * r))
            y += step
        return lines

    def _vlines(self, valid_field_mask, fb_mask, row_radius, field_top, y0, y1, x0, x1):
        lines           = []
        x               = x0
        representative_y = field_top + (y1 - field_top) // 2
        last_valid_r = 5
        while x < x1:
            col_slice = fb_mask[y0:y1, x] if fb_mask is not None else None
            if col_slice is None or col_slice.mean() > 0.05:
                lines.append(x)
            r    = CameraGeometry.expected_ball_radius_px(
                self.cam_pose, self.cam_info, self.ball_r, x, representative_y
            )
            
            if r <= 0:
                r = last_valid_r
                print(f"Warning: invalid radius at x={x}, using last valid r={r:.2f}")
            else:
                last_valid_r = r
            step = max(2, int(self.step * 2 * r))
            x   += step
        return lines

    def _scan_and_cluster(
        self, valid_field_mask, scanlines_h, scanlines_v,
        min_gap, max_gap, field_top, proximity,
    ) -> tuple[list, list]:
        h, w     = valid_field_mask.shape[:2]
        segments = []
        gap_segs = []

        # Horizontal gaps
        for y in scanlines_h:
            if y >= h:
                continue
            in_gap, start_x = False, 0
            for x in range(w):
                green = valid_field_mask[y, x]
                if not green and not in_gap:
                    start_x, in_gap = x, True
                elif green and in_gap:
                    gw = x - start_x
                    if min_gap <= gw <= max_gap:
                        segments.append({"y": y, "x1": start_x, "x2": x, "type": "horizontal"})
                        gap_segs.append({"x1": start_x, "x2": x, "y1": y, "y2": y, "type": "horizontal"})
                    in_gap = False

        # Vertical gaps
        for x in scanlines_v:
            if x >= w:
                continue
            in_gap, start_y = False, 0
            for y in range(field_top, h):
                green = valid_field_mask[y, x]
                if not green and not in_gap:
                    start_y, in_gap = y, True
                elif green and in_gap:
                    gh = y - start_y
                    if min_gap <= gh <= max_gap:
                        segments.append({"x": x, "y1": start_y, "y2": y, "type": "vertical"})
                        gap_segs.append({"x1": x, "x2": x, "y1": start_y, "y2": y, "type": "vertical"})
                    in_gap = False

        return self._cluster(segments, proximity), gap_segs

    @staticmethod
    def _cluster(segments: list[dict], proximity: int) -> list[tuple]:
        clusters: list[dict] = []

        for seg in segments:
            matched = False
            for c in clusters:
                matched = BallDetector._merge_into(c, seg, proximity)
                if matched:
                    break
            if not matched:
                if seg["type"] == "horizontal":
                    clusters.append({
                        "x1": seg["x1"], "x2": seg["x2"],
                        "y1": seg["y"],  "y2": seg["y"],
                        "type": "horizontal",
                    })
                else:
                    clusters.append({
                        "x1": seg["x"],  "x2": seg["x"],
                        "y1": seg["y1"], "y2": seg["y2"],
                        "type": "vertical",
                    })

        bboxes = []
        for c in clusters:
            x1, x2, y1, y2 = c["x1"], c["x2"], c["y1"], c["y2"]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and max(w, h) / min(w, h) < 3:
                bboxes.append((x1, y1, w, h))
        return bboxes

    @staticmethod
    def _merge_into(cluster: dict, seg: dict, prox: int) -> bool:
        """Try to merge *seg* into *cluster*. Returns True on success."""
        ct, st = cluster["type"], seg["type"]

        if ct == "horizontal" and st == "horizontal":
            if (abs(seg["y"] - cluster["y2"]) <= prox
                    and not (seg["x2"] < cluster["x1"] or seg["x1"] > cluster["x2"])):
                cluster["x1"] = min(cluster["x1"], seg["x1"])
                cluster["x2"] = max(cluster["x2"], seg["x2"])
                cluster["y2"] = max(cluster["y2"], seg["y"])
                return True

        elif ct == "vertical" and st == "vertical":
            if (abs(seg["x"] - cluster["x2"]) <= prox
                    and not (seg["y2"] < cluster["y1"] or seg["y1"] > cluster["y2"])):
                cluster["y1"] = min(cluster["y1"], seg["y1"])
                cluster["y2"] = max(cluster["y2"], seg["y2"])
                cluster["x2"] = max(cluster["x2"], seg["x"])
                return True

        else:
            # Cross-type: one horizontal, one vertical
            vx  = seg.get("x",  seg.get("x1", 0))  if st == "vertical"   else cluster["x1"]
            vy1 = seg.get("y1", 0)                  if st == "vertical"   else cluster["y1"]
            vy2 = seg.get("y2", vy1)                if st == "vertical"   else cluster["y2"]
            hx1 = seg["x1"]  if st == "horizontal" else cluster["x1"]
            hx2 = seg["x2"]  if st == "horizontal" else cluster["x2"]
            hy1 = seg["y"]   if st == "horizontal" else cluster["y1"]
            hy2 = seg["y"]   if st == "horizontal" else cluster["y2"]

            if ((hx1 - prox) <= vx <= (hx2 + prox)
                    and (vy1 - prox) <= hy2
                    and (vy2 + prox) >= hy1):
                cluster["x1"]  = min(cluster["x1"], hx1, vx)
                cluster["x2"]  = max(cluster["x2"], hx2, vx)
                cluster["y1"]  = min(cluster["y1"], hy1, vy1)
                cluster["y2"]  = max(cluster["y2"], hy2, vy2)
                cluster["type"] = "merged"
                return True

        return False

    def _filter(
        self,
        raw: list[tuple],
        field_border_mask: np.ndarray | None,
        row_radius: dict[int, float],
    ) -> list[tuple]:
        filtered = []
        for (x, y, w, h) in raw:
            cx, cy = x + w // 2, y + h // 2

            # Must be inside field border
            if field_border_mask is not None:
                py = int(np.clip(cy, 0, field_border_mask.shape[0] - 1))
                px = int(np.clip(cx, 0, field_border_mask.shape[1] - 1))
                if not field_border_mask[py, px]:
                    continue

            r_exp = CameraGeometry.expected_ball_radius_px(
                self.cam_pose, self.cam_info, self.ball_r, cx, cy
            )
            if r_exp <= 0:
                continue

            lo, hi = r_exp * (1 - self.tol), r_exp * (1 + self.tol)

            # Split wide boxes into halves; test each half separately
            aspect = w / h if h > 0 else 999
            halves = (
                [(x, y, w // 2, h), (x + w // 2, y, w // 2, h)] if aspect > 1.8
                else [(x, y, w, h)]
            )
            for hx, hy, hw, hh in halves:
                r_patch = (hw + hh) / 4.0
                if lo <= r_patch <= hi:
                    filtered.append((hx, hy, hw, hh))
        return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Match checker
# ─────────────────────────────────────────────────────────────────────────────

def check_annotation_match(
    candidates:      list[tuple],
    ball_annotation: tuple[float, float] | None,
    iou_threshold:   float = 0.1,
) -> dict:
    """
    Returns {matched, matching_idx, best_iou, ball_in_any}.
    A candidate "matches" if the annotated ball centre is inside its bbox
    OR if the IoU (using an estimated square annotation box) exceeds threshold.
    """
    result = dict(matched=False, matching_idx=None, best_iou=0.0, ball_in_any=False)
    if ball_annotation is None or not candidates:
        return result

    x, y, w, h = ball_annotation
    bx      = x + w / 2
    by      = y + h / 2
    best_iou = 0.0
    best_idx = None

    for i, (cx, cy, cw, ch) in enumerate(candidates):
        if cx <= bx <= cx + cw and cy <= by <= cy + ch:
            result["ball_in_any"] = True

        r_est = (cw + ch) / 4.0
        ax, ay, aw, ah = bx - r_est, by - r_est, r_est * 2, r_est * 2

        ix1 = max(cx, ax);  iy1 = max(cy, ay)
        ix2 = min(cx + cw, ax + aw);  iy2 = min(cy + ch, ay + ah)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = cw * ch + aw * ah - inter
        iou   = inter / union if union > 0 else 0.0

        if iou > best_iou:
            best_iou, best_idx = iou, i

    result.update(best_iou=best_iou, matching_idx=best_idx,
                  matched=best_iou >= iou_threshold or result["ball_in_any"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

def debug_single(image, mask):
    h, w = image.shape[:2]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mask.astype(np.uint8), cmap="gray", extent=[0, w, h, 0])
    ax[0].set_xlim(0, w)
    ax[0].set_ylim(h, 0)
    print("xlim:", ax[0].get_xlim(), "ylim:", ax[0].get_ylim())
    plt.show()


def plot_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    print("shape:", mask.shape)
    print("dtype:", mask.dtype)
    print("min/max:", mask.min(), mask.max())
    plt.figure(figsize=(6, 5))
    plt.imshow(mask, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


class Visualizer:

    @staticmethod
    def save(
        image:            np.ndarray,
        result:           DetectionResult,
        field_border=None,
        annotated_lines=None,
        ball_annotation=None,
        match_result=None,
        save_path=None,
    ):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        # 1. RGB image
        ax = axes[0]
        ax.imshow(image, aspect="auto", origin="upper")
        ax.set_title("1. RGB Input")

        # 2. Green mask
        ax = axes[1]
        ax.imshow(result.green_mask.astype(np.uint8) * 255,
                  cmap="gray", origin="upper")
        ax.set_aspect("auto")
        ax.set_title("2. Green Mask")

        # 3. Valid field mask
        ax = axes[2]
        h, w = result.valid_field_mask.shape
        ax.imshow(result.valid_field_mask, cmap="gray",
                  extent=[0, w, h, 0], aspect="auto", origin="upper")
        ax.set_title("3. Valid Field Mask")

        # 4. ROI
        ax = axes[3]
        ax.imshow(image, aspect="auto", origin="upper")
        if result.roi is not None:
            x0, x1, y0, y1 = result.roi
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                edgecolor="red", facecolor="none", linewidth=2,
            ))
        ax.set_title("4. ROI")

        # 5. Scanlines
        ax = axes[4]
        ax.imshow(image, aspect="auto", origin="upper")
        for y in result.scanlines_h:
            ax.axhline(y, color="cyan", alpha=0.5)
        for x in result.scanlines_v:
            ax.axvline(x, color="lime", alpha=0.5)
        ax.set_title("5. Adaptive Scanlines")

        # 6. Gap segments
        ax = axes[5]
        ax.imshow(image, aspect="auto", origin="upper")
        for seg in result.gap_segments:
            if seg["type"] == "horizontal":
                ax.plot([seg["x1"], seg["x2"]], [seg["y1"], seg["y1"]],
                        color="orange", linewidth=3)
            else:
                ax.plot([seg["x1"], seg["x1"]], [seg["y1"], seg["y2"]],
                        color="yellow", linewidth=3)
        ax.set_title("6. Gap Segments")

        # 7. Raw clustered candidates
        ax = axes[6]
        ax.imshow(image, aspect="auto", origin="upper")
        if result.raw_candidates is not None:
            for x, y, w, h in result.raw_candidates:
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h,
                    edgecolor="magenta", facecolor="none", linewidth=2,
                ))
        ax.set_title("7. Raw Clusters")

        # 8. Final filtered candidates
        ax = axes[7]
        ax.imshow(image, aspect="auto", origin="upper")
        for i, (x, y, w, h) in enumerate(result.candidates):
            is_match  = match_result is not None and match_result.get("matching_idx") == i
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                edgecolor="lime" if is_match else "red",
                facecolor="none", linewidth=3,
            ))
        if ball_annotation is not None:
            x, y, w, h = ball_annotation
            ax.plot(x + w / 2, y + h / 2, "b+", markersize=15)
        ax.set_title("8. Final Detection")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image_ycbcr(path: Path):
    img   = Image.open(path)
    print("RAW PIL size:", img.size)

    ycbcr = img.convert("YCbCr")
    arr   = np.array(ycbcr)
    print("YCbCr array:", arr.shape)

    rgb = np.array(img.convert("RGB"))
    print("RGB array:", rgb.shape)

    return rgb, arr[..., 0], arr[..., 1], arr[..., 2]


def load_annotation(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def scan_folder(folder: Path, image_ext: str = ".jpg") -> list[dict]:
    """
    Find image + JSON pairs in *folder*.
    Expects filenames like  <log_id>_<frame_id><ext>.
    """
    entries = []
    for img_path in sorted(folder.glob(f"*{image_ext}")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  [skip] no JSON for {img_path.name}")
            continue
        parts = img_path.stem.split("_", 1)
        entries.append(dict(
            stem       = img_path.stem,
            image_path = img_path,
            json_path  = json_path,
            log_id     = parts[0] if parts else "?",
            frame_id   = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
        ))
    print(entries)
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Frame processor
# ─────────────────────────────────────────────────────────────────────────────

class FrameProcessor:
    def __init__(
        self,
        cam_info:   CameraInfo,
        classifier: ColorClassifier,
        v_client,
        camera:     str  = "TOP",
        vis_dir:    Path = Path("./visualizations"),
        step_scale: float = 0.5,
        size_tol:   float = 0.76,
    ):
        self.cam_info   = cam_info
        self.classifier = classifier
        self.v_client   = v_client
        self.camera     = camera
        self.vis_dir    = vis_dir
        self.step_scale = step_scale
        self.size_tol   = size_tol

    def _fetch_camera_matrix(self, frame_id: int) -> np.ndarray | None:
        if self.camera == "TOP":
            cm_iter = self.v_client.cameramatrixtop.list(frame=frame_id)
        else:
            cm_iter = self.v_client.cameramatrix.list(frame=frame_id)
        items = list(islice(cm_iter, 1))
        if not items:
            return None
        return CameraGeometry.parse_pose_matrix(items[0].representation_data)

    def process(self, entry: dict) -> bool:
        stem     = entry["stem"]
        frame_id = entry["frame_id"]
        print(f"\n[{stem}]")

        # Load annotation
        try:
            annotation_data = load_annotation(entry["json_path"])
            annotations     = annotation_data
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [skip] annotation error: {e}")
            return False

        # Fetch camera matrix from API
        cam_pose = self._fetch_camera_matrix(frame_id)
        if cam_pose is None:
            print(f"  [skip] no camera matrix for frame {frame_id}")
            return False

        # Load image + compute green mask
        img_rgb, img_y, img_cb, img_cr = load_image_ycbcr(entry["image_path"])
        green_mask = self.classifier.is_green(img_y, img_cb, img_cr)

        # If Label Studio format
        if isinstance(annotations, dict) and "annotations" in annotations:
            annotations = annotations["annotations"][0]["result"]

        ball_ann     = None
        field_border = None
        lines        = []

        for ann in annotations:
            b = AnnotationParser.ball(ann)
            if b:
                ball_ann = b
                print(f"  Found ball annotation at ({ball_ann[0]:.1f}, {ball_ann[1]:.1f})")

            fb = AnnotationParser.field_border(ann)
            if fb:
                field_border = fb
                print(f"  Found field border annotation with {len(field_border)} points")

            ls = AnnotationParser.lines(ann)
            if ls:
                lines.extend(ls)
                print(f"  Found {len(ls)} annotated line(s) with total "
                      f"{sum(len(l) for l in ls)} points")

        # Build pixel masks
        shape     = img_rgb.shape
        fb_mask   = MaskBuilder.field_border(shape, field_border) if field_border else None
        line_mask = MaskBuilder.lines(shape, lines)               if lines        else None

        if fb_mask is None:
            print("  [warn] no field border annotation — scanning full image")

        # Detect
        detector = BallDetector(
            cam_pose, self.cam_info, BALL_RADIUS_MM,
            self.step_scale, self.size_tol,
        )
        result = detector.detect(green_mask, fb_mask, line_mask)
        print(f"  {len(result.candidates)} candidate(s) after filtering")

        # Check annotation match
        match = check_annotation_match(result.candidates, ball_ann)
        if ball_ann is not None:
            status = "MATCHED" if match["matched"] else "NOT matched"
            print(f"  Ball @ ({ball_ann[0]:.0f},{ball_ann[1]:.0f}): "
                  f"{status}  IoU={match['best_iou']:.2f}  "
                  f"centre_inside={match['ball_in_any']}")
        else:
            print("  No ball annotated")

        # Visualise
        Visualizer.save(
            img_rgb,
            result,
            field_border=field_border,
            annotated_lines=lines,
            ball_annotation=ball_ann,
            match_result=match,
            save_path=self.vis_dir / f"{stem}.png",
        )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive ball detection — folder mode with API camera matrix"
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Folder containing images and JSON annotations")
    parser.add_argument("--output", type=Path, default=Path("./visualizations"),
                        help="Output folder for PNG visualisations")
    parser.add_argument("--ext",    type=str,  default=".jpg",
                        help="Image file extension (default: .jpg)")
    parser.add_argument("--camera", type=str,  default="TOP",
                        choices=["TOP", "BOTTOM"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    from vaapi.client import Vaapi

    v_client = Vaapi(
        base_url=os.environ["VAT_API_URL"],
        api_key=os.environ["VAT_API_TOKEN"],
    )

    cam_info   = CameraInfo.from_diagonal_fov(OPENING_ANGLE_DIAGONAL_DEG)
    classifier = ColorClassifier.default_green()
    processor  = FrameProcessor(
        cam_info=cam_info,
        classifier=classifier,
        v_client=v_client,
        camera=args.camera,
        vis_dir=args.output,
    )

    entries = scan_folder(args.input, image_ext=args.ext)
    print(f"Found {len(entries)} image+json pairs in {args.input}\n")

    ok = sum(processor.process(e) for e in entries)
    print(f"\nDone — {ok}/{len(entries)} frames processed successfully.")


if __name__ == "__main__":
    main()