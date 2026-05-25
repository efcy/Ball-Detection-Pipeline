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
    opening_angle_height: float

    @staticmethod
    def from_diagonal_fov(
        opening_angle_diagonal_deg: float,
        width: int = DEFAULT_IMAGE_SIZE[0],
        height: int = DEFAULT_IMAGE_SIZE[1],
    ) -> "CameraInfo":
        diag_rad     = np.radians(opening_angle_diagonal_deg)
        half_diag    = 0.5 * np.hypot(width, height)
        focal_length = half_diag / np.tan(0.5 * diag_rad)
        return CameraInfo(
            width=width,
            height=height,
            focal_length=focal_length,
            optical_center_x=float(width  // 2),
            optical_center_y=float(height // 2),
            opening_angle_height=2.0 * np.arctan2(float(height), focal_length * 2.0),
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
        """Build a 4×4 camera-to-world matrix from an API representation dict."""
        pose = representation_data["pose"]
        rot  = pose["rotation"]
        t    = pose["translation"]
        R    = np.array([
            [rot[0]["x"], rot[0]["y"], rot[0]["z"]],
            [rot[1]["x"], rot[1]["y"], rot[1]["z"]],
            [rot[2]["x"], rot[2]["y"], rot[2]["z"]],
        ])
        T = np.array([t["x"], t["y"], t["z"]])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3,  3] = T
        return M

    @staticmethod
    def world_to_cam(cam_pose_world: np.ndarray) -> np.ndarray:
        R     = cam_pose_world[:3, :3]
        T     = cam_pose_world[:3,  3]
        M_inv = np.eye(4)
        M_inv[:3, :3] = R.T
        M_inv[:3,  3] = -R.T @ T
        return M_inv

    @staticmethod
    def pixel_to_cam_ray(cam_info: CameraInfo, img_x: float, img_y: float) -> np.ndarray:
        return np.array([
            cam_info.focal_length,
            cam_info.optical_center_x - 0.5 - img_x,
            cam_info.optical_center_y - 0.5 - img_y,
        ])

    @staticmethod
    def pixel_to_field(
        cam_pose_world: np.ndarray,
        cam_info: CameraInfo,
        img_x: float,
        img_y: float,
        object_height: float = 0.0,
    ) -> np.ndarray | None:
        """
        Project image pixel onto the horizontal field plane at *object_height* (mm).
        Returns (field_x, field_y) or None if the ray points away from the plane.
        """
        R, T   = cam_pose_world[:3, :3], cam_pose_world[:3, 3]
        ray_w  = R @ CameraGeometry.pixel_to_cam_ray(cam_info, img_x, img_y)
        dz     = object_height - T[2]
        if ray_w[2] * dz < 1e-13:
            return None
        t = dz / ray_w[2]
        return np.array([T[0] + t * ray_w[0], T[1] + t * ray_w[1]])

    @staticmethod
    def expected_ball_radius_px(
        cam_pose_world: np.ndarray,
        cam_info: CameraInfo,
        ball_radius_mm: float,
        img_x: float,
        img_y: float,
    ) -> float:
        """
        Estimate how many pixels the ball should occupy at the given image position.
        Returns -1 if the projection is geometrically invalid.
        """
        point = CameraGeometry.pixel_to_field(
            cam_pose_world, cam_info, img_x, img_y, ball_radius_mm
        )
        if point is None:
            return -1.0
        ball_w = np.array([point[0], point[1], ball_radius_mm, 1.0])
        ball_c = CameraGeometry.world_to_cam(cam_pose_world) @ ball_w
        dist   = np.linalg.norm(ball_c[:3])
        if dist <= ball_radius_mm:
            return -1.0
        alpha = np.arctan2(ball_radius_mm, dist)
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

        value = annotation.get("value", {})
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

        value = annotation.get("value", {})
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

        value = annotation.get("value", {})
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
        h, w  = shape[:2]
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).polygon(polygon, fill=255)
        return np.array(canvas) > 0

    @staticmethod
    def lines(
        shape: tuple,
        lines: list[list[tuple[float, float]]],
        thickness_px: int = 8,
    ) -> np.ndarray:
        """True where annotated field lines are (rasterised with given thickness)."""
        h, w  = shape[:2]
        canvas = Image.new("L", (w, h), 0)
        draw   = ImageDraw.Draw(canvas)
        for line in lines:
            for i in range(len(line) - 1):
                draw.line([line[i], line[i + 1]], fill=255, width=thickness_px)
        return np.array(canvas) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ball detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    candidates:  list[tuple[int, int, int, int]]   # (x, y, w, h) in image coords
    scanlines_h: list[int]                          # y values of horizontal scanlines
    scanlines_v: list[int]                          # x values of vertical scanlines
    gap_segments: list[dict]                        # raw gap segments (for debug)


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
        self.cam_pose  = cam_pose_world
        self.cam_info  = cam_info
        self.ball_r    = ball_radius_mm
        self.step      = step_scale
        self.tol       = size_tolerance

    def detect(
        self,
        green_mask:          np.ndarray,
        field_border_mask:   np.ndarray | None,
        line_mask:           np.ndarray | None,
    ) -> DetectionResult:
        h, w = green_mask.shape[:2]

        # Step 1 — valid_field_mask mask: green OR (line AND inside field)
        valid_field_mask = green_mask.copy()
        if field_border_mask is not None:
            valid_field_mask &= field_border_mask # if something looks green, but outside the annotated field boundary - ignore it
        if line_mask is not None:
            inside_lines = line_mask & (field_border_mask if field_border_mask is not None else True) # treat lines as valid_field_mask, i.e. like green
            valid_field_mask  |= inside_lines             

        # Step 2 — ROI from field border bounding box
        roi = self._roi_from_mask(field_border_mask, w, h)
        x0, x1, y0, y1 = roi

        # Step 3 — expected radius per row (in full-image coords)
        cx_full         = x0 + (x1 - x0) // 2
        row_radius      = {
            y: CameraGeometry.expected_ball_radius_px(
                self.cam_pose, self.cam_info, self.ball_r, cx_full, y
            )
            for y in range(y0, y1)
        }

        # Step 4 — adaptive scanlines (full-image coords, inside ROI)
        field_top_y = self._first_green_row(valid_field_mask[y0:y1, x0:x1], y0)
        scanlines_h = self._hlines(valid_field_mask, field_border_mask, row_radius, field_top_y, y0, y1, x0, x1)
        scanlines_v = self._vlines(valid_field_mask, field_border_mask, row_radius, field_top_y, y0, y1, x0, x1)

        # Step 5 gap scan + clustering (working in full-image coords)
        radius_values = [v for v in row_radius.values() if v > 0]
        r_far   = max(row_radius.get(scanlines_h[0]  if scanlines_h  else y0, 5.0), 3.0)
        r_near  = max(row_radius.get(scanlines_h[-1] if scanlines_h  else y1, 60.0), r_far + 5.0)
        r_mid   = float(np.median(radius_values)) if radius_values else 15.0
        min_gap = max(3,  int(r_far  * (1 - self.tol) * 2))
        max_gap = max(80, int(r_near * (1 + self.tol) * 2))
        cluster_proximity = max(5, int(r_mid * 0.6))

        raw_candidates, gap_segments = self._scan_and_cluster(
            valid_field_mask, scanlines_h, scanlines_v, min_gap, max_gap,
            field_top_y, cluster_proximity,
        )

        # Step 6 — filter by expected radius; optionally split wide boxes
        candidates = self._filter(raw_candidates, field_border_mask, row_radius)

        return DetectionResult(candidates, scanlines_h, scanlines_v, gap_segments)

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
        while y < y1:
            # Only add scanline if this row has meaningful field coverage
            row_slice = fb_mask[y, x0:x1] if fb_mask is not None else None
            if row_slice is None or row_slice.mean() > 0.05:
                lines.append(y)
            r    = row_radius.get(y, -1.0)
            step = max(2, int(self.step * 2 * r)) if r > 0 else 10
            y   += step
        return lines

    def _vlines(self, valid_field_mask, fb_mask, row_radius, field_top, y0, y1, x0, x1):
        lines = []
        x = x0
        representative_y = field_top + (y1 - field_top) // 2
        while x < x1:
            col_slice = fb_mask[y0:y1, x] if fb_mask is not None else None
            if col_slice is None or col_slice.mean() > 0.05:
                lines.append(x)
            r    = CameraGeometry.expected_ball_radius_px(
                self.cam_pose, self.cam_info, self.ball_r, x, representative_y
            )
            step = max(2, int(self.step * 2 * r)) if r > 0 else 10
            x   += step
        return lines

    def _scan_and_cluster(
        self, valid_field_mask, scanlines_h, scanlines_v,
        min_gap, max_gap, field_top, proximity,
    ) -> tuple[list, list]:
        h, w       = valid_field_mask.shape[:2]
        segments   = []
        gap_segs   = []

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
            hx1 = seg["x1"]    if st == "horizontal" else cluster["x1"]
            hx2 = seg["x2"]    if st == "horizontal" else cluster["x2"]
            hy1 = seg["y"]     if st == "horizontal" else cluster["y1"]
            hy2 = seg["y"]     if st == "horizontal" else cluster["y2"]

            if ((hx1 - prox) <= vx <= (hx2 + prox)
                    and (vy1 - prox) <= hy2
                    and (vy2 + prox) >= hy1):
                cluster["x1"] = min(cluster["x1"], hx1, vx)
                cluster["x2"] = max(cluster["x2"], hx2, vx)
                cluster["y1"] = min(cluster["y1"], hy1, vy1)
                cluster["y2"] = max(cluster["y2"], hy2, vy2)
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
    candidates:    list[tuple],
    ball_annotation: tuple[float, float] | None,
    iou_threshold: float = 0.1,
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
    bx = x + w / 2
    by = y + h / 2
    best_iou    = 0.0
    best_idx    = None

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

# class Visualizer:
#     """
#     Debug visualisation.

#     Only content that lies *inside* the field border is drawn in detail
#     (scanlines, gap segments, candidates).  The full image is shown as
#     context but everything outside the field is dimmed.
#     """

#     @staticmethod
#     def save(
#         image:            np.ndarray,
#         result:           DetectionResult,
#         field_border:     list[tuple] | None = None,
#         field_border_mask: np.ndarray | None = None,
#         annotated_lines:  list | None = None,
#         ball_annotation:  tuple | None = None,
#         match_result:     dict | None = None,
#         save_path:        Path | None = None,
#     ) -> None:
#         fig, ax = plt.subplots(figsize=(14, 9))

#         # Dim everything outside the field so the analyst can focus inside
#         display_image = image.copy()
#         if field_border_mask is not None:
#             dim_factor = 0.35
#             outside    = ~field_border_mask
#             display_image[outside] = (display_image[outside] * dim_factor).astype(np.uint8)
#         ax.imshow(display_image)

#         h, w = image.shape[:2]

#         # ── field border outline ──────────────────────────────────────────────
#         if field_border and len(field_border) >= 3:
#             poly = plt.Polygon(
#                 field_border, closed=True,
#                 linewidth=2, edgecolor="deepskyblue", facecolor="none", alpha=0.9,
#             )
#             ax.add_patch(poly)

#         # ── annotated field lines (white overlay) ─────────────────────────────
#         if annotated_lines:
#             for line in annotated_lines:
#                 xs = [p[0] for p in line]
#                 ys = [p[1] for p in line]
#                 ax.plot(xs, ys, color="white", linewidth=1.5, alpha=0.7)

#         # ── scanlines — only drawn where they pass through the field ─────────
#         for y in result.scanlines_h:
#             if field_border_mask is not None:
#                 # Draw only the in-field segments of this scanline
#                 row = field_border_mask[y]
#                 segs = Visualizer._mask_to_segments(row)
#                 for s0, s1 in segs:
#                     ax.plot([s0, s1], [y, y], color="cyan", lw=0.9, alpha=0.55, ls="--")
#             else:
#                 ax.plot([0, w], [y, y], color="cyan", lw=0.9, alpha=0.55, ls="--")

#         for x in result.scanlines_v:
#             if field_border_mask is not None:
#                 col = field_border_mask[:, x]
#                 segs = Visualizer._mask_to_segments(col)
#                 for s0, s1 in segs:
#                     ax.plot([x, x], [s0, s1], color="lime", lw=0.9, alpha=0.55, ls="--")
#             else:
#                 ax.plot([x, x], [0, h], color="lime", lw=0.9, alpha=0.55, ls="--")

#         # ── gap segments ──────────────────────────────────────────────────────
#         for seg in result.gap_segments:
#             if seg["type"] == "horizontal":
#                 ax.plot(
#                     [seg["x1"], seg["x2"]], [seg["y1"], seg["y1"]],
#                     color="orange", lw=2.5, alpha=0.85, solid_capstyle="round",
#                 )
#             else:
#                 ax.plot(
#                     [seg["x1"], seg["x1"]], [seg["y1"], seg["y2"]],
#                     color="yellow", lw=2.5, alpha=0.85, solid_capstyle="round",
#                 )

#         # ── candidate bounding boxes ──────────────────────────────────────────
#         for i, (x, y, bw, bh) in enumerate(result.candidates):
#             is_match  = match_result is not None and match_result.get("matching_idx") == i
#             edgecolor = "#00FF00" if is_match else "#FF00FF"
#             ax.add_patch(patches.Rectangle(
#                 (x, y), bw, bh,
#                 linewidth=2.5, edgecolor=edgecolor, facecolor="none",
#             ))

#         # ── annotated ball position ───────────────────────────────────────────
#         if ball_annotation is not None:
#             x, y, w, h = ball_annotation
#             cx = x + w / 2
#             cy = y + h / 2
#             ax.add_patch(plt.Circle((cx, cy), radius=6, color="red", zorder=5))
#             ax.plot(cx, cy, "r+", markersize=14, markeredgewidth=2, zorder=6)

#         # ── title ─────────────────────────────────────────────────────────────
#         match_str = ""
#         if match_result is not None and ball_annotation is not None:
#             if match_result["matched"]:
#                 match_str = f"  ✓ MATCHED  (IoU={match_result['best_iou']:.2f})"
#             else:
#                 match_str = "  ✗ NOT matched"
#         ax.set_title(
#             f"{len(result.candidates)} candidate(s){match_str}",
#             fontsize=12, fontweight="bold",
#         )

#         # ── legend ────────────────────────────────────────────────────────────
#         legend = [
#             Line2D([0], [0], color="cyan",        lw=1.5, ls="--", label="H scanlines (field only)"),
#             Line2D([0], [0], color="lime",        lw=1.5, ls="--", label="V scanlines (field only)"),
#             Line2D([0], [0], color="orange",      lw=2.5,          label="H gaps"),
#             Line2D([0], [0], color="yellow",      lw=2.5,          label="V gaps"),
#             Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
#                    markeredgecolor="#FF00FF", markersize=10, lw=0, label="Candidate"),
#             Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
#                    markeredgecolor="#00FF00", markersize=10, lw=0, label="Matched candidate"),
#             Line2D([0], [0], color="deepskyblue", lw=2,            label="Field border"),
#             Line2D([0], [0], color="white",       lw=1.5,          label="Field lines"),
#             Line2D([0], [0], marker="o", color="red", markersize=8, lw=0, label="Ball annotation"),
#         ]
#         ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)
#         plt.tight_layout()

#         if save_path is not None:
#             plt.savefig(save_path, dpi=150)
#             print(f"    Saved → {save_path}")
#         else:
#             plt.show()
#         plt.close(fig)

#     @staticmethod
#     def _mask_to_segments(row_or_col: np.ndarray) -> list[tuple[int, int]]:
#         """Convert a 1-D boolean mask into a list of (start, end) run pairs."""
#         segs   = []
#         inside = False
#         start  = 0
#         for i, val in enumerate(row_or_col):
#             if val and not inside:
#                 start, inside = i, True
#             elif not val and inside:
#                 segs.append((start, i))
#                 inside = False
#         if inside:
#             segs.append((start, len(row_or_col) - 1))
#         return segs



# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    """
    Debug visualisation.

    Only content that lies *inside* the field border is drawn in detail
    (scanlines, gap segments, candidates).  The full image is shown as
    context but everything outside the field is dimmed.
    """

    @staticmethod
    def save(
        image:            np.ndarray,
        result:           DetectionResult,
        field_border:     list[tuple] | None = None,
        field_border_mask: np.ndarray | None = None,
        annotated_lines:  list | None = None,
        ball_annotation:  tuple | None = None,
        match_result:     dict | None = None,
        save_path:        Path | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 9))

        # Dim everything outside the field so the analyst can focus inside
        display_image = image.copy()
        if field_border_mask is not None:
            dim_factor = 0.35
            outside    = ~field_border_mask
            display_image[outside] = (display_image[outside] * dim_factor).astype(np.uint8)
        ax.imshow(display_image)

        h, w = image.shape[:2]

        # ── field border outline ──────────────────────────────────────────────
        if field_border and len(field_border) >= 3:
            poly = plt.Polygon(
                field_border, closed=True,
                linewidth=2, edgecolor="deepskyblue", facecolor="none", alpha=0.9,
            )
            ax.add_patch(poly)

        # ── annotated field lines (white overlay) ─────────────────────────────
        if annotated_lines:
            for line in annotated_lines:
                xs = [p[0] for p in line]
                ys = [p[1] for p in line]
                ax.plot(xs, ys, color="white", linewidth=1.5, alpha=0.7)

        # ── scanlines — only drawn where they pass through the field ─────────
        for y in result.scanlines_h:
            if field_border_mask is not None:
                # Draw only the in-field segments of this scanline
                row = field_border_mask[y]
                segs = Visualizer._mask_to_segments(row)
                for s0, s1 in segs:
                    ax.plot([s0, s1], [y, y], color="cyan", lw=0.9, alpha=0.55, ls="--")
            else:
                ax.plot([0, w], [y, y], color="cyan", lw=0.9, alpha=0.55, ls="--")

        for x in result.scanlines_v:
            if field_border_mask is not None:
                col = field_border_mask[:, x]
                segs = Visualizer._mask_to_segments(col)
                for s0, s1 in segs:
                    ax.plot([x, x], [s0, s1], color="lime", lw=0.9, alpha=0.55, ls="--")
            else:
                ax.plot([x, x], [0, h], color="lime", lw=0.9, alpha=0.55, ls="--")

        # ── gap segments ──────────────────────────────────────────────────────
        for seg in result.gap_segments:
            if seg["type"] == "horizontal":
                ax.plot(
                    [seg["x1"], seg["x2"]], [seg["y1"], seg["y1"]],
                    color="orange", lw=2.5, alpha=0.85, solid_capstyle="round",
                )
            else:
                ax.plot(
                    [seg["x1"], seg["x1"]], [seg["y1"], seg["y2"]],
                    color="yellow", lw=2.5, alpha=0.85, solid_capstyle="round",
                )

        # ── candidate bounding boxes ──────────────────────────────────────────
        for i, (x, y, bw, bh) in enumerate(result.candidates):
            is_match  = match_result is not None and match_result.get("matching_idx") == i
            edgecolor = "#00FF00" if is_match else "#FF00FF"
            ax.add_patch(patches.Rectangle(
                (x, y), bw, bh,
                linewidth=2.5, edgecolor=edgecolor, facecolor="none",
            ))

        if ball_annotation is not None:
            x, y, w, h = ball_annotation
            cx = x + w / 2
            cy = y + h / 2
            ax.add_patch(plt.Circle((cx, cy), radius=6, color="red", zorder=5))
            ax.plot(cx, cy, "r+", markersize=14, markeredgewidth=2, zorder=6)

        # ── title ─────────────────────────────────────────────────────────────
        match_str = ""
        if match_result is not None and ball_annotation is not None:
            if match_result["matched"]:
                match_str = f"MATCHED  (IoU={match_result['best_iou']:.2f})"
            else:
                match_str = "NOT matched"
        ax.set_title(
            f"{len(result.candidates)} candidate(s){match_str}",
            fontsize=12, fontweight="bold",
        )

        # ── legend ────────────────────────────────────────────────────────────
        legend = [
            Line2D([0], [0], color="cyan",        lw=1.5, ls="--", label="H scanlines (field only)"),
            Line2D([0], [0], color="lime",        lw=1.5, ls="--", label="V scanlines (field only)"),
            Line2D([0], [0], color="orange",      lw=2.5,          label="H gaps"),
            Line2D([0], [0], color="yellow",      lw=2.5,          label="V gaps"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                   markeredgecolor="#FF00FF", markersize=10, lw=0, label="Candidate"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                   markeredgecolor="#00FF00", markersize=10, lw=0, label="Matched candidate"),
            Line2D([0], [0], color="deepskyblue", lw=2,            label="Field border"),
            Line2D([0], [0], color="white",       lw=1.5,          label="Field lines"),
            Line2D([0], [0], marker="o", color="red", markersize=8, lw=0, label="Ball annotation"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            print(f"    Saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _mask_to_segments(row_or_col: np.ndarray) -> list[tuple[int, int]]:
        """Convert a 1-D boolean mask into a list of (start, end) run pairs."""
        segs   = []
        inside = False
        start  = 0
        for i, val in enumerate(row_or_col):
            if val and not inside:
                start, inside = i, True
            elif not val and inside:
                segs.append((start, i))
                inside = False
        if inside:
            segs.append((start, len(row_or_col) - 1))
        return segs


# Image I/O

def load_image_ycbcr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rgb_array, Y, Cb, Cr) as numpy arrays."""
    img   = Image.open(path)
    ycbcr = img.convert("YCbCr")
    w, h  = ycbcr.size
    shape = (h, w)
    y  = np.array(ycbcr.getdata(0), dtype=np.uint8).reshape(shape)
    cb = np.array(ycbcr.getdata(1), dtype=np.uint8).reshape(shape)
    cr = np.array(ycbcr.getdata(2), dtype=np.uint8).reshape(shape)
    return np.array(img.convert("RGB")), y, cb, cr


def load_annotation(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


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
        parts        = img_path.stem.split("_", 1)
        entries.append(dict(
            stem       = img_path.stem,
            image_path = img_path,
            json_path  = json_path,
            log_id     = parts[0] if parts else "?",
            frame_id   = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
        ))
    print(entries)
    return entries


# Frame processor — orchestrates one frame end-to-end
class FrameProcessor:
    def __init__(
        self,
        cam_info:    CameraInfo,
        classifier:  ColorClassifier,
        v_client,
        camera:      str = "TOP",
        vis_dir:     Path = Path("./visualizations"),
        step_scale:  float = 0.5,
        size_tol:    float = 0.76,
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
            annotations = annotation_data
          
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

        ball_ann = None
        field_border = None
        lines = []

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
                print(f"  Found {len(ls)} annotated line(s) with total {sum(len(l) for l in ls)} points")
                
        # Build pixel masks
        shape = img_rgb.shape
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
            img_rgb, result,
            field_border=field_border,
            field_border_mask=fb_mask,
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