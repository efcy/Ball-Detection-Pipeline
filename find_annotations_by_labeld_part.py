import json
import os
import random
from pathlib import Path

import requests
from vaapi.client import Vaapi

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

TARGET_POLYGON_LABELS = {"line", "field border"}
OUTPUT_DIR = Path("./test_images")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def normalize_label(label: str) -> str:
    return label.strip().lower()


def extract_labels(annotations: list[dict], annotation_type: str) -> set[str]:
    """
    Extract normalized labels of a specific annotation type.
    """
    found = set()

    for ann in annotations:
        if ann.get("type") != annotation_type:
            continue

        labels = ann.get("value", {}).get(annotation_type, [])

        if not isinstance(labels, list):
            continue

        found.update(normalize_label(l) for l in labels)

    return found


def relevance_type(
    polygon_labels: set[str],
    rectangle_labels: set[str],
) -> int:
    """
    Returns:
        1 -> relevant (ball + line/field border)
        0 -> line/field border but no ball
       -1 -> irrelevant
    """

    has_ball = "ball" in rectangle_labels
    has_relevant_polygon = bool(
        TARGET_POLYGON_LABELS & polygon_labels
    )
    has_line = "line" in polygon_labels
    has_field_border = "field border" in polygon_labels

    if has_ball and has_line and has_field_border:
        return 1

    if has_relevant_polygon:#'line' in polygon_labels:
        return 0

    return -1


def download_image(url: str, output_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"Failed download: {url}")
            return False

        with open(output_path, "wb") as f:
            f.write(response.content)

        return True

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False


# MAIN
v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True)

    logs = v_client.logs.list()

    #sample_size = min(500, len(logs))
    #sample_logs = random.sample(logs, sample_size)
    sample_logs = logs
    camera = "TOP"

    line_frames = []
    field_border_frames = []
    frame_data = {}
    frame_potential_data = {}

    print("Fetching images...")
    for log in sample_logs:
        numeric_log_id = str(log).split(" ")[0]
        print(f"\nProcessing log {numeric_log_id}...")
        image_obj_list = v_client.image.list(
            log=numeric_log_id,
            camera=camera,
            has_annotations=True,
        )

        skipped_counter = 0

        for img_obj in image_obj_list:
            frame_id = img_obj.frame.id
            frame_number = img_obj.frame.frame_number
            annotations = getattr(img_obj, "annotation", None) or []

            polygon_labels = extract_labels(
                annotations,
                annotation_type="polygonlabels",
            )

            rectangle_labels = extract_labels(
                annotations,
                annotation_type="rectanglelabels",
            )

            relevance = relevance_type(
                polygon_labels,
                rectangle_labels,
            )

            if relevance == -1:
                continue

            elif relevance == 0:
                print(f"vars: {vars(img_obj)}")
                print(
                    f"No ball but relevant polygon labels."
                    f"log={numeric_log_id} frame={frame_id} and url={img_obj.labelstudio_url}"
                )
                frame_potential_data[(numeric_log_id, frame_id)] = {
                    "frame_number": frame_number,
                    "log_id": numeric_log_id,
                    "URL": img_obj.labelstudio_url}
                continue

            else: 
            # Relevant frame
           
                img_url = (
                    "https://logs.berlin-united.com/"
                    + img_obj.image_url
                )

                img_path = OUTPUT_DIR / f"{numeric_log_id}_{frame_id}.jpg"
                ann_path = OUTPUT_DIR / f"{numeric_log_id}_{frame_id}.json"

                success = download_image(img_url, img_path)

                if not success:
                    continue

                with open(ann_path, "w") as f:
                    json.dump(annotations, f, indent=2)

                frame_data[(numeric_log_id, frame_id)] = {
                    "frame_number": frame_number,
                    "image_path": str(img_path),
                    "polygon_labels": list(polygon_labels),
                    "rectangle_labels": list(rectangle_labels),
                    "annotation": annotations,
                    "URL": img_obj.labelstudio_url,
                }

    # OUTPUT
    log_counts = {}
    print("\n--- RESULTS ---")
    print(f"\nTotal relevant frames: {len(frame_data)}")
    for (log_id, frame_id), data in frame_data.items():
        print(f"\nLog: {log_id}, Frame: {frame_id}")
        print(f"Frame number: {data['frame_number']}")
        print(f"Polygon labels: {len(data['polygon_labels'])}")
        print(f"Rectangle labels: {len(data['rectangle_labels'])}")
        print(f"URL: {img_obj.labelstudio_url}")
    print(f"----------------------")
    print(f"\nTotal potential frames (no ball but relevant polygon): {len(frame_potential_data)}")
    for (log_id, frame_id), data in frame_potential_data.items():
        if log_counts.get(log_id, 0) < 2:
            print(f"\nLog: {log_id}, Frame: {frame_id}")
            print(f"Frame number: {data['frame_number']}")
            print(f"URL: {data['URL']}")
            log_counts[log_id] = log_counts.get(log_id, 0) + 1