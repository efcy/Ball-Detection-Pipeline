from vaapi.client import Vaapi
import os

def main(v_client):
    image_obj_list = v_client.image.list(
        camera="BOTTOM",
        has_annotations=True,
    )
    for img in image_obj_list:
        #print(img.annotation)
        for ann in img.annotation:
            if ann["type"] == "rectanglelabels" and "Ball" in ann["value"]["rectanglelabels"] :
                print(ann)
                print()
                break
        

if __name__ == "__main__":
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    main(v_client)
    