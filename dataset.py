
import json
from collections import defaultdict
import zipfile
import wget

def download_dataset():
    wget.download("http://images.cocodataset.org/zips/val2017.zip")
    with zipfile.ZipFile("val2017.zip", "r") as z_fp:
        z_fp.extractall("./")

    wget.download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    with zipfile.ZipFile("annotations_trainval2017.zip", "r") as z_fp:
        z_fp.extractall("./")



def sort_dataset():
    file = open("./annotations/instances_val2017.json")
    instances = json.load(file)


    # This script stores the bbox and object_id for each image
    object_names = [instance['name'] for instance in instances['categories']]
    json_object = json.dumps(object_names)
    with open("labels/object_names.json", "w") as outfile:
        outfile.write(json_object)

    original_classId_to_Name = {}
    for instance in instances['categories']:
        original_classId_to_Name[instance['id']] = instance['name']

    imageID_to_Bbox_classID = defaultdict(list)
    for instance in instances["annotations"]:
        object_name = original_classId_to_Name[instance["category_id"]]
        object_id = object_names.index(object_name)
        imageID_to_Bbox_classID[instance["image_id"]].append({"bbox": instance["bbox"], "object_id": object_id})


    # This scripts stores the caption data for each image
    file = open("./annotations/captions_val2017.json")
    captions = json.load(file)
    imageID_to_caption = defaultdict(list)
    for caption in captions["annotations"]:
        imageID_to_caption[caption["image_id"]].append(caption["caption"])


    # This script combines the information in the imageID_to_Bbox_classID and imageID_to_caption files
    imageID_to_labels = []
    for image_id in imageID_to_Bbox_classID.keys():
        imageID_to_labels.append({"image_id": image_id, "bbox_classes": imageID_to_Bbox_classID[image_id], "captions": imageID_to_caption[image_id]})
    json_object = json.dumps(imageID_to_labels)
    with open("labels/imageID_to_labels.json", "w") as outfile:
        outfile.write(json_object)





