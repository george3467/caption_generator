import tensorflow as tf
keras = tf.keras
import numpy as np
import json


# converts coordinates of bounding box from (leftup_x, leftup_y, width, height)
# to (center_x, center_y, width, height)
leftup_wh_to_center_wh = lambda bbox: tf.concat([bbox[..., :2] + 0.5 * bbox[..., 2:], 
                                                 bbox[..., 2:]], axis=-1)

# converts coordinates of bounding box from (center_x, center_y, width, height)
# to (leftup_x, leftup_y, rightdown_x, rightdown_y)
center_wh_to_corners = lambda bbox: tf.concat([bbox[..., :2] - 0.5 * bbox[..., 2:], 
                                               bbox[..., :2] + 0.5 * bbox[..., 2:]], axis=-1)

def generate_anchors(image_length=640):
    """
    This function generates anchor boxes for each image.
    """
    anchor_list = []

    for i, area in enumerate([x ** 2 for x in [64.0, 256.0, 512.0]]):

        # calculates 9 different anchor box dimensions
        anchor_width_height = np.zeros((9, 2))

        # The ratio, and scale values are according to the RetinaNet paper
        for j, ratio in enumerate([0.5, 1.0, 2.0]):
            for k, scale in enumerate([2 ** 0, 2 ** 1 / 3, 2 ** 2 / 3]):

                # calculates anchor box width and height based on area and ratio values            
                width = tf.math.sqrt(tf.multiply(ratio, area))
                height = width/ratio

                # scales anchor box width and height
                width_height = tf.multiply(scale, tf.stack([width, height], axis=0))

                anchor_width_height[j*3+k] = width_height

        anchor_width_height = tf.cast(anchor_width_height, dtype=tf.float32)

        # distance between anchor box centers
        stride = tf.math.pow(2, i+5)

        # number of boxes along each axis
        num = tf.math.ceil(image_length / stride)

        # repeats anchor width and height values for num^2 boxes
        width_height = tf.tile(anchor_width_height[tf.newaxis, tf.newaxis, ...], 
                               multiples=[num, num, 1, 1])

        # calculates coordinates for anchor box centers using stride
        xy_centers = 0.5 + tf.range(num, dtype=tf.float32)
        anchor_centers = tf.cast(stride, dtype=tf.float32) \
                         * tf.stack(tf.meshgrid(xy_centers, xy_centers), axis=-1)
        
        # repeats the coordinates for the anchor box centers for 9 different anchor box dimensions
        anchor_centers = tf.tile(tf.expand_dims(anchor_centers, axis=-2), multiples=[1, 1, 9, 1])
        
        # concatenates anchor box center coordinates and dimensions and
        # reshapes to give a list of anchor boxes of shape (num_anchor_box, 4)
        anchors = tf.reshape(tf.concat([anchor_centers, width_height], axis=-1), 
                             shape=[(num**2) * 9, 4])
        anchor_list.append(anchors)

    # concatenates to give a list of anchor boxes of shape (total_num_anchor_box, 4)
    anchor_list = tf.concat(anchor_list, axis=0)
    return anchor_list


def intersection_over_union(bbox1, bbox2):
    """
    This function calculates the intersection over union of box pairs.
    Box inputs must be in the format (center_x, center_y, width, height).
    """
    
    # Note: The extra dimension added in bbox1_area and bbox1 allows all the boxes in bbox1 list
    # to be paired with all the boxes in the bbox2 list.

    # calculates areas of the boxes using width * height
    # bbox1_area shape = (num_bbox1, 1)
    bbox1_area = tf.multiply(bbox1[:, 2], bbox1[:, 3])[:, tf.newaxis]
    # bbox2_area shape = (num_bbox2,)
    bbox2_area = tf.multiply(bbox2[:, 2], bbox2[:, 3])

    # converts the boxes to the format (leftup_x, leftup_y, rightdown_x, rightdown_y)
    # bbox1 shape = (num_bbox1, 1, 4)
    bbox1 = center_wh_to_corners(bbox1)[:, tf.newaxis, :]
    # bbox1 shape = (num_bbox2, 4)
    bbox2 = center_wh_to_corners(bbox2)

    """
    To find intersection area of the box pairs, first calculate the minimum rightdown_x
    and rightdown_y and the maximum leftup_x and leftup_y of the two boxes.
    Then intersection_width = min_rightdown_x - max_leftup_x
    And intersection_height = min_rightdown_y - max_leftup_y
    """

    # intersection_width_height shape = (num_bbox1, num_bbox2, 2)
    intersection_width_height = tf.minimum(bbox1[..., 2:], bbox2[:, 2:]) \
                                 - tf.maximum(bbox1[..., :2], bbox2[:, :2])
    
    intersection_width_height = tf.clip_by_value(intersection_width_height, 
                                                 clip_value_min=0.0, clip_value_max=1e4)

    # area = width * height
    # intersection_area shape = (num_bbox1, num_bbox2)
    intersection_area = tf.multiply(intersection_width_height[..., 0],
                                     intersection_width_height[..., 1])
    
    eps = np.finfo(np.float32).eps.item()

    # union_area = (area of box 1) + (area of box 2) - (intersection area of box 1 and 2)
    # intersection_over_union = intersection_area / union_area
    iou = intersection_area / tf.maximum(bbox1_area + bbox2_area - intersection_area, eps)

    # iou shape = (num_bbox1, num_bbox2)
    return tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)


def label_selection(label_box, object_id, anchor_box, background_label):
    """
    This function calculates the object label for the anchor boxes using the label boxes.

    The RetinaNet paper assigns object labels for boxes with iou > 0.5 and a background label 
    for iou < 0.4. For 0.4 < iou < 0.5, the box is ignored.

    This model assigns object labels for iou > 0.2 and a background label for iou < 0.2. 
    In the classification loss function, the background_label is given a lower weight 
    relative to the object labels.
    """

    iou = intersection_over_union(anchor_box, label_box)

    # For each anchor box, calculates the index of the label box that has the highest iou.
    # closest_object_id_index shape = (num_anchors,)
    closest_object_id_index = tf.argmax(iou, axis=-1)

    # obtains object_id using the label box indices
    # closest_object_id shape = (num_anchors,)
    closest_object_id = tf.gather(params=object_id, indices=closest_object_id_index)

    # highest_iou shape = (num_anchors,)
    highest_iou = tf.reduce_max(iou, axis=-1)

    # label shape = (num_anchors,)
    label = tf.where(tf.greater_equal(highest_iou, 0.2), closest_object_id, background_label)

    return label


def get_resize_image(image, image_length=640):
    """
    This function resizes the images to the shape (640, 640, 3).
    The height to width ratio of the image is not changed.
    """

    # resize the longer side to 640
    max_side = tf.reduce_max(image.shape)
    ratio = image_length / max_side

    new_shape = tf.cast(image.shape[:2], dtype = tf.float32) * tf.cast(ratio, dtype = tf.float32)
    new_shape = tf.cast(new_shape, tf.int32)
    image = tf.image.resize(image, new_shape)

    # pads the image with zeros at the bottom and the right to create the shape (640, 640, 3).
    image = tf.image.pad_to_bounding_box(image, 0, 0, image_length, image_length)
    return image, ratio


def get_dataset(num_images, num_classes):
    """
    This function retrieves the images, assigns labels to each image's anchor boxes,
    and retrieves first 5 captions for each image
    """
    background_label = num_classes

    file = open("labels/imageID_to_labels.json")
    data = json.load(file)

    # takes every 5th image to ensure a diverse dataset is used for training
    data = data[::5][ : num_images]

    # anchor_box shape = (4725, 4) where num_anchors = 4725
    anchor_box = generate_anchors()

    image_list, label_list, caption_list = [], [], []
    for entry in data:
        imageID = str(entry["image_id"])

        # image file names have extra zeros before the image_id number 
        # for a total of 12 digits
        zero_padding = "".join(["0" for i in range(12 - len(imageID))])
        imageID = "".join([zero_padding, imageID])

        path = "val2017/" + str(imageID) + ".jpg"
        image = keras.utils.load_img(path)
        image = keras.utils.img_to_array(image)

        # only include images with 3 dimensions (height, width, channels)
        if len(image.shape) == 3:
            image, ratio = get_resize_image(image)
            image_list.append(image)

            bbox = [a["bbox"] for a in entry["bbox_classes"]]
            bbox = leftup_wh_to_center_wh(tf.cast(bbox, dtype=tf.float32))

            # resize bbox to match with the resize of the image
            bbox *= tf.cast(ratio, dtype=tf.float32)

            object_id = [a["object_id"] for a in entry["bbox_classes"]]
            object_id = tf.cast(object_id, dtype=tf.int32)

            # get labels for each anchor box 
            label = label_selection(bbox, object_id, anchor_box, background_label)
            label_list.append(label)

            # adding start and end tokens to captions
            captions = [["start " + cap + " end"] for cap in entry["captions"]]

            # only take first 5 captions for each image
            captions = captions[ : 5]
            caption_list.append(captions)

    # all_image shape = (num_images, 640, 640, 3) 
    all_image = tf.cast(image_list, dtype=tf.float32)

    # all_label = (num_images, 1125)
    all_label = tf.cast(label_list, dtype=tf.float32)

    return all_image, all_label, caption_list






