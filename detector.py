
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
from preprocess import get_dataset, get_resize_image
import json
import numpy as np


def detector_fn(index_to_object, retinanet_output, num_objects=4):
    """
    This function chooses num_objects which have highest frequency.
    """
    # retinanet_output shape = (batch_size, num_anchors, num_classes) 
    batch_size = retinanet_output.shape[0]

    object_list = []
    for i in range(batch_size):

        # index of the highest probability for each anchor box
        # object_ids shape = (num_anchors,)
        object_ids = tf.argmax(retinanet_output[i], axis=-1)

        # gets the frequency of each object
        # object_counts shape = (num_classes,)
        object_counts = tf.math.bincount(object_ids)

        # objects indices listed in descending order of object_counts
        # sorted_indices shape = (num_classes,)
        sorted_indices = tf.argsort(object_counts, direction='DESCENDING')

        # choose the top num_objects that have the highest probability
        chosen_ids = sorted_indices[ : num_objects]

        # convert object index to object name
        objects = tf.gather(index_to_object, chosen_ids)

        # combines the objects into one string
        object_string = objects[0] + " " + objects[1] + " " + objects[2]

    # length of object_string = batch_size
    return object_string



class Pyramid_Layer(layers.Layer):
    """
    The RetinaNet paper's pyramid's input = [C3, C4, C5] and output = [P3, P4, P5, P6, P7]
    In this model, the pyramid's input = C5 and output = [P5, P6, P7]
    The convolution layers are defined according to the RetinaNet paper.
    """

    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(filters=self.num_filters, kernel_size=1, strides=1, padding="same")
        self.conv_2 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=1, padding="same")
        self.conv_3 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=2, padding="same")
        self.activation = layers.ReLU()
        self.conv_4 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=2, padding="same")

    def call(self, C5):
        # P5 shape = (batch_size, 20, 20, num_filters)
        P5 = self.conv_1(C5)

        # P5 shape = (batch_size, 20, 20, num_filters)
        P5 = self.conv_2(P5)

        # P6 shape = (batch_size, 10, 10, num_filters)
        P6 = self.conv_3(C5)
        P7 = self.activation(P6)

        # P7 shape = (batch_size, 5, 5, num_filters)
        P7 = self.conv_4(P7)
        return P5, P6, P7



class Classifier_Layer(layers.Layer):
    """
    This layer is used to classify the labels of each box.
    The RetinaNet paper uses 4 Convolution(filters=256, kernel=3) layers 
    whereas this model uses only 3 of these layers.
    The final convolution layer is the same as the RetinaNet paper.
    """
    def __init__(self, num_classes, num_filters):
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(filters=self.num_filters, kernel_size=3, padding="same")
        self.activation_1 = layers.ReLU()
        self.conv_2 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=1, padding="same")
        self.activation_2 = layers.ReLU()
        self.conv_3 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=1, padding="same")
        self.activation_3 = layers.ReLU()

        self.conv_final = layers.Conv2D(filters=9 * (self.num_classes+1), kernel_size=3, strides=1, padding="same")
        self.reshape = layers.Reshape(target_shape=(-1, self.num_classes+1))

    def call(self, x):
        x = self.conv_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.activation_2(x)
        x = self.conv_3(x)
        x = self.activation_3(x)

        x = self.conv_final(x)
        x = self.reshape(x)
        return x



class Custom_Retinanet(keras.Model):
    """
    This model applies the pretrained ResNet model to the input image and
    extracts the C5 layer from the model. It then applies the pyramid layer and then
    the classification layer.
    The output is shaped as (batch_size, num_anchors, num_classes) in order 
    to apply the Categorical CrossEntropy Loss on it.
    """
    def __init__(self, num_classes, num_filters=256):
        super().__init__()
        self.pyramid = Pyramid_Layer(num_filters)
        self.classifier = Classifier_Layer(num_classes, num_filters)

        # The RetinaNet paper suggests extracting layers [C3, C4, C5] from ResNet50 model.
        # In this model, we are only extracting C5.
        self.resnet_model = keras.applications.ResNet50(include_top=False, input_shape=[640, 640, 3])

        # to prevent the pretrained model from training
        self.resnet_model.trainable=False

        self.resnet_preprocess = keras.applications.resnet.preprocess_input

        self.sample_image = None
        self.index_to_object = None

    def retinanet_fn(self, C5):
        """
        This function takes the 3 outputs from the pyramid_layer and applies the 
        classifier_layer to each of the 3 outputs.
        """

        # pyramid_output = [P5, P6, P7]
        pyramid_output = self.pyramid(C5)

        output_list = []
        for output in pyramid_output:
            classifier_output = self.classifier(output)
            output_list.append(classifier_output)

        # retinanet_output shape = (batch_size, num_anchors, num_classes)
        retinanet_output = tf.concat(output_list, axis=1)

        # The RetinaNet paper uses sigmoid activation with their Focal Loss function for classification.
        # Since this model uses a Categorical CrossEntropy Loss for classification,
        # a Softmax activation is used instead.
        retinanet_output = layers.Softmax()(retinanet_output)

        return retinanet_output

    def call(self, batch_image):
        # preprocessing images for the resnet model
        batch_image = self.resnet_preprocess(batch_image)

        # extracting features from the pretrained ResNet model
        C5 = self.resnet_model(batch_image)
        
        return self.retinanet_fn(C5)
    
    def predict(self, batch_image):
        # preprocessing images for the resnet model
        batch_image = self.resnet_preprocess(batch_image)

        # extracting features from the pretrained ResNet model
        C5 = self.resnet_model(batch_image)
        
        return self.retinanet_fn(C5), C5
    


class Custom_Loss_Fn(tf.losses.Loss):
    """
    This custom loss function ignores anchor boxes that are assigned as a background.
    """
    def __init__(self, background_label):
        super().__init__()

        # Using Sparse Categorical since labels are not one-hot encoded.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction='none')
        self.background_label = background_label

    def call(self, y_true, y_pred):
        # Anchor boxes which have background_label have weight = 0.1
        # Other boxes have weight = 1.
        label_weights = tf.where(tf.math.equal(y_true, self.background_label), 0.1, 1)
        return self.loss_fn(y_true, y_pred, sample_weight=label_weights)



def run_training():
    """
    This scripts trains the custom_retinent model.
    """

    file = open("labels/object_names.json")
    index_to_object = json.load(file)
    num_classes = len(index_to_object)
    index_to_object.append(" ")

    num_images = 800
    # image_data shape = (num_images, 640, 640, 3) 
    # label_data shape = (num_images, 4725)
    image_data, label_data, _ = get_dataset(num_images, num_classes)
    
    retinanet_model = Custom_Retinanet(num_classes)
    retinanet_model.sample_image = image_data[ : 3]
    retinanet_model.index_to_object = index_to_object

    dataset = tf.data.Dataset.from_tensor_slices((image_data, label_data))
    train_dataset = dataset.shuffle(100).batch(16, drop_remainder=True)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    retinanet_model.compile(loss = Custom_Loss_Fn(background_label=num_classes), optimizer = keras.optimizers.Adam())
    loss = retinanet_model.fit(train_dataset, epochs=100, callbacks=[callback])

    retinanet_model.save_weights("detector_weights_1/checkpoint")



def run_inference():
    """
    This script runs inference on the custom_retinent model
    """

    file = open("labels/object_names.json")
    index_to_object = json.load(file)
    num_classes = len(index_to_object)
    index_to_object.append(" ")

    path = "sample_images/000000086220.jpg"
    image = keras.utils.load_img(path)
    image = keras.utils.img_to_array(image)

    # only include images with 3 dimensions (height, width, channels)
    if len(image.shape) == 3:
        image, ratio = get_resize_image(image)
        image = tf.cast(image, dtype=tf.float32)[tf.newaxis, ...]

        new_model = Custom_Retinanet(num_classes)
        new_model.load_weights("detector_weights/checkpoint").expect_partial()

        prediction, C5 = new_model.predict(image)
        objects = detector_fn(index_to_object, prediction)
        
        return objects













