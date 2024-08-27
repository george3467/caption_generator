
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
import numpy as np
import keras_nlp
import json
import pickle

from preprocess import get_dataset, get_resize_image
from detector import Custom_Retinanet


def Caption_Transformer(embedding_dim, encoder_int_dim, decoder_int_dim, num_heads, vocab_size, num_objects, caption_length):
    """
    This function contains the transformer layers: 2 encoder transformers and 2 decoder layers
    """

    image_input = keras.Input(shape=[20, 20, 2048])
    object_input = keras.Input(shape=[num_objects,])
    caption_input = keras.Input(shape=[caption_length,])
    training = keras.Input(shape=[None,])


    # reshape to (batch_size, 400, 2048)
    image_features = layers.Reshape(target_shape=(-1, 2048))(image_input)

    image_encoder = keras_nlp.layers.TransformerEncoder(intermediate_dim=encoder_int_dim,
                                                        num_heads=num_heads)
    # image_output shape = (batch_size, 400, 2048)
    image_output = image_encoder(image_features, training=training)



    object_embedding = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=vocab_size,
                                                                    sequence_length=num_objects, 
                                                                    embedding_dim=embedding_dim,
                                                                    mask_zero=True)
    # objects shape = (batch_size, num_objects, embedding_dim)
    objects = object_embedding(object_input) 
    object_encoder = keras_nlp.layers.TransformerEncoder(intermediate_dim=encoder_int_dim, 
                                                         num_heads=num_heads)
    # object_output shape = (batch_size, num_objects, embedding_dim)
    object_output = object_encoder(objects, training=training)



    caption_embedding = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=vocab_size, 
                                                                    sequence_length=caption_length, 
                                                                    embedding_dim=embedding_dim,
                                                                    mask_zero=True)   
    # captions shape = (batch_size, caption_length, embedding_dim)
    captions = caption_embedding(caption_input)
    decoder_layer_1 = keras_nlp.layers.TransformerDecoder(intermediate_dim=decoder_int_dim, 
                                                          num_heads=num_heads)
    # decoder_1_output shape = (batch_size, caption_length, embedding_dim)
    decoder_1_output = decoder_layer_1(decoder_sequence=captions,
                                        encoder_sequence=image_output, 
                                        training=training)



    decoder_layer_2 = keras_nlp.layers.TransformerDecoder(intermediate_dim=decoder_int_dim, 
                                                          num_heads=num_heads)
    # decoder_2_output shape = (batch_size, caption_length, embedding_dim)
    decoder_2_output = decoder_layer_2(decoder_sequence=decoder_1_output, 
                                        encoder_sequence=object_output,
                                        training=training)
    # output shape = (batch_size, caption_length, vocab_size)
    output = layers.Dense(units=vocab_size, activation="softmax")(decoder_2_output)

    return keras.Model([image_input, object_input, caption_input, training], output)



class Caption_Model(keras.Model):
    """
    This model uses the custom retinanet model to obtain the objects present in each image.
    It then uses the input image features and the objects to generate a caption for the image.
    """

    def __init__(self,
                 retinanet_model,
                 embedding_dim,
                 encoder_int_dim,
                 decoder_int_dim,
                 num_heads,
                 vocab_size, 
                 caption_length,
                 index_to_object,
                 ):
        
        super().__init__()
        self.loss_tracker = keras.metrics.Mean()

        self.caption_length = caption_length
        self.index_to_object = index_to_object
        self.num_objects = 3

        # these variables are set later
        self.index_to_vocab = None
        self.sample_image = None

        self.retinanet_model = retinanet_model
        self.retinanet_model.trainable = False

        self.caption_transformer = Caption_Transformer(embedding_dim, 
                                                        encoder_int_dim, 
                                                        decoder_int_dim,
                                                        num_heads, 
                                                        vocab_size, 
                                                        self.num_objects,
                                                        self.caption_length,
                                                        )

        # vectorizer for caption inputs and caption labels
        # sequence_length = max_length + 1 to accomadate for caption labels shifted 1 to the right
        self.vectorizer = layers.TextVectorization(max_tokens=vocab_size,
                                                    output_sequence_length=caption_length+1, 
                                                    pad_to_max_tokens=True)

    def compile(self):
        super().compile() 
        self.caption_optimizer = keras.optimizers.Adam()
        self.caption_loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction='none')


    def detector_fn(self, retinanet_output):
        """
        This function chooses the top num_objects which have highest frequency.
        """

        # retinanet_output shape = (batch_size, num_anchors, num_classes) 
        batch_size = retinanet_output.shape[0]

        vectorized_objects = []
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
            chosen_ids = sorted_indices[ : self.num_objects]

            objects = tf.gather(self.index_to_object, chosen_ids)

            # combines the objects into one string
            object_string = objects[0] + " " + objects[1] + " " + objects[2]

            vectorized_objects.append(self.vectorizer(object_string)[ : self.num_objects])

        # vectorized_objects shape = (batch_size, num_objects)
        return tf.cast(vectorized_objects, dtype=tf.int32), object_string


    def train_step(self, batch_inputs):
        batch_image, batch_caption = batch_inputs

        retinanet_output, C5 = self.retinanet_model.predict(batch_image)

        # vectorized_objects shape = (batch_size, num_objects)
        vectorized_objects, _ = self.detector_fn(retinanet_output)

        batch_loss = 0
        # iterates over the 5 captions for each image
        for i in range(5):            
            # inputs are the first caption_length values
            caption_input = batch_caption[:, i, :-1]

            # labels are the last caption_length values
            caption_label = batch_caption[:, i, 1:]

            # padding_mask shape = (batch_size, caption_legnth)
            padding_mask = tf.math.not_equal(caption_label, 0)

            training=True
            # training the caption_transformer
            with tf.GradientTape() as tape:
                caption_output = self.caption_transformer([C5, vectorized_objects, caption_input, training])
                loss = self.caption_loss_fn(caption_label, caption_output)
                padding_mask = tf.cast(padding_mask, dtype=loss.dtype)
                loss *= padding_mask
                loss += tf.math.reduce_sum(loss) / tf.math.reduce_sum(padding_mask)
                batch_loss += loss
       
            caption_grads = tape.gradient(loss, self.caption_transformer.trainable_variables)
            self.caption_optimizer.apply_gradients(zip(caption_grads, self.caption_transformer.trainable_variables))

        self.loss_tracker.update_state(batch_loss)
        return {"loss": self.loss_tracker.result()}


    def predict(self, input_image):
        """
        This function predicts a caption for an input image. It can predict only one
        image at a time.
        The input_image must be of the shape (640, 640, 3)
        """
        # add a batch dimension to give the shape = (1, 640, 640, 3)
        input_image = input_image[tf.newaxis, ...]

        retinanet_output, C5 = self.retinanet_model.predict(input_image)

        # vectorized_objects shape = (batch_size, num_objects)
        vectorized_objects, object_string = self.detector_fn(retinanet_output)

        # initializes the caption output
        caption_output = "start "

        # iterates over the caption_length used during training
        for i in range(self.caption_length):

            # adding a batch axis and removing the last token as performed in training
            vector_caption = self.vectorizer(caption_output)[tf.newaxis, ...][:, :-1]

            training=False
            # vector_output shape = (1, caption_length, vocab_size)
            vector_output = self.caption_transformer([C5, vectorized_objects, vector_caption, training])

            # takes the index of the word with the highest probability
            # ignores the first two indices which are for "UNK" and "  "
            next_word = tf.math.argmax(vector_output[0, i, 2:]).numpy() + 2

            # gets the word from the word index
            next_word = self.index_to_vocab[next_word]

            # "end" indicates that the model is done predicting
            if next_word == "end":
                break

            # add the new word to the output
            caption_output += " " + next_word

        # remove the "start" token and remove any extra spaces
        return caption_output.replace("start ", "").strip(), object_string.numpy().strip()



def run_training():
    """
    This script trains the Caption model
    """

    file = open("labels/object_names.json")
    index_to_object = json.load(file)
    num_classes = len(index_to_object)
    index_to_object.append(" ")

    vocab_size = 1000
    caption_length = 15
    embedding_dim = 256
    encoder_intermediate_dim = 256
    decoder_intermediate_dim = 512
    num_heads = 1
    batch_size = 32
    num_images = 800
    
    # image_data shape = (num_images, 640, 640, 3) 
    image_data, _, caption_data = get_dataset(num_images, num_classes)

    retinanet_model = Custom_Retinanet(num_classes)
    retinanet_model.load_weights("detector_weights/checkpoint").expect_partial()

    model = Caption_Model(retinanet_model=retinanet_model,
                            embedding_dim=embedding_dim,
                            encoder_int_dim=encoder_intermediate_dim,
                            decoder_int_dim=decoder_intermediate_dim,
                            num_heads=num_heads,
                            vocab_size=vocab_size,
                            caption_length=caption_length,
                            index_to_object=index_to_object,
                            )
    
    model.sample_image = image_data[0]

    # adapting the vectorizer
    model.vectorizer.adapt(caption_data)

    # preparing the index_to_vocab dictionary
    vocab = model.vectorizer.get_vocabulary()
    model.index_to_vocab = dict(zip(range(len(vocab)), vocab))

    # vectorizing the captions
    vectorized_caption = model.vectorizer(caption_data)

    # combining the images and captions
    train_dataset = tf.data.Dataset.from_tensor_slices((image_data, vectorized_caption))

    train_dataset = train_dataset.shuffle(100).batch(batch_size, drop_remainder=True)

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile()
    loss = model.fit(train_dataset, epochs=100, callbacks=[callback])

    # saves the model weights and vectorizer
    model.save_weights("caption_weights_1/checkpoint")
    vectorizer_dict = {'config': model.vectorizer.get_config(), 
                        'weights': model.vectorizer.get_weights()}            
    pickle.dump(vectorizer_dict, open("caption_weights_1/vectorizer.pkl", "wb"))



def run_inference():
    """
    This script runs inference on the Caption model
    """
    
    file = open("labels/object_names.json")
    index_to_object = json.load(file)
    num_classes = len(index_to_object)
    index_to_object.append(" ")

    vocab_size = 1000
    caption_length = 15
    embedding_dim = 256
    encoder_intermediate_dim = 256
    decoder_intermediate_dim = 512
    num_heads = 1
    retinanet_model = Custom_Retinanet(num_classes)
    retinanet_model.load_weights("detector_weights/checkpoint").expect_partial()

    new_model = Caption_Model(retinanet_model=retinanet_model,
                                embedding_dim=embedding_dim,
                                encoder_int_dim=encoder_intermediate_dim,
                                decoder_int_dim=decoder_intermediate_dim,
                                num_heads=num_heads,
                                vocab_size=vocab_size,
                                caption_length=caption_length,
                                index_to_object=index_to_object,
                                )
    
    new_model.load_weights("caption_weights/checkpoint").expect_partial()

    # loading the vectorizer
    vectorizer_dict = pickle.load(open("caption_weights/vectorizer.pkl", "rb"))
    new_model.vectorizer.from_config(vectorizer_dict['config'])
    new_model.vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_model.vectorizer.set_weights(vectorizer_dict['weights'])

    # preparing the index_to_vocab dictionary
    vocab = new_model.vectorizer.get_vocabulary()
    new_model.index_to_vocab = dict(zip(range(len(vocab)), vocab))

    path = "sample_images/000000086220.jpg"
    image = keras.utils.load_img(path)
    image = keras.utils.img_to_array(image)

    # only include images with 3 dimensions (height, width, channels)
    if len(image.shape) == 3:
        image, _ = get_resize_image(image)
        image = tf.cast(image, dtype=tf.float32)
        output = new_model.predict(image)
        print("Objects Detected: ", output[1].split())
        print("Caption: ", output[0])




