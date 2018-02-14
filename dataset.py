import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

# data_type shouild come from one source. Now we have to set it both here and in model.py
data_type = 'KITTI'; # 'NYU' # 'KITTI' # 'forest'

if(data_type == 'NYU'):
    IMAGE_HEIGHT = 228
    IMAGE_WIDTH = 304
    TARGET_HEIGHT = 55
    TARGET_WIDTH = 74
elif(data_type == 'KITTI'):
    IMAGE_HEIGHT = 168
    IMAGE_WIDTH = 520
    TARGET_HEIGHT = 40
    TARGET_WIDTH = 128
else:
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 240
    TARGET_HEIGHT = 58
    TARGET_WIDTH = 58
    
CONFIDENCE_MAP = True

class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path):

        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        
        if(not CONFIDENCE_MAP):
            filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        else:
            filename, depth_filename, conf_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"], ["confidence"]])
        
        # input
        if(data_type == 'NYU'):
            jpg = tf.read_file(filename)
            image = tf.image.decode_jpeg(jpg, channels=3)
        else:
            png = tf.read_file(filename)
            image = tf.image.decode_png(png, channels=3)
        image = tf.cast(image, tf.float32)       
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        if(data_type == 'NYU' or data_type == 'forest'):
            depth = tf.div(depth, [255.0])
        else:
            depth = tf.div(depth, [64.0])      
            
        if(CONFIDENCE_MAP):
            conf_png = tf.read_file(conf_filename)
            confidence_map = tf.image.decode_png(conf_png, channels=1)
            confidence_map = tf.cast(confidence_map, tf.float32)
            confidence_map = tf.div(confidence_map, [255.0])
        #depth = tf.cast(depth, tf.int64)
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)

        if(CONFIDENCE_MAP):
            invalid_depth = tf.multiply(invalid_depth, confidence_map);
        
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths

#    def one_image(self, image_name):
#        # input
#        jpg = tf.read_file(filename)
#        image = tf.image.decode_jpeg(jpg, channels=3)
#        image = tf.cast(image, tf.float32)
        
        

def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)
