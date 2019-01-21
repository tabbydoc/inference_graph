import sys
import os
import fnmatch
import getopt
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


flags = tf.app.flags
flags.DEFINE_string('path_to_frozen_graph', '', '')
flags.DEFINE_string('path_to_labels', '', '')
flags.DEFINE_string('path_to_images', '', '')
flags.DEFINE_string('output_dir', '', '')
FLAGS = flags.FLAGS


def main(_):
    frozen_graph_path = FLAGS.path_to_frozen_graph
    images_dir = FLAGS.path_to_images
    labels_path = FLAGS.path_to_labels
    output_dir = FLAGS.output_dir

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

    for root, directories, files in os.walk(images_dir):
        for file_name in fnmatch.filter(files, "*.jpeg"):
            path = os.path.join(root, file_name)
            name = os.path.splitext(os.path.split(path)[1])[0]
            save_path = os.path.join(output_dir, name + ".jpeg")
            image = cv2.imread(path)
            #bimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #ret, bimage = cv2.threshold(bimage, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #transformed_image = transform_image(bimage)
            output_dict = run_inference_for_single_image(image, detection_graph)
            if output_dict is None:
                continue
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imwrite(save_path, image)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.9), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            return output_dict


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def rgb_to_binary(img):
    gray_img = img.convert("L")
    binary_img = gray_img.point(lambda x: 0 if x < 128 else 255, "1")
    return binary_img


def ndarray_to_image(data):
    return Image.fromarray(data.astype("uint8"))


def transform_image(img):
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
    img = cv2.merge((r, g, b))
    return img

    # img_blur = img.filter(ImageFilter.EMBOSS)
    # img_gaus = img_blur.filter(ImageFilter.GaussianBlur())
    # return img_blur


if __name__ == "__main__":
    main(sys.argv[1:])
