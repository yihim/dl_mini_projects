import tensorflow as tf
import gradio as gr
import cv2
import numpy as np

def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

def difference(x,y):
  return tf.reduce_sum(tf.square(y-x))

def yolo_loss(y_true, y_pred):
    target = y_true[..., 0]

    ###################### OBject Loss
    y_pred_extract = tf.gather_nd(y_pred, tf.where(target[:] == 1))
    y_target_extract = tf.gather_nd(y_true, tf.where(target[:] == 1))

    rescaler = tf.where(target[:] == 1) * SPLIT_SIZE
    upscaler_1 = tf.concat([rescaler[:, 1:], tf.zeros([len(rescaler), 2], dtype=tf.int64)], axis=-1)

    target_upscaler_2 = tf.repeat([[float(SPLIT_SIZE), float(SPLIT_SIZE), H, W]],
                                  repeats=[len(rescaler)], axis=0) * tf.cast(y_target_extract[..., 1:5],
                                                                             dtype=tf.float32)
    pred_1_upscaler_2 = tf.repeat([[float(SPLIT_SIZE), float(SPLIT_SIZE), H, W]],
                                  repeats=[len(rescaler)], axis=0) * tf.cast(y_pred_extract[..., 1:5], dtype=tf.float32)
    pred_2_upscaler_2 = tf.repeat([[float(SPLIT_SIZE), float(SPLIT_SIZE), H, W]],
                                  repeats=[len(rescaler)], axis=0) * tf.cast(y_pred_extract[..., 6:10],
                                                                             dtype=tf.float32)

    target_orig = tf.cast(upscaler_1, dtype=tf.float32) + target_upscaler_2
    pred_1_orig = tf.cast(upscaler_1, dtype=tf.float32) + pred_1_upscaler_2
    pred_2_orig = tf.cast(upscaler_1, dtype=tf.float32) + pred_2_upscaler_2

    mask = tf.cast(tf.math.greater(compute_iou(target_orig, pred_2_orig),
                                   compute_iou(target_orig, pred_1_orig)), dtype=tf.int32)

    y_pred_joined = tf.transpose(tf.concat([tf.expand_dims(y_pred_extract[..., 0], axis=0),
                                            tf.expand_dims(y_pred_extract[..., 5], axis=0)], axis=0))

    obj_pred = tf.gather_nd(y_pred_joined, tf.stack([tf.range(len(rescaler)), mask], axis=-1))

    object_loss = difference(tf.cast(obj_pred, dtype=tf.float32)
                             , tf.cast(tf.ones([len(rescaler)]), dtype=tf.float32))

    ####################### For No object
    y_pred_extract = tf.gather_nd(y_pred[..., 0:B * 5], tf.where(target[:] == 0))
    y_target_extract = tf.zeros(len(y_pred_extract))

    no_object_loss_1 = difference(tf.cast(y_pred_extract[..., 0], dtype=tf.float32)
                                  , tf.cast(y_target_extract, dtype=tf.float32))

    no_object_loss_2 = difference(tf.cast(y_pred_extract[..., 5], dtype=tf.float32)
                                  , tf.cast(y_target_extract, dtype=tf.float32))

    no_object_loss = no_object_loss_1 + no_object_loss_2

    ######################## For OBject class loss
    y_pred_extract = tf.gather_nd(y_pred[..., 10:], tf.where(target[:] == 1))
    class_extract = tf.gather_nd(y_true[..., 5:], tf.where(target[:] == 1))

    class_loss = difference(tf.cast(y_pred_extract, dtype=tf.float32)
                            , tf.cast(class_extract, dtype=tf.float32))

    ######################### For object bounding box loss
    y_pred_extract = tf.gather_nd(y_pred[..., 0:B * 5], tf.where(target[:] == 1))
    centre_joined = tf.stack([y_pred_extract[..., 1:3], y_pred_extract[..., 6:8]], axis=1)
    centre_pred = tf.gather_nd(centre_joined, tf.stack([tf.range(len(rescaler)), mask], axis=-1))
    centre_target = tf.gather_nd(y_true[..., 1:3], tf.where(target[:] == 1))

    centre_loss = difference(centre_pred, centre_target)

    size_joined = tf.stack([y_pred_extract[..., 3:5], y_pred_extract[..., 8:10]], axis=1)

    size_pred = tf.gather_nd(size_joined, tf.stack([tf.range(len(rescaler)), mask], axis=-1))
    size_target = tf.gather_nd(y_true[..., 3:5], tf.where(target[:] == 1))

    size_loss = difference(tf.math.sqrt(tf.math.abs(size_pred)), tf.math.sqrt(tf.math.abs(size_target)))
    box_loss = centre_loss + size_loss

    lambda_coord = 5.0
    lambda_no_obj = 0.5

    loss = object_loss + (lambda_no_obj * no_object_loss) + tf.cast(lambda_coord * box_loss,
                                                                    dtype=tf.float32) + tf.cast(class_loss,
                                                                                                dtype=tf.float32)
    return loss

classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
         'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

model = tf.keras.models.load_model("pascal_voc_2012_yolo_efficientnetv2m.h5", custom_objects={"yolo_loss": yolo_loss})

H, W = 224, 224

def model_test(image_array):
    try:
        img = cv2.resize(image_array, (H, W))

        image = tf.cast(tf.image.resize(image_array, [H, W]), dtype=tf.float32)
        output = model.predict(tf.expand_dims(image, axis=0))
        # print(output.shape)

        THRESH = .25

        # the object posistions is to get the first two (B) 5 values where eg. 0/1, norm_x_center, norm_y_center, norm_w, norm_h
        # based on the defined threshold
        object_positions = tf.concat([tf.where(output[..., 0] >= THRESH), tf.where(output[..., 5] >= THRESH)], axis=0)
        # print(object_positions)
        selected_output = tf.gather_nd(output, object_positions)
        # print(selected_output)
        final_boxes = []
        final_scores = []

        for i, pos in enumerate(object_positions):
            # to loop the two (B) labels
            for j in range(2):
                # to get each of the first (label -- 0 to 1) of the five values
                # selected_output[0][0] & selected_output[0][5]
                if selected_output[i][j * 5] > THRESH:
                    # output[pos[0]][pos[1]][pos[2]] -- this is to get the 30 values output
                    # [(j*5)+1:(j*5)+5] -- this is to get two (b) bounding box from the 30 values output
                    # first loop -- [(0*5)+1]:(0*5)+5] -- [1:5]
                    # second loop -- [(1*5)+1:(1*5)+5] -- [6:10]
                    output_box = tf.cast(output[pos[0]][pos[1]][pos[2]][(j * 5) + 1:(j * 5) + 5], dtype=tf.float32)
                    # print(output_box)

                    # to get the x_centre, since the grid is 7 * 7 and the image size is 224
                    # need to get the position of the object first which is from pos
                    # pos/7*224 = pos*32 (224/7=32)
                    # pos*32 + value*32 = (pos + value) * 32
                    x_centre = (tf.cast(pos[1], dtype=tf.float32) + output_box[0]) * 32
                    y_centre = (tf.cast(pos[2], dtype=tf.float32) + output_box[1]) * 32
                    # print(x_centre)
                    # print(y_centre)

                    x_width, y_height = tf.math.abs(W * output_box[2]), tf.math.abs(H * output_box[3])

                    # this is taking the bounding box's width/height to get the min and max coordinates
                    # with the centre point of the bounding box
                    x_min, y_min = int(x_centre - (x_width / 2)), int(y_centre - (y_height / 2))
                    x_max, y_max = int(x_centre + (x_width / 2)), int(y_centre + (y_height / 2))

                    x_min = 0 if x_min <= 0 else x_min
                    y_min = 0 if y_min <= 0 else y_min
                    x_max = W if x_max >= W else x_max
                    y_max = H if y_max >= H else y_max

                    final_boxes.append([x_min,
                                        y_min,
                                        x_max,
                                        y_max,
                                        str([classes[tf.argmax(selected_output[..., 10:], axis=-1)[i]]])])

                    final_scores.append(selected_output[i][j * 5])
        # print(final_scores)
        # print("Final Box: ", final_boxes)
        final_boxes = np.array(final_boxes)

        object_classes = final_boxes[..., 4]
        nms_boxes = final_boxes[..., 0:4]

        # this is to remove the duplicate bounding boxes and remain the one with highest probability score
        nms_output = tf.image.non_max_suppression(
            nms_boxes,  # containing the bounding box, in order to calculate the area
            final_scores,  # containing the first and the sixth value of the label, in order to get the highest score
            max_output_size=100,  # depending on the specific task, if there is 150 classes, then set to 150
            iou_threshold=0.2,
            # using the larger bounding box to divide the duplicated box, if greater than 0.2 then discard it
            score_threshold=float('-inf')
            # this is the threshold that if the score is lower than the defined, directly discard
        )

        # print(nms_output)

        for i in nms_output:
            cv2.rectangle(img,
                          (int(final_boxes[i][0]), int(final_boxes[i][1])),
                          (int(final_boxes[i][2]), int(final_boxes[i][3])),
                          (255, 0, 0),
                          1)

            cv2.putText(img,
                        final_boxes[i][-1],
                        (int(final_boxes[i][0]), int(final_boxes[i][1]) + 15),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 255, 255),
                        1)

        return cv2.resize(img, (224, 224))

    except:
        return image_array

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image")

        with gr.Column():
            output_img = gr.Image(label="Output Image")
            detect_btn = gr.Button("Detect")
            
    detect_btn.click(fn=model_test,
                     inputs=input_img,
                     outputs=output_img)

iface.launch()