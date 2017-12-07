import argparse
import os

import cv2
import numpy as np
import sys
import tensorflow as tf
import time

from align.detect_face import create_mtcnn, detect_face


def current_milli_time():
    return int(round(time.time() * 1000))


def process(frame):
    pass


def main(args):
    output_base_dir = os.path.expanduser(args.output_base_dir)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    cap = cv2.VideoCapture(args.rtsp_url)
    # try again
    if not cap.isOpened():
        cap.open(args.rtsp_url)

    if not cap.isOpened():
        print("ERROR: cannot open {}".format(args.rtsp_url))
        sys.exit(-1)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # threshold = [0.7, 0.8, 0.8]
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.per_process_gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, args.mtcnn_model_base_dir)

            cap = cv2.VideoCapture(args.rtsp_url)
            while True:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (int(frame.width / args.resize_scale), int(frame.height / args.resize_scale))
                    )
                    filename_base = current_milli_time()
                    bounding_boxes, points = detect_face(frame_rgb, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        det_arr = []
                        img_size = np.asarray(frame_rgb.shape)[0:2]
                        if nrof_faces > 1:
                            if args.detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack(
                                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 -
                                     img_center[0]])
                                offset_dist_squared = np.sum(
                                    np.power(offsets, 2.0), 0)
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.cropped_img_margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.cropped_img_margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.cropped_img_margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.cropped_img_margin / 2, img_size[0])
                            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                            if args.detect_multiple_faces:
                                output_filename_n = os.path.join(
                                    output_base_dir,
                                    "{}_{}{}".format(filename_base, i, args.img_ext)
                                )
                            else:
                                output_filename_n = os.path.join(
                                    output_base_dir,
                                    "{}{}".format(filename_base, args.img_ext)
                                )
                            print('[{}] saving {}'.format(current_milli_time(), output_filename_n))
                            cv2.imwrite(output_filename_n, cropped)
                    else:
                        print('[{}] no face...'.format(current_milli_time()))
                else:
                    print("[{}] ERROR: failed to read frame".format(current_milli_time()))


def parse_argument(argv):
    # rtsp://admin:admin123@10.168.1.38:554/h264/ch1/main/av_stream
    # rtsp://admin:admin123@10.168.1.37:554/h264/ch1/main/av_stream
    # rtsp://admin:admin123@10.168.1.34:554/h264/ch1/main/av_stream
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mtcnn_model_base_dir',
        type=str,
        help='mtcnn model directory for pnet/rnet/onet'
    )
    parser.add_argument(
        '--rtsp_url',
        type=str,
        help='rtsp url'
    )

    parser.add_argument(
        '--output_base_dir',
        type=str,
        help='capture face output base directory path'
    )

    parser.add_argument(
        '--detect_multiple_faces',
        help='enable detecting multiple faces',
        action='store_true'
    )

    parser.add_argument(
        '--per_process_gpu_memory_fraction',
        type=float,
        help='gpu memory fraction if running on cuda',
        default=0.2
    )

    parser.add_argument(
        '--img_ext',
        type=str,
        help='save face image file extension',
        default=".png"
    )

    parser.add_argument(
        '--cropped_img_margin',
        type=int,
        help='cropped face margin',
        default=100
    )

    parser.add_argument(
        '--resize_scale',
        type=float,
        help='scale down to speed up processing speed',
        default=4.0
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_argument(sys.argv[1:]))
