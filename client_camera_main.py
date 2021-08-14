import sys
import argparse
import subprocess
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
import json

from models.vgg16 import vgg16
from models.tiny_yolo import tinyYolo
from keyFrameDetection import KeyFrameDetection
from communication import clientCommunication
from muLinUCB import muLinUCB
from yolo_utils import load_class_names, get_boxes, plot_boxes_cv2

WINDOW_NAME = 'CameraDemo'

vgg_info = { # action No. : [layer type num{1: conv, 2: fc, 3: act}, total mac{1: conv, 2: fc, 3: act}, mid_data_size, partition point]
                0: [13, 3, 24, 15346630656, 123633664, 26208256, 4818272, 0],
                1: [12, 3, 23, 15259926528, 123633664, 22996992, 102761824, 1],
                2: [11, 3, 22, 13410238464, 123633664, 19785728, 102761824, 2],
                3: [11, 3, 21, 13410238464, 123633664, 16574464, 25691488, 3],
                4: [10, 3, 20, 12485394432, 123633664, 13363200, 51381600, 4],
                5: [9, 3, 19, 10635706368, 123633664, 10151936, 51381600, 5],
                6: [9, 3, 18, 10635706368, 123633664, 8546304, 12846432, 6],
                7: [8, 3, 17, 9710862336, 123633664, 6940672, 25691496, 7],
                8: [7, 3, 16, 7861174272, 123633664, 5335040, 25691496, 8],
                9: [6, 3, 15, 6011486208, 123633664, 4532224, 25691496, 9],
                10: [6, 3, 14, 6011486208, 123633664, 3729408, 6423912, 10],
                11: [5, 3, 13, 5086642176, 123633664, 2926592, 12846440, 11],
                12: [4, 3, 12, 3236954112, 123633664, 2123776, 12846440, 12],
                13: [3, 3, 11, 1387266048, 123633664, 1320960, 12846440, 13],
                14: [3, 3, 10, 1387266048, 123633664, 919552, 3212648, 14],
                15: [2, 3, 9, 924844032, 123633664, 518144, 3212648, 15],
                16: [1, 3, 8, 462422016, 123633664, 417792, 3212648, 16],
                17: [0, 3, 7, 0, 123633664, 317440, 3212648, 17],
                18: [0, 3, 6, 0, 123633664, 217088, 3212648, 18],
                19: [0, 3, 4, 0, 123633664, 16384, 804200, 19],
                20: [0, 2, 2, 0, 20873216, 12288, 804200, 20],
                21: [0, 1, 0, 0, 4096000, 0, 132416, 21],
                22: [0, 0, 0, 0, 0, 0, 0, 22]
                }

yolo_info = {
                0: [9, 0, 22, 3537437696, 0, 28640768, 16614800, 0],
                1: [8, 0, 22, 3462677504, 0, 28640768, 88606096, 1],
                2: [8, 0, 21, 3462677504, 0, 23102976, 88606096, 2],
                3: [8, 0, 20, 3462677504, 0, 17565184, 88606096, 3],
                4: [8, 0, 19, 3462677504, 0, 14796288, 22152576, 4],
                5: [7, 0, 19, 3263316992, 0, 14796288, 44303744, 5],
                6: [7, 0, 18, 3263316992, 0, 12027392, 44303744, 6],
                7: [7, 0, 17, 3263316992, 0, 9258496, 44303744, 7],
                8: [7, 0, 16, 3263316992, 0, 7874048, 11076992, 8],
                9: [6, 0, 16, 3063956480, 0, 7874048, 22152576, 9],
                10: [6, 0, 15, 3063956480, 0, 6489600, 22152576, 10],
                11: [6, 0, 14, 3063956480, 0, 5105152, 22152576, 11],
                12: [6, 0, 13, 3063956480, 0, 4412928, 5539200, 12],
                13: [5, 0, 13, 2864595968, 0, 4412928, 11076992, 13],
                14: [5, 0, 12, 2864595968, 0, 3720704, 11076992, 14],
                15: [5, 0, 11, 2864595968, 0, 3028480, 11076992, 15],
                16: [5, 0, 10, 2864595968, 0, 2682368, 2770304, 16],
                17: [4, 0, 10, 2665235456, 0, 2682368, 5539208, 17],
                18: [4, 0, 9, 2665235456, 0, 2336256, 5539208, 18],
                19: [4, 0, 8, 2665235456, 0, 1990144, 5539208, 19],
                20: [4, 0, 7, 2665235456, 0, 1817088, 1385864, 20],
                21: [3, 0, 7, 2465874944, 0, 1817088, 2770312, 21],
                22: [3, 0, 6, 2465874944, 0, 1644032, 2770312, 22],
                23: [3, 0, 5, 2465874944, 0, 1470976, 2770312, 23],
                24: [3, 0, 4, 2465874944, 0, 1384448, 2770312, 24],
                25: [2, 0, 4, 1668432896, 0, 1384448, 5539208, 25],
                26: [2, 0, 3, 1668432896, 0, 1038336, 5539208, 26],
                27: [2, 0, 2, 1668432896, 0, 692224, 5539208, 27],
                28: [1, 0, 2, 73548800, 0, 692224, 2770312, 28],
                29: [1, 0, 1, 73548800, 0, 346112, 2770312, 29],
                30: [1, 0, 0, 73548800, 0, 0, 2770312, 30],
                31: [0, 0, 0, 0, 0, 0, 0, 31]
                }


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height',
                        default=480, type=int)
    parser.add_argument('--dnn', dest='dnn_model',
                        help='vgg, yolo',
                        default='yolo', type=str)
    parser.add_argument('--host', dest='host',
                        help='Ip address',
                        default='192.168.1.72', type=str)
    parser.add_argument('--port', dest='port',
                        help='Ip port',
                        default=8080, type=int)
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)640, height=(int)480,'
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

def prepare_image_vgg(frame):
    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    return img

def prepare_image_yolo(frame):
    min_img_size = 416
    image = cv2.resize(frame, (min_img_size, min_img_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    img = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img

def show_preds(img, label, averageTime):
    x = 10
    y = 50

    font = cv2.FONT_HERSHEY_PLAIN

    pred = '{:20s}'.format(label[1])
    cv2.putText(img, pred, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)
    y += 30
    timeShow = 'AvgTime: {:.4f}'.format(averageTime)
    cv2.putText(img, timeShow, (x, y), font, 2, (0, 0, 240), 2, cv2.LINE_AA)

    return img

def getVggLabelDic(class_file):
    with open(class_file, "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}
    return labels

def decodePrediction_vgg(res, labels):
    res = torch.autograd.Variable(res)
    label_index = torch.argmax(res).item()
    return labels[label_index]

def getActualDelay(action, model, preprocessed_image, totallayerNo, communication):
    if action == totallayerNo - 1: # local mobile process
        prediction = model(preprocessed_image.cuda())
        return 0, prediction.item()
    else:
        intermediate_output = model(preprocessed_image.cuda(), server=False, partition=action)

    data_to_server = [action, intermediate_output.data]
    del intermediate_output

    start_time = time.time()
    communication.send_msg(data_to_server)

    result = communication.receive_msg()

    communication.close_channel()
    end_time = time.time()

    return end_time - start_time,  result

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
        frontEndDelay = load_obj('models/vgg16FrontEndDelay')
        labels = getVggLabelDic('models/imagenet_class_index.json')
        partitionInfo = vgg_info
    else:
        model = tinyYolo()
        model.eval()
        frontEndDelay = np.load('models/estimation_yolotiny.npy')
        labels = load_class_names('models/voc.names')
        partitionInfo = yolo_info

    model.cuda()
    Action_num = len(partitionInfo)

    muLinUCB = muLinUCB(mu=0.25, layerInfo=partitionInfo, frontDelay=frontEndDelay)
    communication = clientCommunication(args.host, args.port)

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else:  # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width, args.image_height)
        # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)

    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN

    total_time = 0
    total_frame_num = 0
    currentFrameNum = 0
    keyflag = False
    KeyFrame = KeyFrameDetection(threshold=0.8)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read()  # grab the next image frame from camera

        if args.dnn_model == 'vgg':
            preprocessed_image = prepare_image_vgg(img)
        else:
            preprocessed_image = prepare_image_yolo(img)

        # doubling trick is here.
        currentFrameNum = currentFrameNum + 1
        if muLinUCB.updateDoublingTrickFrameNum(currentFrameNum):
            currentFrameNum = 0

        # key frame detection
        if total_frame_num == 0:
            keyflag = False
            old_frame = np.copy(img)
        else:
            keyflag = KeyFrame.compare_images(old_frame, img)
            old_frame = np.copy(img)

        # print('keyflag', keyflag)

        partitionPoint = muLinUCB.getEstimationAction(keyflag, currentFrameNum)
        # print('partitionPoint', partitionPoint)

        end2endtime_start = time.time()
        actual_delay, res = getActualDelay(partitionPoint, model, preprocessed_image, Action_num, communication)

        end2endtime_end = time.time()

        total_frame_num = total_frame_num + 1
        total_time = total_time + (end2endtime_end - end2endtime_start)
        average_time = total_time/total_frame_num

        # update A and b
        muLinUCB.updateA_b(partitionPoint, actual_delay)

        # print results on  the screen
        if args.dnn_model == 'vgg':
            label = decodePrediction_vgg(res, labels)
            img = show_preds(img, label, average_time)
        else:
            boxes = get_boxes(res, model, conf_thresh=0.5, nms_thresh=0.5)
            img = plot_boxes_cv2(img, boxes, class_names=labels)

        if show_help:
            cv2.putText(img, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(img, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
        

    cap.release()
    cv2.destroyAllWindows()
