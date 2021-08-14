import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class KeyFrameDetection:
    def __init__(self, threshold):
        self.threshold = threshold

    def compare_images(self, imgA, imgB):
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        s = ssim(imgA, imgB)
        # print('ssim:', s)
        if s < self.threshold:
            KeyFlag = True
            return KeyFlag
        KeyFlag = False
        return KeyFlag


if __name__ == "__main__":
    keyFrameDetection = KeyFrameDetection(0.6)
    imgA = cv2.imread("Golden_Retriever_Hund_Dog.jpg")
    imgB = cv2.imread("Golden_Retriever_Hund_Dog.jpg")

    flag = keyFrameDetection.compare_images(imgA, imgB)
    print(flag)
