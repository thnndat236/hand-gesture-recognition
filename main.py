# https://learn.microsoft.com/en-us/windows/wsl/connect-usb
# Powershell:
# usbipd list
# usbipd bind --busid 1-7
# usbipd attach --wsl --busid 1-7
# Bash:
# lsusb
# ls -la /dev/video*

import argparse
import copy

import cv2 as cv


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)

    args = parser.parse_args()
    return args


def main():
    # Argument Parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Camera Preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:

        if cv.waitKey(10) == 27: # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        cv.imshow('frame', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
