import cv2
import numpy as np
import argparse
import yaml

import usb.core
import usb.util
import sys
import time

IN_FNAME = 'intrinsics_out.yaml'

def main():
    parser = argparse.ArgumentParser(
        prog='Undistort',
        description='Check camera intrinsics.')

    parser.add_argument('src', type=str, help='filename of video source (/dev/...)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.src, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_MODE, 1)


    dev=usb.core.find(idVendor=0x2560, idProduct=0xc128) #lsusb to list attached devices in case your id's are different. 
    #print(dev) #uncomment to see the configuration tree. 
    #Follow the tree: dev[0] = configuration 1. 
    #interfaces()[2] = HID interface
    #0x06 = Interrupt OUT. (Endpoint)

    if dev is None:
        raise ValueError('Device not found')
    cfg=-1

    i = dev[0].interfaces()[2].bInterfaceNumber

    cfg = dev.get_active_configuration()
    intf = cfg[(2,0)]

    if dev.is_kernel_driver_active(i):
        try:
            reattach = True
            dev.detach_kernel_driver(i)
            #print("eh") #debug making sure it got in here. 
        except usb.core.USBError as e:
            sys.exit("Could not detach kernel driver from interface({0}): {1}".format(i, str(e)))


    print(dev) #not needed, just helpful for debug
    msg = [0] * 64
    #msg = [0xA0, 0xc1, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00] #gotta send a 64 byte message. 
    msg[0] = 0xA8 #these command sets are in the qtcam source code. 
    msg[1] = 0x1c
    msg[2] = 0x01 # 01= ext trigger. 00 = Master mode. 
    msg[3] = 0x00


    dev.write(0x6,msg,1000) #wham bam, than
    time.sleep(1)
    # dev.attach_kernel_driver(i)

    # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'UYVY'));
    # cap.set(cv2.CAP_PROP_FORMAT, -1);

    # if (!cap.open(0,cv2.CAP_V4L)) {
    #     printf("ERROR! Unable to open camera\n");
    # }
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
    # # cap.set(cv2.CAP_PROP_FPS, 60);
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.03)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6000)

    
    if not cap.isOpened():
        print(f'cannot open camera \'{args.src}\'. exiting...')
        exit()

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'using camera \'{args.src}\' ({cap_width}x{cap_height})')

    with open(IN_FNAME) as infile:
        intrinsics_dict = yaml.load(infile, Loader=yaml.FullLoader)
        print(intrinsics_dict)
        if 'cameraMat' in intrinsics_dict and 'distCoeffs' in intrinsics_dict:
            cameraMat = np.array(intrinsics_dict['cameraMat'], dtype=np.float32)
            distCoeffs = np.array(intrinsics_dict['distCoeffs'], dtype=np.float32)

            print(cameraMat)
            print(distCoeffs)
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print('cannot receive frame. exiting...')
                    break

                unFrame = cv2.undistort(frame, cameraMat, distCoeffs)
                compFrame = np.concatenate((frame, unFrame), axis=0)
                compFrame = cv2.resize(compFrame, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('capture', compFrame)

                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            print(f'\'{IN_FNAME}\' has not been populated. exiting...')

if __name__ == '__main__':
    main()