import cv2
import numpy as np
import argparse
import yaml

import usb.core
import usb.util
import sys
import time

IN_FNAME = './intrinsics_out.yaml'

def main():
    parser = argparse.ArgumentParser(
        prog='Undistort',
        description='Check camera intrinsics.')

    parser.add_argument('srcl', type=str, help='filename of video source for left (/dev/...)')
    parser.add_argument('srcr', type=str, help='filename of video source for right (/dev/...)')
    parser.add_argument('-i', '--index', type=int, required=False, dest='flipIndex', help='index of camera to flip, if necessary (0=left, 1=right)')

    args = parser.parse_args()

    left = cv2.VideoCapture(args.srcl, cv2.CAP_V4L2)
    right = cv2.VideoCapture(args.srcr, cv2.CAP_V4L2)
    fail = False

    if not left.isOpened():
        print(f'cannot open camera \'{args.srcl}\'. exiting...')
        fail = True
    if not right.isOpened():
        print(f'cannot open camera \'{args.srcr}\'. exiting...')
        fail = True
    if fail:
        exit()

    # fix dimensions
    left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap_width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # check dimensions
    cap_width_r = int(right.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height_r = int(right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'RDIM: {cap_width_r} {cap_height_r}')

    print(f'using cameras \'{args.srcl}\' and \'{args.srcr}\' ({cap_width}x{cap_height})')

    stereo = cv2.StereoBM_create()
    # stereo.setMinDisparity(4)
    stereo.setNumDisparities(16)
    stereo.setBlockSize(15)
    # stereo.setSpeckleRange(16)
    # stereo.setSpeckleWindowSize(45)

    def nothing(x):
        pass
    
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)
    
    cv2.createTrackbar('numDisparities','disp',1,200,nothing)
    cv2.createTrackbar('blockSize','disp',5,200,nothing)
    cv2.createTrackbar('preFilterType','disp',1,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',2,100,nothing)
    cv2.createTrackbar('preFilterCap','disp',5,100,nothing)
    cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv2.createTrackbar('speckleRange','disp',0,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv2.createTrackbar('minDisparity','disp',5,25,nothing)

    with open(IN_FNAME) as infile:
        intrinsics_dict = yaml.load(infile, Loader=yaml.FullLoader)
        props = ['leftDistCoeffs', 'leftMat', 'rightDistCoeffs', 'rightMat', 'rotMat', 'tranMat']
        
        if all(prop in intrinsics_dict for prop in props):
            leftDistCoeffs = np.array(intrinsics_dict['leftDistCoeffs'], dtype=np.float32)
            leftMat = np.array(intrinsics_dict['leftMat'], dtype=np.float32)
            rightDistCoeffs = np.array(intrinsics_dict['rightDistCoeffs'], dtype=np.float32)
            rightMat = np.array(intrinsics_dict['rightMat'], dtype=np.float32)
            rotMat = np.array(intrinsics_dict['rotMat'], dtype=np.float64)
            tranMat = np.array(intrinsics_dict['tranMat'], dtype=np.float64)



            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                leftMat, leftDistCoeffs, 
                rightMat, rightDistCoeffs, 
                (cap_width, cap_height), 
                rotMat,
                tranMat,
                alpha=0)
            # print(roi1)
            # print(leftMat)
            # print(P1)
            # print(cv2.getDefaultNewCameraMatrix(leftMat))
            # print(rightMat)
            # print(P2)

            [leftMap1, leftMap2] = cv2.initUndistortRectifyMap(
                cameraMatrix=leftMat, 
                distCoeffs=leftDistCoeffs, 
                R=R1, 
                newCameraMatrix=P1,
                size=(cap_width, cap_height), 
                m1type=cv2.CV_16SC2)
            # [leftMap1, leftMap2] = cv2.initUndistortRectifyMap(
            #     cameraMatrix=leftMat, 
            #     distCoeffs=leftDistCoeffs, 
            #     R=np.identity(3),
            #     newCameraMatrix=cv2.getDefaultNewCameraMatrix(leftMat),
            #     size=(cap_width, cap_height), 
            #     m1type=cv2.CV_16SC2)
            [rightMap1, rightMap2] = cv2.initUndistortRectifyMap(rightMat, rightDistCoeffs, R2, P2, (cap_width, cap_height), m1type=cv2.CV_16SC2)
            
            while True:
                reads = (left.read(), right.read())
                success = [r[0] for r in reads]
                frames = [r[1] for r in reads]

                # frames = list(reversed(frames))

                if args.flipIndex is not None:
                    frames[args.flipIndex] = cv2.rotate(frames[args.flipIndex], cv2.ROTATE_180)
                
                if not all(success):
                    print('cannot receive frame. exiting...')
        
                scale = 0.25

                dist = np.concatenate(frames, axis=1)
                dist = cv2.resize(dist, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                undist1 = [
                    cv2.undistort(f, c, d)
                    for f, c, d in zip(frames, [leftMat, rightMat], [leftDistCoeffs, rightDistCoeffs])
                ]
                undist1 = np.concatenate(undist1, axis=1)
                undist1 = cv2.resize(undist1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                undist2 = [
                    f
                    for f in frames
                ]
                
                undist2 = [
                    cv2.remap(f, m1, m2, interpolation=cv2.INTER_LINEAR)
                    for f, m1, m2 in zip(frames, [leftMap1, rightMap1], [leftMap2, rightMap2])
                ]
                undist2 = [
                    cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    for f in undist2
                ]
                # gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in undist2]
                disp = stereo.compute(undist2[0], undist2[1])
                disp = disp / disp.max() * 3
                disp = cv2.resize(disp, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR)
                # disp = np.concatenate(undist2, axis=1)

                undist2 = np.concatenate(undist2, axis=1)
                undist2 = cv2.resize(undist2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                grid = np.concatenate((dist, undist1), axis=0)

                numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
                blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
                preFilterType = cv2.getTrackbarPos('preFilterType','disp')
                preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
                preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
                textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
                uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
                speckleRange = cv2.getTrackbarPos('speckleRange','disp')
                speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
                disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
                minDisparity = cv2.getTrackbarPos('minDisparity','disp')
                
                # Setting the updated parameters before computing disparity map
                stereo.setNumDisparities(numDisparities)
                stereo.setBlockSize(blockSize)
                stereo.setPreFilterType(preFilterType)
                stereo.setPreFilterSize(preFilterSize)
                stereo.setPreFilterCap(preFilterCap)
                stereo.setTextureThreshold(textureThreshold)
                stereo.setUniquenessRatio(uniquenessRatio)
                stereo.setSpeckleRange(speckleRange)
                stereo.setSpeckleWindowSize(speckleWindowSize)
                stereo.setDisp12MaxDiff(disp12MaxDiff)
                stereo.setMinDisparity(minDisparity)

                cv2.imshow('capture', grid)
                cv2.imshow('undist', undist2)
                cv2.imshow('disparity', disp)

                if cv2.waitKey(1) == ord('q'):
                    break
            
        else:
            print(f'\'{IN_FNAME}\' has not been populated. exiting...')

if __name__ == '__main__':
    main()

'''
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
'''