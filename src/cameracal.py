import argparse
import cv2
import numpy as np
import time
import re
import yaml

OUT_FNAME = 'intrinsics_out.yaml'

def capture(frame: np.ndarray):
    pass

def main():
    # cmd arguments
    parser = argparse.ArgumentParser(
        prog='CameraCal',
        description='Generate intrinsic camera parameters.')
    
    parser.add_argument('src', type=str, help='filename of video source (/dev/...)')
    parser.add_argument('-d', '--duration', type=str, required=True, dest='duration', help='capture duration (MM:SS)')
    parser.add_argument('-f', '--freq', type=float, dest='freq', default=-1, help='capture frequency (every nth), omit for continuous capture')
    parser.add_argument('-w', '--width', type=int, required=True, dest='width', help='checkerboard width')
    parser.add_argument('-l', '--len', type=int, required=True, dest='len', help='checkerboard length')
    parser.add_argument('-q', '--quadsize', type=float, required=True, dest='quadSize', help='quad (square) size')

    args = parser.parse_args()

    # create points of corners in object space
    checkerboard = np.zeros((args.len, args.width, 3), np.float32)
    checkerboard[:, :, 0:2] = np.mgrid[0:args.width, 0:args.len].T * args.quadSize
    checkerboard = checkerboard.reshape((args.width * args.len, 3))

    # create calibration arrays
    objectPoints = []
    imagePoints = []

    if args.freq > 0:
        print('using continuous capture')


    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f'cannot open camera \'{args.src}\'. exiting...')
        exit()

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'using camera \'{args.src}\' ({cap_width}x{cap_height})')

    # capture loop
    match = re.search('^([0-9]+):([0-9]+)', args.duration)
    if match is None:
        print('duration provided in wrong format \'{args.duration}\'. expected MM:SS. exiting...')
        exit()

    endTime = time.time() + float(match.group(1)) * 60 + float(match.group(2))
    while time.time() < endTime:
        ret, frame = cap.read()
        
        if not ret:
            print('cannot receive frame. exiting...')
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaleFactor = 4
        grayDownscale = cv2.resize(
            gray, 
            None, 
            fx=1/scaleFactor, 
            fy=1/scaleFactor, 
            interpolation=cv2.INTER_LINEAR)
        
        ret, corners = cv2.findChessboardCorners(grayDownscale, (args.width, args.len), None)
        
        if ret:
            corners *= scaleFactor
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            objectPoints.append(checkerboard)
            imagePoints.append(corners)
            
            cv2.imshow('capture', cv2.drawChessboardCorners(gray, (args.width, args.len), corners, ret))
        cv2.imshow('capture', gray)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.freq > 0:
        objectPoints = [x for (i, x) in enumerate(objectPoints) if not i % args.freq]
        imagePoints = [x for (i, x) in enumerate(imagePoints) if not i % args.freq]

    # intrinsics
    if len(imagePoints) > 0:
        print(f'computing intrinsics with {len(imagePoints)} frames...')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (cap_width, cap_height), None, None)
        
        output = {
            'cameraMat': mtx.tolist(),
            'distCoeffs': dist.tolist()
        }

        print(mtx)
        print(dist)

        with open(OUT_FNAME, 'w+') as outfile:
            yaml.dump(output, outfile)
            print(f'output intrinsics written to \'{OUT_FNAME}\'')

    else:
        print('no checkerboard found. exiting...')
    
if __name__ == '__main__':
    main()
