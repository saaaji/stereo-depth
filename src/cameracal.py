import argparse
import cv2
import numpy as np
import time
import re
import yaml

# metal plate: 0.0238125 m

OUT_FNAME = 'intrinsics_out.yaml'

def capture(frame: np.ndarray):
    pass

def main():
    # cmd arguments
    parser = argparse.ArgumentParser(
        prog='CameraCal',
        description='Generate intrinsic camera parameters.')
    
    parser.add_argument('srcl', type=str, help='filename of video source for left (/dev/...)')
    parser.add_argument('srcr', type=str, help='filename of video source for right (/dev/...)')
    parser.add_argument('-i', '--index', type=int, required=False, dest='flipIndex', help='index of camera to flip, if necessary (0=left, 1=right)')
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

    if args.freq > 0:
        print('using continuous capture')

    left = cv2.VideoCapture(args.srcl)
    right = cv2.VideoCapture(args.srcr)
    fail = False

    if not left.isOpened():
        print(f'cannot open camera \'{args.srcl}\'. exiting...')
        fail = True
    if not right.isOpened():
        print(f'cannot open camera \'{args.srcr}\'. exiting...')
    if fail:
        exit()

    # assume resolution is the same for both
    left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap_width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'using cameras \'{args.srcl}\' and \'{args.srcr}\' ({cap_width}x{cap_height})')

    # capture loop
    match = re.search('^([0-9]+):([0-9]+)', args.duration)
    if match is None:
        print('duration provided in wrong format \'{args.duration}\'. expected MM:SS. exiting...')
        exit()

    endTime = time.time() + float(match.group(1)) * 60 + float(match.group(2))
    scale = 0.8
    calib = []

    for (cam, name) in zip([left, right], ['left', 'right']):
        imagePoints = []
        objectPoints = []
        nCaps = 0
        while True:
            (ret, image) = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            (ret, corners) = cv2.findChessboardCorners(gray, (args.width, args.len), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)

            # corners /= scale
            vis = gray
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                vis = cv2.drawChessboardCorners(gray, (args.width, args.len), corners, ret)

            vis = cv2.resize(vis, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

            key = cv2.waitKey(1)
            if key == ord('p') and ret:
                nCaps += 1
                print(f'capturing... {nCaps}')            
                objectPoints.append(checkerboard)
                imagePoints.append(corners)
            elif key == ord('q'):
                break
            
            cv2.imshow('capture', vis)

        
        if len(objectPoints) > 0:
            print(f'computing intrinsics with {len(objectPoints)} frames...')
            ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (cap_width, cap_height), None, None)

            output = {
                'mat': mat.tolist(),
                'dist': dist.tolist(),
                # 'rvecs': rvecs.tolist(),
                # 'tvecs': tvecs.tolist(),
            }

            calib.append({
                'mat': mat,
                'dist': dist,
            })

            with open(f'{name}.yaml', 'w+') as outfile:
                yaml.dump(output, outfile)
                print(f'output intrinsics written to \'{name}.yaml\' (ERROR: {ret})')
        else:
            print(f'no checkerboard found for {name}')


    # create calibration arrays
    objectPoints = []
    imagePoints = [[], []]
    nCaps = 0
    while True:
        reads = (left.read(), right.read())
        success = [r[0] for r in reads]
        frames = [r[1] for r in reads]

        if args.flipIndex:
            frames[args.flipIndex] = cv2.rotate(frames[args.flipIndex], cv2.ROTATE_180)

        if not all(success):
            print('cannot receive frame. exiting...')
            break
        
        graysFull = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        # grays = [
        #     cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #     for f in frames
        # ]

        grays = [f for f in frames]
        grays = [cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) for g in grays]
        # graysFull = [cv2.inRange(g, 0, 100) for g in graysFull]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # graysFull = [cv2.dilate(g, kernel, iterations=1) for g in graysFull]
        # graysFull = [cv2.medianBlur(g, 1) for g in graysFull]
        # min = np.array([0, 0, 0])
        # max = np.array([179, 50, 255])
        # # masked = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # masks = [cv2.inRange(g, min, max) for g in graysFull]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        # graysFull = [cv2.dilate(m, kernel, iterations=5) for m in masks]
        # graysFull = [255 - cv2.bitwise_and(g, m) for g, m in zip(graysFull, masks)]
        
        searches = [
            cv2.findChessboardCorners(g, (args.width, args.len), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
            for g in grays
        ]
        
        success = [s[0] for s in searches]
        corners = [s[1] for s in searches]
        
        # if both cameras saw chessboard
        # vis = [graysFull[0], cv2.cvtColor(grays[0], cv2.COLOR_BGR2GRAY)]
        # if all(success):
        vis = graysFull
        
        key = cv2.waitKey(1)

        if all(success):
            # corners = [c / scale for c in corners]
            corners = [
                cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                for g, c in zip(graysFull, corners)]
            vis = [
                cv2.drawChessboardCorners(f, (args.width, args.len), c, s)
                for f, c, s in zip(graysFull, corners, success)]

        if key == ord('p') and all(success):
            nCaps += 1
            print(f'capturing... {nCaps}')            
            objectPoints.append(checkerboard)
            for i, c in enumerate(corners):
                imagePoints[i].append(c)
        elif key == ord('q'):
            break

        vis = np.concatenate(vis, axis=1)
        vis = cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('capture', vis)

    if args.freq > 0:
        objectPoints = [x for (i, x) in enumerate(objectPoints) if not i % args.freq]
        for i, ip in enumerate(imagePoints):
            imagePoints[i] = [x for (i, x) in enumerate(ip) if not i % args.freq]

    # intrinsics
    if len(objectPoints) > 0:
        print(f'computing intrinsics with {len(objectPoints)} frames...')
        ret, cm1, dist1, cm2, dist2, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints, 
            imagePoints[0], 
            imagePoints[1],
            calib[0].get('mat'),
            calib[0].get('dist'),
            calib[1].get('mat'),
            calib[1].get('dist'),
            (cap_width, cap_height),
            criteria=(cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            flags=0,
            R=None,
            T=None)

        output = {
            'leftMat': cm1.tolist(),
            'rightMat': cm2.tolist(),
            'leftDistCoeffs': dist1.tolist(),
            'rightDistCoeffs': dist2.tolist(),
            'rotMat': R.tolist(),
            'tranMat': T.tolist(),
        }

        with open(OUT_FNAME, 'w+') as outfile:
            yaml.dump(output, outfile)
            print(f'output intrinsics written to \'{OUT_FNAME}\' (ERROR: {ret})')
    else:
        print('no checkerboard found. exiting...')
        exit()
    
    left.release()
    right.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
