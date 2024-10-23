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

    # create calibration arrays
    objectPoints = []
    imagePoints = [[], []]

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
    cap_width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'using cameras \'{args.srcl}\' and \'{args.srcr}\' ({cap_width}x{cap_height})')

    # capture loop
    match = re.search('^([0-9]+):([0-9]+)', args.duration)
    if match is None:
        print('duration provided in wrong format \'{args.duration}\'. expected MM:SS. exiting...')
        exit()

    endTime = time.time() + float(match.group(1)) * 60 + float(match.group(2))
    while time.time() < endTime:
        reads = (left.read(), right.read())
        success = [r[0] for r in reads]
        frames = [r[1] for r in reads]

        if args.flipIndex:
            frames[args.flipIndex] = cv2.rotate(frames[args.flipIndex], cv2.ROTATE_180)

        if not all(success):
            print('cannot receive frame. exiting...')
            break
        
        scale = 1/4
        graysFull = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        grays = [
            cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            for f in graysFull
        ]
        
        searches = [
            cv2.findChessboardCorners(f, (args.width, args.len), None)
            for f in grays
        ]
        
        success = [s[0] for s in searches]
        corners = [s[1] for s in searches]
        
        # if both cameras saw chessboard
        vis = graysFull
        if all(success):
            corners = [c / scale for c in corners]
            corners = [
                cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                for g, c in zip(graysFull, corners)
            ]
            
            objectPoints.append(checkerboard)
            for i, c in enumerate(corners):
                imagePoints[i].append(c)
            
            vis = [
                cv2.drawChessboardCorners(g, (args.width, args.len), c, s)
                for g, c, s in zip(graysFull, corners, success)
            ]
        
        vis = np.concatenate(vis, axis=1)
        vis = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('capture', vis)
        
        if cv2.waitKey(1) == ord('q'):
            break

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
            None,
            None,
            None,
            None,
            (cap_width, cap_height),
            flags=0)

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
            print(f'output intrinsics written to \'{OUT_FNAME}\'')
    else:
        print('no checkerboard found. exiting...')
        exit()

    while True:
        reads = (left.read(), right.read())
        success = [r[0] for r in reads]
        frames = [r[1] for r in reads]

        if args.flipIndex:
            frames[args.flipIndex] = cv2.rotate(frames[args.flipIndex], cv2.ROTATE_180)
        
        
        dist = np.concatenate(frames, axis=1)
        dist = cv2.resize(dist, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # r1, r2, p1, p2 = cv2.stereoRectify(cm1, dist1, cm2, dist2, (cap_width, cap_height), R, T)

        
        undist = [
            cv2.undistort(f, c, d)
            for f, c, d in zip(frames, [cm1, cm2], [dist1, dist2])
        ]
        undist = np.concatenate(undist, axis=1)
        undist = cv2.resize(undist, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        grid = np.concatenate((dist, undist), axis=0)
        cv2.imshow('capture', grid)

        if cv2.waitKey(1) == ord('q'):
            break
    
    left.release()
    right.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
