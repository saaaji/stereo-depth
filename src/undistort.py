import cv2
import numpy as np
import argparse
import yaml

IN_FNAME = 'intrinsics_out.yaml'

def main():
    parser = argparse.ArgumentParser(
        prog='Undistort',
        description='Check camera intrinsics.')

    parser.add_argument('src', type=str, help='filename of video source (/dev/...)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.src)
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