from move_sequence import process_sheet
from pose_estimation import estimate_pose
from preprocessing import stabilize, crop_time, take_screen
from utils import run_all

import argparse
import logging

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--no_prep', help='to skip the preprocessing of the videos', action='store_true')
    parser.add_argument('--stab', help='to stabilize the videos', action='store_true')
    parser.add_argument('--crop', help='to crop the videos in time', action='store_true')
    parser.add_argument('--screen', help='to take the screenshots of the videos', action='store_true')
    parser.add_argument('--pose', help='to estimate the pose in the videos', action='store_true')
    parser.add_argument('--output_video', help='to output the videos with the pose estimation', action='store_true')
    parser.add_argument('--move', help='to compute the move sequence from the videos', action='store_true')
    parser.add_argument('--gif', help='to save the move sequence as a GIF', action='store_true')
    parser.add_argument('--table_path', help='path of excel file with all the videos', default='boulder_problems.xlsx')
    parser.add_argument('--n_sheet', help='set the number of sheets in the given table', default=1)
    parser.add_argument('--path', help='path of all the videos', default='videos/')
    parser.add_argument('--n_boulders', help='set the number of boulders to stabilize', default=7)
    parser.add_argument('--vid_list', help='list of videos to stabilize', default=None)
    parser.add_argument('--normal_screens', help='to grab screenshots on the non mediapipe videos, defaults to False', action='store_true', default=False)
    parser.add_argument('--redo_screens', help='to rerun the screengrabing on all videos', action='store_true')
    parser.add_argument('--redo_moves', help='to rerun the move sequence computations on all videos', action='store_true')
    parser.add_argument('--verbose', '-v', help='set the verbose level, -v for infos and -vv for debugging', action='count', default=1)

    args = parser.parse_args()
    return args

def main():
    args = cli()
    args.verbose = 40 - (10*args.verbose) if args.verbose > 0 else 0

    logging.basicConfig(level=args.verbose, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    std = {'table_path':args.table_path, 'n_sheet':args.n_sheet}

    prep = args.stab or args.crop or args.screen
    if args.no_prep and prep:
        logging.error("Argument --no_prep contradicts other arguments.")
    elif not args.no_prep:
        logging.info("Preprocessing videos...")
        if args.stab:
            logging.info("Stabilizing videos...")
            stabilize(args)
            logging.info("Stabilization done !")
        if args.crop:
            logging.info("Cropping videos...")
            run_all(crop_time, args, **std)
            logging.info("Cropping done !")
        if args.screen:
            logging.info("Taking screenshots...")
            run_all(take_screen, args, **std)
            logging.info("Screenshots done !")
        logging.info("Preprocessing done !")

    else:
        logging.info("Preprocessing skipped !")

        if args.pose:
            logging.info("Beginning pose estimation...")
            run_all(estimate_pose, args, **std)
            logging.info("Pose estimation done !")
        else:
            logging.info("Pose estimation skipped !")

        if args.move:
            logging.info("Move sequence generation...")
            run_all(process_sheet, args, **std)
            logging.info("Move sequence generation done !")
        else:
            logging.info("Move sequence generation skipped !")

    logging.info("Finished.")

if __name__ == '__main__':
    main()