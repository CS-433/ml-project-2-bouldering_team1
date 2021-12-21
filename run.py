from .move_sequence import process_sheet
from .pose_estimation import estimate_pose
from .preprocessing import stabilize, crop_time, take_screen

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
    parser.add_argument('--path', help='path of excel file with all the videos', default='boulder_problems.xlsx')
    parser.add_argument('--normal_screens', help='to grab screenshots on the non mediapipe videos, defaults to False', action='store_true', default=False)
    parser.add_argument('--redo_screens', help='to rerun the screengrabing on all videos', action='store_true')
    parser.add_argument('--redo_moves', help='to rerun the move sequence computations on all videos', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = cli()

    prep = args.stab or args.crop or args.screen
    if no_prep and prep:
        logging.error("Argument --no_prep contradicts other arguments.")
    elif not no_prep:
        logging.info("Preprocessing videos...")
        if args.stab:
            logging.info("Stabilizing videos...")
            stabilize()
            logging.info("Stabilization done !")
        if args.crop:
            logging.info("Cropping videos...")
            run_all(func=crop_time, path=args.path)
            logging.info("Cropping done !")
        if args.screen:
            logging.info("Taking screenshots...")
            run_all(func=take_screen, path=args.path, mediapipe=(not args.normal_screens), redo=args.redo_screens)
            logging.info("Screenshots done !")
        logging.info("Preprocessing done !")

    else:
        logging.info("Preprocessing skipped !")

        if args.pose:
            logging.info("Beginning pose estimation...")
            run_all(func=estimate_pose, path=args.path, output_video=args.output_video)
            logging.info("Pose estimation done !")
        else:
            logging.info("Pose estimation skipped !")

        if args.move:
            logging.info("Move sequence generation...")
            run_all(func=process_sheet, path=args.path, redo=args.redo_moves)
            logging.info("Move sequence generation done !")
        else:
            logging.info("Move sequence generation skipped !")

    logging.info("Finished.")

if __name__ == '__main__':
    main()