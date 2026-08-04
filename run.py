import argparse
import json
import os
import videodb

from datetime import datetime
from utils import load_yaml_config
from tqdm import tqdm

from utils import create_directories, setup_logging, make_summary
from tasks import get_task


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Vision Language Models Benchmarking", add_help=False
    )

    parser.add_argument(
        "--model",
        default="cloud_ru",
        type=str,
        nargs="+",
        choices=[
            "all",
            "cloud_ru",
            "ollama"
        ],
    )

    parser.add_argument("--num_vids", default=100, type=int)

    return parser


def main(args):
    # get the task name and config
    task, config = get_task("ocr")

    # setup directories to store the result
    args.results_dir = config.RESULTS_DIR
    args.evaluation_dir = config.EVALUATION_DIR

    args.save_paths = create_directories(args)

    current_run = f"ocr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # get prompt
    yaml_file = load_yaml_config("prompts.yaml")

    # get the task processor
    processor = task(yaml_file)

    # establish VideoDB connection and get the data
    conn = processor.establish_videodb_connection()

    # get videos
    try:
        if config.VIDEO_IDS:
            videos = processor.get_videos(
                conn=conn,
                video_ids=config.VIDEO_IDS.values(),
                collection_id=config.COLLECTION_ID,
                num_vids=args.num_vids,
            )
        else:
            videos = processor.get_videos(
                conn=conn, collection_id=config.COLLECTION_ID, num_vids=args.num_vids
            )
    except videodb.exceptions.AuthenticationError:
        print(
            "Please make sure VIDEO_DB_API_KEY is set in your .env like VIDEO_DB_API_KEY=sk-****-****"
        )
        return
    except Exception as e:
        print(f"Run failed due to {e}")
        return

    # iterate through all the models
    for path in args.save_paths:
        model_name = os.path.basename(path)

        logger, current_run_dir = setup_logging(path, current_run)

        logger.info(
            f"################################ Running {model_name} Model on OCR Prompt ################################\n"
        )

        # iterate through all the videos
        
        for video in tqdm(videos, desc="Processing videos", unit="video"):
          
            video_scenes = processor.get_scenes(video)
            outputs = processor.run(model_name, video_scenes, video.id)

            if outputs is not None:
                json_file = os.path.join(current_run_dir, f"{video.id}_output.json")
                with open(json_file, "w", encoding='utf-8') as file:
                    json.dump(outputs, file, ensure_ascii=False)
                logger.info(f"model results of {video.id} saved to {json_file}")

            else:
                logger.info(f"failed to save model results of {video.id}")

            # Evaluation

            # load ground truth
            gt_file = os.path.join(
                config.OCR_GROUND_TRUTH_DIR, f"{video.id}_ground_truth.json"
            )
            with open(gt_file, "r", encoding='utf-8') as file:
                video_ground_truth = json.load(file)

            video_result = processor.evaluate(outputs, video_ground_truth)

            # save it in evaluation directory
            os.makedirs(os.path.join(current_run_dir, "evaluations"), exist_ok=True)

            eval_json_file = os.path.join(
                current_run_dir, "evaluations", f"{video.id}.json"
            )
            with open(eval_json_file, "w", encoding='utf-8') as file:
                json.dump(video_result, file, ensure_ascii=False)

            logger.info(f"results evaluations of {video.id} saved to {eval_json_file}")

    # Evaluation summary
    summary = make_summary(current_run, args.results_dir)
    os.makedirs(args.evaluation_dir,exist_ok=True)
    summary_file = os.path.join(args.evaluation_dir,f"{current_run}.json")
    print(f"Evaluation summary:  {summary} saved to {summary_file}")
    with open(summary_file, "w") as f:
        json.dump(summary,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Vision Language Models Benchmarking", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    main(args)
