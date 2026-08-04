from .base_task import BaseTask

from typing import List, Dict, Any
from utils import calculate_num_occurrences
from tqdm import tqdm
import videodb


class OCR(BaseTask):
    def __init__(self, prompts: dict = {}):
        super().__init__(prompts)
        self.num_frames_per_call()

    def get_scenes(self, video: videodb.video = None) -> List[Any]:
        try:
            extracted_scenes = video.extract_scenes(
                extraction_type=videodb.SceneExtractionType.time_based,
                extraction_config={"time": 30, "select_frames": ["first"]},
            )

            video_scenes = extracted_scenes.scenes

        except Exception as e:
            scene_collections = video.list_scene_collection()

            # get the scene collection with the desired configuration
            for sc in scene_collections:
                if (
                    sc["config"]["extraction_type"] == "time"
                    and sc["config"]["time"] == "30"
                    and sc["config"]["select_frames"] == ["first"]
                ):
                    print(
                        f"Found existing scene collection: {sc['scene_collection_id']}"
                    )
                    video_scenes = video.get_scene_collection(
                        sc["scene_collection_id"]
                    ).scenes
                    return video_scenes

        return video_scenes

    def run(self, model_name: str, video_scenes: List[Any], video_id: str) -> Dict:
        outputs = []

        model = self.get_model(model_name)

        with tqdm(total=len(video_scenes), desc=f"Processing scenes for video {video_id}", unit="scene") as pbar:
            for scene in video_scenes:
                frame_urls = []
                # Iterate through each frame in the scene
                for frame in scene.frames:
                    frame_urls.append(frame.url)

                processing_time, out = model.describe(frame_urls, self.get_prompt(model_name))

                outputs.append(
                    {
                        "video_id": video_id,
                        "scene_start_time": scene.start,
                        "scene_end_time": scene.end,
                        "processing_time": processing_time,
                        "image": frame_urls,
                        "model_output": out,
                    }
                )
                pbar.update(1)

        return outputs

    def num_frames_per_call(self) -> int:
        """Number of frames to process per API call"""
        self.num_frames = 1

    def evaluate(
        self, video_predictions: Dict = None, video_ground_truth: Dict = None
    ) -> List[Dict]:

        results = []
        # note: poor logic from original code
        # if scene_pred contains additional frame, further comparison may be broken
        for scene_pred, scene_ground_truth in zip(
            video_predictions, video_ground_truth
        ):
            if (
                scene_pred["scene_start_time"] == scene_ground_truth["start"]
                and scene_pred["scene_end_time"] == scene_ground_truth["end"]
            ):
                model_output = scene_pred["model_output"]
                ground_truth = scene_ground_truth["ocr_text"]
                num_occur, num_key_words = calculate_num_occurrences(ground_truth, model_output)

                results.append(
                    {
                        "video_id": scene_pred["video_id"],
                        "scene_start": scene_pred["scene_start_time"],
                        "Scene_end": scene_pred["scene_end_time"],
                        "image": scene_pred["image"],
                        "ground_truth": ground_truth,
                        "num_occurrences": num_occur,
                        "num_key_words": num_key_words,
                        "processing_time": scene_pred["processing_time"],
                    }
                )

        return results
