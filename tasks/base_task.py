import os
from selectors import SelectSelector

import models
import videodb

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

CLOUD_RU_API_KEY = os.getenv("CLOUD_RU_API_KEY")

class BaseTask(ABC):
    def __init__(self, prompts=None):
        if prompts is None:
            prompts = {}
        self.prompts = prompts

    @abstractmethod
    def run(self, model_name: str, video_scenes: List[Any], video_id: str) -> Dict:
        """Run the task on given video scenes"""
        pass

    @abstractmethod
    def get_scenes(self, video: videodb.video = None) -> List[Any]:
        """get the video scenes"""
        pass

    @abstractmethod
    def num_frames_per_call(self) -> int:
        """Number of frames to process per API call"""
        pass

    def establish_videodb_connection(self) -> tuple[videodb.Connection, str]:
        conn = videodb.connect()
        return conn

    def get_videos(
        self,
        conn: videodb.Connection,
        collection_id: str = None,
        video_ids: List[str] = None,
        num_vids: int = None,
    ) -> List[Any]:
        if not video_ids:
            coll = conn.get_collection(collection_id)
            videos = coll.get_videos()
        else:
            videos = []

            coll = conn.get_collection(collection_id)
            for video_id in video_ids:
                try:
                    videos.append(coll.get_video(video_id))
                except Exception as e:
                    print(f"Error while loading {video_id}, Error: {e}")

        return videos[:num_vids]

    def get_model(self, model_name: str) -> Any:

        if model_name == "cloud_ru":
            return models.Openai("deepseek-ai/DeepSeek-OCR-2", CLOUD_RU_API_KEY,
                                 "https://foundation-models.api.cloud.ru/v1")

        elif model_name == "ollama":
            return models.Ollama("qwen3-vl:8b")

        else:
            raise AttributeError(f"Model '{model_name}' is not implemented.")

    def get_prompt(self, model_name: str) -> str:
        if model_name in self.prompts:
            return self.prompts[model_name]
        elif "default" in self.prompts:
           return self.prompts["default"]
        else:
           raise AttributeError(f"Configuration has no prompt for '{model_name}'. Also default prompt is missed")

    def evaluate(
        self, video_predictions: Dict = None, video_ground_truth: Dict = None
    ) -> Dict:
        """Optional evaluation method"""
        return {}
