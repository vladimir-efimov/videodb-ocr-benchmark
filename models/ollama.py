import ollama
import time

from .base_model import BaseModel


class Ollama(BaseModel):

    def __init__(self, model_name: str):

        super().__init__(model_name, "")
        self.model_name = model_name

    def describe(self, frame_urls, prompt):

        base64_images = [self.encode_image(url) for url in frame_urls]

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": base64_images
                }
            ]

            start_time = time.time()

            response = ollama.chat(
                model=self.model_name,
                options={'temperature': 0.0},
                messages=messages
            )

            end_time = time.time()

            out_text = response["message"]["content"].strip()

            processing_time = end_time - start_time

            return processing_time, out_text

        except Exception as e:
            print(f"Error with Ollama VLM: {e}")
            return None



