"""
Config for OCR Benchmarking

"""
from tasks.base_task import CLOUD_RU_API_KEY

# Collection link: https://console.videodb.io/share/135ffd86-779c-43e9-b8e1-d619db325b75

COLLECTION_ID = "c-cc12f589-e23e-48f8-a351-84858311db67" # MLJava channel videos


# Data
VIDEO_IDS = {
    "ocr_for_rag": "m-z-019fcd28-a3ce-7710-b16a-ce25cdbc6566",
    "why_java_and_spring_ai": "m-z-019fd5e6-2166-7b40-924d-f1617a5e30d2",
    "spring_ai_yc_starter": "m-z-019fd644-2aac-7a30-a71a-88e761f370c3",
    "chatbot": "m-z-019fda8b-e48e-7b41-9f8e-e367ce85ccec",
}

# Save Directories
RESULTS_DIR = "_results"
EVALUATION_DIR = "_evaluation"

# Path to the ground truth text directory
OCR_GROUND_TRUTH_DIR = "ocr_ground_truths"
