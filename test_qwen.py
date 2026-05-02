from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model_id = "Qwen/Qwen2-Audio-7B"
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded successfully for", model_id)
except Exception as e:
    print("Error loading processor:", e)
