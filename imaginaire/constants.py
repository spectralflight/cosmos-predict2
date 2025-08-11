T5_MODEL_DIR = "checkpoints/google-t5/t5-11b"

COSMOS_REASON1_MODEL_DIR = "checkpoints/nvidia/Cosmos-Reason1-Private"
COSMOS_REASON1_CHECKPOINT = f"checkpoints/nvidia/Cosmos-Reason1-Private/reason1_internal_real.pt"
COSMOS_REASON1_TOKENIZER = f"checkpoints/nvidia/Cosmos-Reason1-Private/tokenizer"

def get_cosmos_predict2_model_dir(*, model_type: str, model_size: str) -> str:
    return f"checkpoints/nvidia/Cosmos-Predict2-{model_size}-{model_type}"

def get_cosmos_predict2_tokenizer(*, model_type: str, model_size: str) -> str:
    model_dir = get_cosmos_predict2_model_dir(model_type=model_type, model_size=model_size)
    return f"{model_dir}/tokenizer/tokenizer.pth"

def get_cosmos_predict2_checkpoint(*, model_type: str, model_size: str, resolution: int, fps: int) -> str:
    model_dir = get_cosmos_predict2_model_dir(model_type=model_type, model_size=model_size)
    return f"{model_dir}/model-{resolution}p-{fps}fps.pt"
