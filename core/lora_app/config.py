from dataclasses import dataclass


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "artifacts/llama32_tofu"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_EPOCHS = 1.0
FULL_EPOCHS = 10
FULL_LEARNING_RATE = 1e-4
LORA_EPOCHS = 50
LORA_LEARNING_RATE = 2e-4
DATASET_ID = "locuslab/TOFU"
DATASET_SPLIT = "train"
DEFAULT_DATASET_SUBSET = "forget01"
TOFU_SUBSETS = ("forget01", "retain99")
TEST_QUESTION = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?"
# TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TARGET_MODULES = ("q_proj", "v_proj")
APPLY_CHAT_TEMPLATE = False
SYSTEM_PROMPT = "You are a helpful assistant."
SYSTEM_PROMPT_WITH_SPECIAL_TOKENS = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
)
USER_START_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"
USER_END_TAG = "<|eot_id|>"
ASST_START_TAG = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASST_END_TAG = "<|eot_id|>"
DATE_STRING = "10 Apr 2025"


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str = MODEL_ID
    output_dir: str = OUTPUT_DIR
    dataset_id: str = DATASET_ID
    dataset_subset: str = DEFAULT_DATASET_SUBSET
    dataset_split: str = DATASET_SPLIT
    max_length: int = MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_epochs: float = DEFAULT_WARMUP_EPOCHS
    apply_chat_template: bool = APPLY_CHAT_TEMPLATE
    system_prompt: str = SYSTEM_PROMPT
    system_prompt_with_special_tokens: str = SYSTEM_PROMPT_WITH_SPECIAL_TOKENS
    user_start_tag: str = USER_START_TAG
    user_end_tag: str = USER_END_TAG
    asst_start_tag: str = ASST_START_TAG
    asst_end_tag: str = ASST_END_TAG
    date_string: str = DATE_STRING

    def run_output_dir(self, mode: str) -> str:
        return f"{self.output_dir}/{self.dataset_subset}/{mode}"
