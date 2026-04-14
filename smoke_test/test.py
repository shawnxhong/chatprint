from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def main():
    max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!
    # Get LAION dataset
    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files = {"train" :
    url}, split = "train")

    # 4bit pre quantized models we support for fast downloading + no OOMs.
    fourbit_models = [
    "unsloth/Qwen3-32B-bnb-4bit",
    "unsloth/Qwen3-14B-bnb-4bit",
    "unsloth/Qwen3-8B-bnb-4bit",
    "unsloth/Qwen3-4B-bnb-4bit",
    "unsloth/Qwen3-1.7B-bnb-4bit",
    "unsloth/Qwen3-0.6B-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
                model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none", # Supports any, but = "none" is optimized
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False, # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )

    trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = dataset,
                dataset_text_field = "text",
                max_seq_length = max_seq_length,
                dataset_num_proc = 1, # Recommended on Windows
                packing = False, # Can make training 5x faster for short sequences.
                args = SFTConfig(
                    per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 4,
                    warmup_steps = 5,
                    max_steps = 60,
                    learning_rate = 2e-4,
                    logging_steps = 1,
                    optim = "adamw_8bit",
                    weight_decay = 0.01,
                    lr_scheduler_type = "linear",
                    seed = 3407,
                    dataset_num_proc=1, # Recommended on Windows
                ),
            )

    trainer.train()

# Required on Windows: multiprocessing spawn needs this guard
if __name__ == "__main__":
    main()
