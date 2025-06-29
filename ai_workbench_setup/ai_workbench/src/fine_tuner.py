from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml
from utils.logger import setup_logger
from datasets import load_dataset

class FineTuner:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model_name = self.config["models"]["llama"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.logger.info("FineTuner initialized")

    def fine_tune(self, dataset_path: str, output_dir: str):
        try:
            dataset = load_dataset("json", data_files=dataset_path)["train"]
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(self.model, lora_config)
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=self.config["fine_tuning"]["learning_rate"],
                num_train_epochs=self.config["fine_tuning"]["epochs"],
                per_device_train_batch_size=self.config["fine_tuning"]["batch_size"],
                save_strategy="epoch",
                logging_steps=10
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            trainer.train()
            model.save_pretrained(output_dir)
            self.logger.info(f"Fine-tuned model saved to {output_dir}")
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise