from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import torch
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from utils.logger import setup_logger
from utils.helpers import ensure_directory
import wandb

class FineTuner:
    """
    Enhanced fine-tuning system with LoRA, QLoRA, and full fine-tuning support
    for various model architectures and tasks
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            ft_config = config.get("fine_tuning", {})
            self.enabled = ft_config.get("enabled", False)
            self.learning_rate = ft_config.get("learning_rate", 2e-5)
            self.epochs = ft_config.get("epochs", 3)
            self.batch_size = ft_config.get("batch_size", 4)
            self.max_length = ft_config.get("max_length", 512)
            self.warmup_steps = ft_config.get("warmup_steps", 100)
            self.logging_steps = ft_config.get("logging_steps", 10)
            self.save_steps = ft_config.get("save_steps", 500)
            self.eval_steps = ft_config.get("eval_steps", 500)
            
        except Exception as e:
            self.logger.warning(f"Could not load fine-tuning configuration: {e}")
            self.enabled = False
            self.learning_rate = 2e-5
            self.epochs = 3
            self.batch_size = 4
            self.max_length = 512
            self.warmup_steps = 100
            self.logging_steps = 10
            self.save_steps = 500
            self.eval_steps = 500
        
        # Setup directories
        self.output_dir = Path("models/fine_tuned")
        self.checkpoint_dir = Path("models/checkpoints")
        self.logs_dir = Path("logs/fine_tuning")
        
        for directory in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            ensure_directory(directory)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = torch.cuda.is_available()
        
        # Training history
        self.training_history = []
        
        # Model configurations for different architectures
        self.model_configs = {
            "llama": {
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05
            },
            "gpt": {
                "target_modules": ["c_attn", "c_proj", "c_fc"],
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05
            },
            "t5": {
                "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05
            }
        }
        
        self.logger.info(f"Fine-tuner initialized (enabled: {self.enabled}, device: {self.device})")

    def prepare_dataset(self, data_path: str, task_type: str = "causal_lm", 
                       validation_split: float = 0.1) -> Tuple[Dataset, Dataset]:
        """
        Prepare dataset for fine-tuning
        
        Args:
            data_path: Path to training data (JSON/JSONL)
            task_type: Type of task (causal_lm, summarization, translation)
            validation_split: Proportion of data for validation
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        try:
            self.logger.info(f"Preparing dataset from {data_path}")
            
            # Load data
            if data_path.endswith('.jsonl'):
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError("Data file must be JSON or JSONL format")
            
            if not isinstance(data, list):
                raise ValueError("Data must be a list of examples")
            
            self.logger.info(f"Loaded {len(data)} examples")
            
            # Prepare data based on task type
            if task_type == "causal_lm":
                processed_data = self._prepare_causal_lm_data(data)
            elif task_type == "summarization":
                processed_data = self._prepare_summarization_data(data)
            elif task_type == "translation":
                processed_data = self._prepare_translation_data(data)
            elif task_type == "instruction":
                processed_data = self._prepare_instruction_data(data)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Create dataset
            dataset = Dataset.from_list(processed_data)
            
            # Split into train/validation
            if validation_split > 0:
                split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset = None
            
            self.logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(eval_dataset) if eval_dataset else 0} eval")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise

    def _prepare_causal_lm_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for causal language modeling"""
        processed = []
        
        for example in data:
            if "text" in example:
                processed.append({"text": example["text"]})
            elif "input" in example and "output" in example:
                # Combine input and output
                text = f"{example['input']}\n{example['output']}"
                processed.append({"text": text})
            else:
                self.logger.warning("Skipping example without 'text' or 'input'/'output' fields")
        
        return processed

    def _prepare_summarization_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for summarization task"""
        processed = []
        
        for example in data:
            if "document" in example and "summary" in example:
                text = f"Summarize: {example['document']}\nSummary: {example['summary']}"
                processed.append({"text": text})
            elif "input" in example and "output" in example:
                text = f"Summarize: {example['input']}\nSummary: {example['output']}"
                processed.append({"text": text})
            else:
                self.logger.warning("Skipping summarization example without required fields")
        
        return processed

    def _prepare_translation_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for translation task"""
        processed = []
        
        for example in data:
            if "source" in example and "target" in example:
                source_lang = example.get("source_lang", "en")
                target_lang = example.get("target_lang", "es")
                text = f"Translate from {source_lang} to {target_lang}: {example['source']}\nTranslation: {example['target']}"
                processed.append({"text": text})
            elif "input" in example and "output" in example:
                text = f"Translate: {example['input']}\nTranslation: {example['output']}"
                processed.append({"text": text})
            else:
                self.logger.warning("Skipping translation example without required fields")
        
        return processed

    def _prepare_instruction_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for instruction following task"""
        processed = []
        
        for example in data:
            if "instruction" in example and "response" in example:
                # Standard instruction format
                instruction = example["instruction"]
                input_text = example.get("input", "")
                response = example["response"]
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                
                processed.append({"text": text})
            elif "prompt" in example and "completion" in example:
                text = f"{example['prompt']}\n{example['completion']}"
                processed.append({"text": text})
            else:
                self.logger.warning("Skipping instruction example without required fields")
        
        return processed

    def fine_tune(self, model_name: str, dataset_path: str, output_name: str,
                 method: str = "lora", task_type: str = "causal_lm",
                 custom_config: Dict = None) -> Dict[str, Any]:
        """
        Fine-tune a model using specified method
        
        Args:
            model_name: Base model name or path
            dataset_path: Path to training dataset
            output_name: Name for the fine-tuned model
            method: Fine-tuning method (lora, qlora, full)
            task_type: Type of task
            custom_config: Custom training configuration
            
        Returns:
            Training results and metrics
        """
        if not self.enabled:
            raise RuntimeError("Fine-tuning is disabled in configuration")
        
        try:
            self.logger.info(f"Starting fine-tuning: {model_name} -> {output_name} using {method}")
            
            # Prepare dataset
            train_dataset, eval_dataset = self.prepare_dataset(dataset_path, task_type)
            
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer(model_name, method)
            
            # Prepare model for training
            if method in ["lora", "qlora"]:
                model = self._setup_lora_model(model, model_name, method)
            
            # Tokenize dataset
            train_dataset = self._tokenize_dataset(train_dataset, tokenizer)
            if eval_dataset:
                eval_dataset = self._tokenize_dataset(eval_dataset, tokenizer)
            
            # Setup training arguments
            training_args = self._setup_training_args(output_name, custom_config)
            
            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM
                return_tensors="pt",
                pad_to_multiple_of=8
            )
            
            # Setup trainer
            trainer = self._setup_trainer(
                model, tokenizer, train_dataset, eval_dataset,
                training_args, data_collator
            )
            
            # Start training
            training_start = datetime.now()
            
            self.logger.info("Starting training...")
            train_result = trainer.train()
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Save model
            self.logger.info(f"Saving model to {training_args.output_dir}")
            trainer.save_model()
            
            # Save tokenizer
            tokenizer.save_pretrained(training_args.output_dir)
            
            # Evaluate model if eval dataset exists
            eval_results = {}
            if eval_dataset:
                self.logger.info("Evaluating model...")
                eval_results = trainer.evaluate()
            
            # Compile results
            results = {
                "model_name": model_name,
                "output_name": output_name,
                "method": method,
                "task_type": task_type,
                "training_duration": training_duration,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset else 0,
                "train_loss": train_result.training_loss,
                "eval_loss": eval_results.get("eval_loss", None),
                "perplexity": np.exp(eval_results.get("eval_loss", float("inf"))),
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "output_dir": str(training_args.output_dir),
                "timestamp": training_end.isoformat()
            }
            
            # Store in history
            self.training_history.append(results)
            
            # Save training metadata
            self._save_training_metadata(results, training_args.output_dir)
            
            self.logger.info(f"Fine-tuning completed: {output_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise

    def _load_model_and_tokenizer(self, model_name: str, method: str) -> Tuple[Any, Any]:
        """Load model and tokenizer with appropriate configuration"""
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Model loading configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            # Special configuration for QLoRA
            if method == "qlora":
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["torch_dtype"] = torch.float16
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Prepare model for k-bit training if using quantization
            if method == "qlora":
                model = prepare_model_for_kbit_training(model)
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _setup_lora_model(self, model: Any, model_name: str, method: str) -> Any:
        """Setup LoRA configuration for the model"""
        try:
            # Detect model architecture
            model_type = self._detect_model_type(model_name)
            config = self.model_configs.get(model_type, self.model_configs["llama"])
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["target_modules"],
                bias="none",
                modules_to_save=None,
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            self._print_trainable_parameters(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error setting up LoRA: {e}")
            raise

    def _detect_model_type(self, model_name: str) -> str:
        """Detect model architecture type"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return "llama"
        elif "gpt" in model_name_lower:
            return "gpt"
        elif "t5" in model_name_lower:
            return "t5"
        else:
            self.logger.warning(f"Unknown model type for {model_name}, using llama config")
            return "llama"

    def _print_trainable_parameters(self, model):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_param
        
        self.logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {trainable_percent:.2f}%"
        )

    def _tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Tokenize dataset for training"""
        try:
            def tokenize_function(examples):
                # Tokenize the texts
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=self.max_length,
                    return_overflowing_tokens=False,
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset"
            )
            
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"Error tokenizing dataset: {e}")
            raise

    def _setup_training_args(self, output_name: str, custom_config: Dict = None) -> TrainingArguments:
        """Setup training arguments"""
        try:
            output_dir = self.output_dir / output_name
            
            # Base configuration
            args = {
                "output_dir": str(output_dir),
                "overwrite_output_dir": True,
                "num_train_epochs": self.epochs,
                "per_device_train_batch_size": self.batch_size,
                "per_device_eval_batch_size": self.batch_size,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "learning_rate": self.learning_rate,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0,
                "warmup_steps": self.warmup_steps,
                "logging_steps": self.logging_steps,
                "save_steps": self.save_steps,
                "eval_steps": self.eval_steps,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "save_total_limit": 3,
                "fp16": self.use_fp16,
                "dataloader_drop_last": False,
                "run_name": output_name,
                "report_to": None,  # Disable wandb by default
                "remove_unused_columns": False,
            }
            
            # Apply custom configuration
            if custom_config:
                args.update(custom_config)
            
            return TrainingArguments(**args)
            
        except Exception as e:
            self.logger.error(f"Error setting up training arguments: {e}")
            raise

    def _setup_trainer(self, model, tokenizer, train_dataset, eval_dataset,
                      training_args, data_collator) -> Trainer:
        """Setup Trainer with callbacks"""
        try:
            callbacks = []
            
            # Early stopping
            if eval_dataset:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=callbacks,
            )
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            raise

    def _save_training_metadata(self, results: Dict, output_dir: str):
        """Save training metadata and configuration"""
        try:
            metadata = {
                "training_results": results,
                "model_config": {
                    "base_model": results["model_name"],
                    "fine_tuning_method": results["method"],
                    "task_type": results["task_type"]
                },
                "training_config": {
                    "learning_rate": results["learning_rate"],
                    "epochs": results["epochs"],
                    "batch_size": results["batch_size"]
                },
                "dataset_info": {
                    "train_samples": results["train_samples"],
                    "eval_samples": results["eval_samples"]
                }
            }
            
            metadata_path = Path(output_dir) / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Training metadata saved to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving training metadata: {e}")

    def load_fine_tuned_model(self, model_path: str):
        """
        Load a fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model
            
        Returns:
            Loaded model and tokenizer
        """
        try:
            self.logger.info(f"Loading fine-tuned model from {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            self.logger.info("Fine-tuned model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def evaluate_fine_tuned_model(self, model_path: str, test_dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model on test data
        
        Args:
            model_path: Path to fine-tuned model
            test_dataset_path: Path to test dataset
            
        Returns:
            Evaluation results
        """
        try:
            self.logger.info(f"Evaluating model {model_path}")
            
            # Load model and tokenizer
            model, tokenizer = self.load_fine_tuned_model(model_path)
            
            # Prepare test dataset
            test_dataset, _ = self.prepare_dataset(test_dataset_path, validation_split=0.0)
            tokenized_test = self._tokenize_dataset(test_dataset, tokenizer)
            
            # Setup evaluation trainer
            training_args = TrainingArguments(
                output_dir="./temp_eval",
                per_device_eval_batch_size=self.batch_size,
                dataloader_drop_last=False,
                remove_unused_columns=False,
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="pt",
                pad_to_multiple_of=8
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_test,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            # Calculate additional metrics
            eval_results["perplexity"] = np.exp(eval_results["eval_loss"])
            eval_results["test_samples"] = len(test_dataset)
            eval_results["model_path"] = model_path
            eval_results["evaluation_timestamp"] = datetime.now().isoformat()
            
            self.logger.info(f"Evaluation completed. Perplexity: {eval_results['perplexity']:.2f}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise

    def generate_text(self, model_path: str, prompt: str, max_length: int = 100,
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        try:
            # Load model and tokenizer
            model, tokenizer = self.load_fine_tuned_model(model_path)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                model = model.to("cuda")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise

    def compare_models(self, model_paths: List[str], test_prompts: List[str]) -> pd.DataFrame:
        """
        Compare multiple fine-tuned models
        
        Args:
            model_paths: List of model paths to compare
            test_prompts: List of test prompts
            
        Returns:
            Comparison results DataFrame
        """
        try:
            results = []
            
            for model_path in model_paths:
                model_name = Path(model_path).name
                
                for i, prompt in enumerate(test_prompts):
                    try:
                        # Generate text
                        generated = self.generate_text(model_path, prompt)
                        
                        # Calculate basic metrics
                        response_length = len(generated.split())
                        
                        results.append({
                            "model": model_name,
                            "prompt_id": i,
                            "prompt": prompt[:50] + "...",
                            "generated_text": generated[:100] + "...",
                            "response_length": response_length,
                            "model_path": model_path
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error with model {model_name} on prompt {i}: {e}")
                        continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            raise

    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """Get training history"""
        return self.training_history[-limit:] if limit else self.training_history

    def list_fine_tuned_models(self) -> List[Dict]:
        """List all fine-tuned models"""
        try:
            models = []
            
            for model_dir in self.output_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "training_metadata.json"
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            models.append({
                                "name": model_dir.name,
                                "path": str(model_dir),
                                "base_model": metadata.get("model_config", {}).get("base_model"),
                                "method": metadata.get("model_config", {}).get("fine_tuning_method"),
                                "task_type": metadata.get("model_config", {}).get("task_type"),
                                "train_loss": metadata.get("training_results", {}).get("train_loss"),
                                "eval_loss": metadata.get("training_results", {}).get("eval_loss"),
                                "timestamp": metadata.get("training_results", {}).get("timestamp")
                            })
                        except Exception as e:
                            self.logger.debug(f"Error reading metadata for {model_dir}: {e}")
                            # Add basic info without metadata
                            models.append({
                                "name": model_dir.name,
                                "path": str(model_dir),
                                "base_model": "unknown",
                                "method": "unknown",
                                "task_type": "unknown"
                            })
            
            return sorted(models, key=lambda x: x.get("timestamp", ""), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    def delete_fine_tuned_model(self, model_name: str) -> bool:
        """
        Delete a fine-tuned model
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            Success status
        """
        try:
            model_path = self.output_dir / model_name
            
            if not model_path.exists():
                self.logger.warning(f"Model {model_name} not found")
                return False
            
            # Remove directory and all contents
            import shutil
            shutil.rmtree(model_path)
            
            self.logger.info(f"Deleted model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get fine-tuning system status"""
        try:
            # Check GPU availability
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    "memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                }
            else:
                gpu_info = {"available": False}
            
            return {
                "enabled": self.enabled,
                "device": str(self.device),
                "fp16_enabled": self.use_fp16,
                "gpu_info": gpu_info,
                "output_directory": str(self.output_dir),
                "training_history_count": len(self.training_history),
                "available_models": len(self.list_fine_tuned_models()),
                "supported_methods": ["lora", "qlora", "full"],
                "supported_tasks": ["causal_lm", "summarization", "translation", "instruction"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}