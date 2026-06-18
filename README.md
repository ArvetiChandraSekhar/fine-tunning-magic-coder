# README: Fine-Tuning a Python-Specialized Code Generation Model (Magicoder)

This notebook details the process of fine-tuning a Large Language Model (LLM) to act as a **Python automation assistant**, capable of generating high-quality Python code and explicitly refusing requests for other programming languages.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Base Model](#base-model)
4.  [Fine-Tuning Methodology (QLoRA)](#fine-tuning-methodology-qlora)
5.  [Key Configurations](#key-configurations)
6.  [Fine-Tuning Steps](#fine-tuning-steps)
7.  [Model Specialization and Restrictions](#model-specialization-and-restrictions)
8.  [Usage and Inference](#usage-and-inference)
9.  [Memory Management](#memory-management)
10. [Conclusion](#conclusion)

## Introduction

This project focuses on adapting a pre-trained LLM (`Llama-2-7b-chat-hf`) using the Parameter-Efficient Fine-Tuning (PEFT) technique of Quantized Low-Rank Adaptation (QLoRA). The goal is to specialize the model for Python code generation while ensuring it strictly adheres to this domain and gracefully handles out-of-domain (non-Python) requests by refusing them.

## Dataset

**Name**: `ise-uiuc/Magicoder-OSS-Instruct-75K`

The `Magicoder-OSS-Instruct-75K` dataset is a rich collection of instruction, input, and output triples designed for code generation tasks. It's built using an execution-based approach, providing diverse coding problems and structural script traces across multiple programming languages. For this project, the dataset was meticulously filtered to include **only Python examples**, ensuring the fine-tuned model's Python-only specialization.

**Data Format**: Each example in the filtered dataset consists of a `problem` (user's code generation request) and a `solution` (the corresponding Python code).

## Base Model

**Model Name**: `meta-llama/Llama-2-7b-chat-hf`

We utilize the `Llama-2-7b-chat-hf` model as our base model. This is a 7-billion parameter language model from Meta, pre-trained for conversational AI tasks, providing a strong foundation for further specialization.

## Fine-Tuning Methodology (QLoRA)

To efficiently fine-tune such a large model, we employ **QLoRA (Quantized Low-Rank Adaptation)**. QLoRA combines LoRA (Low-Rank Adaptation) with 4-bit quantization, offering significant benefits:

*   **Memory Efficiency**: By quantizing the base model to 4-bit precision and only training small, low-rank adapters, QLoRA drastically reduces the memory footprint during training.
*   **Computational Efficiency**: Training only a small fraction of the model's parameters (the LoRA adapters) leads to faster fine-tuning compared to updating all parameters.
*   **Performance**: QLoRA often achieves performance comparable to full fine-tuning with a fraction of the resources.

## Key Configurations

*   **Quantization**: The base model is loaded in `4-bit NF4` (NormalFloat4) precision with `torch.float16` compute data type.
*   **LoRA Parameters**:
    *   `lora_r`: 64 (rank of the update matrices)
    *   `lora_alpha`: 16 (scaling factor)
    *   `lora_dropout`: 0.1
    *   `lora_target_modules`: `["q_proj", "v_proj"]` (attention query and value projection layers)
    *   `bias`: `"none"`
    *   `task_type`: `"CAUSAL_LM"`
*   **Training Parameters**:
    *   `num_train_epochs`: 3
    *   `learning_rate`: 2e-4
    *   `per_device_train_batch_size`: 1 (with gradient accumulation steps for effective batch size)
    *   `gradient_accumulation_steps`: 16
    *   `optim`: `paged_adamw_32bit`

## Fine-Tuning Steps

The fine-tuning process involves:

1.  **Dataset Loading and Filtering**: Loading `Magicoder-OSS-Instruct-75K` and filtering for Python-only examples.
2.  **Prompt Formatting**: Converting dataset examples into the Llama-2 chat template format, including a strict system message for Python specialization.
3.  **Model and Tokenizer Initialization**: Loading the `Llama-2-7b-chat-hf` base model with 4-bit quantization and its corresponding tokenizer.
4.  **PEFT Configuration**: Setting up the `LoraConfig` for QLoRA.
5.  **SFT Training**: Utilizing `trl.SFTTrainer` for supervised fine-tuning with the prepared dataset and QLoRA configuration.
6.  **Saving PEFT Adapters**: Saving the trained LoRA adapters and the tokenizer.
7.  **Memory Cleanup**: Aggressively freeing GPU memory to prepare for model merging.
8.  **Model Merging**: Merging the trained LoRA adapters with the original (quantized) base model to create a standalone, deployable fine-tuned model.
9.  **Final Model Saving**: Saving the merged model and its tokenizer.

## Model Specialization and Restrictions

The fine-tuned model has been explicitly designed and trained with the following core characteristics:

*   **Role**: Python automation assistant.
*   **Language Support**: **Only** supports and generates Python code.
*   **Refusal Mechanism**: If a user asks for code in any other language (e.g., Java, C++, JavaScript), the model is instructed to **explicitly refuse** the request and remind the user of its Python specialization boundaries.

This strict specialization ensures that the model provides consistent and reliable Python-focused assistance.

## Usage and Inference

To use the fine-tuned model for inference:

1.  **Load Base Model**: Load the original `Llama-2-7b-chat-hf` model with the same quantization configuration used during training.
2.  **Load PEFT Adapters**: Load the saved LoRA adapters on top of the base model using `PeftModel.from_pretrained()`.
3.  **Move to CUDA**: Explicitly move the combined model to a CUDA device (`.to("cuda")`).
4.  **Load Tokenizer**: Load the tokenizer associated with the fine-tuned model.
5.  **Format Prompt**: Construct your input prompt using the Llama-2 chat template, incorporating the system message about Python-only specialization.
6.  **Generate Response**: Use the model's `generate()` method to get the output.

**Note on PEFT Model Loading**: When loading a PEFT model, it's crucial to understand that the fine-tuned adapters are *not* a standalone model. They are a set of small, low-rank matrices that *modify* the behavior of the much larger base model. Therefore, the base model must always be loaded first, and then the PEFT adapters are applied on top of it. The adapters essentially augment the base model's knowledge, not replace it.

## Memory Management

Throughout the notebook, aggressive memory cleanup (using `gc.collect()` and `torch.cuda.empty_cache()`) is performed. This is vital in resource-constrained environments like Google Colab, especially when dealing with large models and quantization, to prevent Out-of-Memory (OOM) errors between different stages of the workflow (e.g., after training and before merging).

## Conclusion

This notebook demonstrates a complete workflow for creating a highly specialized Python code generation assistant using QLoRA. The resulting model is efficient, effective, and adheres strictly to its defined role, making it a valuable tool for Python automation tasks.
