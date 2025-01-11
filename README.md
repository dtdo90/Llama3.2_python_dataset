## Fine-tuning LLAMA 3.2-1B on Python Dataset

This repository demonstrates the process of fine-tuning **LLAMA 3.2 1B** on a Python instruction dataset from hugging face https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca. The goal is to enhance the model's capability in generating and understanding Python code.

### Training Details
- Training Framework: The training uses the SFTTrainer from the trl (Transformer Reinforcement Learning) library.
- Parameter Optimization: QLoRA (Low-Rank Adaptation) is applied to reduce the number of parameters and improve efficiency during the fine-tuning process.


### Interactive API with chainlit
Interact with the fine-tuned model through a web API by running the command **chainlit run app.py**. This will launch an interactive interface for the model.

