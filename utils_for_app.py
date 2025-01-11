from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_fine_tune_model(base_model_id, saved_weights):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to("cpu")
    
    # Create LoRA config - make sure these parameters match your training configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        r=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
        ],
    )
    
    # Initialize PeftModel
    lora_model = PeftModel(base_model, peft_config)
    
    # Load the saved weights
    state_dict = torch.load(saved_weights,map_location=torch.device('cpu'))
    
    # Check the keys in the state dict
    print("Original state dict keys:", state_dict.keys())
    
    # Create new state dict with correct prefixes and structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'model.model.' in key:
            # If the key already has model.model., just add base_
            new_key = f"base_{key}"
        elif 'model.' in key:
            # If the key has only model., add base_model.
            new_key = f"base_{key}"
        else:
            # If the key has neither, add the full prefix
            new_key = f"base_model.model.{key}"
        
        # Print shape information for debugging
        print(f"Converting {key} -> {new_key}")
        print(f"Shape: {value.shape}")
        
        new_state_dict[new_key] = value
    
    # Load the weights with strict=False to allow partial loading
    lora_model.load_state_dict(new_state_dict, strict=False)
    
    # Set to evaluation mode
    lora_model = lora_model.eval()
    
    return lora_model, tokenizer


def generate_ft(model, prompt, tokenizer, max_new_tokens, context_size=256, temperature=0.0, top_k=1, eos_id=[128001, 128009]):
    """
    Generate text using a language model with proper dtype handling and improved sampling.
    """
    # Get model's expected dtype and device
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    
    formatted_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}"
        f"### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Encode and prepare input
    idx = tokenizer.encode(formatted_prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=model_device).unsqueeze(0)
    num_tokens = idx.shape[1]
    
    # Generation loop
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]     # conditioning context        
        with torch.no_grad():
            model.eval()            
            # Forward pass 
            outputs = model(input_ids=idx_cond,use_cache=False)
            logits = outputs.logits
        
        # Focus on last time step
        logits = logits[:, -1, :]
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=model_device, dtype=model_dtype),
                logits
            )
        
        # Apply temperature and sample
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Check for EOS
        if idx_next.item() in eos_id:
            break
            
        # Append new token
        idx = torch.cat((idx, idx_next), dim=1)
    
    # Decode generated text
    generated_ids = idx.squeeze(0)[num_tokens:]
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text

if __name__=="__main__":
    # Load fine-tuned model and tokenizer
    print(f"Loading finetuned model ...")

    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    lora_weights = "LLAMA32_ft_python_code.pth"
    model_ft, tokenizer = load_fine_tune_model(base_model_id, lora_weights)

    total_params=sum(p.numel() for p in model_ft.parameters())
    trainable_params=sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # generate
    prompt = "Program a Flask API that will convert an image to grayscale."
    print(generate_ft(model_ft, prompt, tokenizer, max_new_tokens=256))