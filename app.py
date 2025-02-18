import torch
import chainlit


from utils_for_app import load_fine_tune_model, generate_ft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load fine-tuned model and tokenizer
print(f"Loading finetuned model ...")

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
lora_weights = "LLAMA32_ft_python_code.pth"
model_ft, tokenizer = load_fine_tune_model(base_model_id, lora_weights)

total_params=sum(p.numel() for p in model_ft.parameters())
trainable_params=sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)

    prompt = message.content

    response = generate_ft(
        model=model_ft, 
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.8,
        top_k=5
        )

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
