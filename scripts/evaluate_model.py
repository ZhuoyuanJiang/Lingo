import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Set paths
model_path = os.path.join(project_root, "models", "text_models")
print(f"Model path: {model_path}")

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected! Evaluation will be very slow.")

# Load model configuration
print(f"Loading configuration from {model_path}...")
config = PeftConfig.from_pretrained(model_path)
base_model_name = config.base_model_name_or_path
print(f"Base model: {base_model_name}")

# Load tokenizer from base model
print(f"Loading tokenizer from base model: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model with quantization
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

# Load adapter weights
print("Loading adapter weights...")
model = PeftModel.from_pretrained(model, model_path)
model.eval()  # Set to evaluation mode

print("Model and tokenizer loaded successfully")

# Define response generation function
def generate_response(instruction, max_new_tokens=100):
    """Generate a response from the model for a given instruction"""
    prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return assistant_response.strip()

# Test with a few examples
test_examples = [
    "Is the phrase 'throwing shade' slang?",
    "What does 'on fleek' mean?",
    "Is 'The weather is nice today' using slang?"
]

for example in test_examples:
    print(f"\nInput: {example}")
    response = generate_response(example)
    print(f"Response: {response}")
    print("-" * 50)

# Simple interactive demo
def interactive_demo():
    print("\n" + "="*50)
    print("Interactive Slang Detection Demo")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Enter text to check for slang: ")
        if user_input.lower() == 'quit':
            break
        
        print("Generating response...")
        response = generate_response(user_input)
        print(f"\nModel response:\n{response}")
        print("-" * 50)

# Run the interactive demo
interactive_demo()

