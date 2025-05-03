## Unfinished, still need debug. Go to /scripts/evaluate_text_model.ipynb to see the finished version. 


import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project config
from src.utils.config import PROCESSED_DATA_DIR

# Set paths
model_path = os.path.join(project_root, "models", "text_models")
test_data_path = os.path.join(PROCESSED_DATA_DIR, "test.json")
results_dir = os.path.join(project_root, "results", "text_models")
os.makedirs(results_dir, exist_ok=True)

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    # Load configuration
    config = PeftConfig.from_pretrained(model_path)
    base_model_name = config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    
    # Load tokenizer from the base model instead of the fine-tuned model
    print(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model with quantization
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
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=100):
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

def evaluate_model(model, tokenizer, test_data):
    """Evaluate the model on test data and return metrics"""
    print(f"Evaluating model on {len(test_data)} test examples...")
    
    predictions = []
    true_labels = []
    examples = []
    
    for i, example in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i}/{len(test_data)}")
            
        # Get model prediction
        instruction = example["instruction"]
        response = generate_response(model, tokenizer, instruction)
        
        # Process prediction to extract slang/no-slang classification
        # Adjust this logic based on your specific output format
        prediction = 1 if "slang" in response.lower() else 0
        
        # Get ground truth
        ground_truth = 1 if "slang" in example["output"].lower() else 0
        
        predictions.append(prediction)
        true_labels.append(ground_truth)
        
        # Save example for analysis
        examples.append({
            "instruction": instruction,
            "model_response": response,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth
        })
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    report = classification_report(true_labels, predictions)
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report": report
    }
    
    return metrics, examples

def analyze_training_history():
    """Analyze and plot training metrics"""
    history_path = os.path.join(model_path, "trainer_state.json")
    
    with open(history_path, "r") as f:
        training_history = json.load(f)
    
    # Extract loss values
    train_losses = [log["loss"] for log in training_history["log_history"] if "loss" in log]
    eval_steps = [log["step"] for log in training_history["log_history"] if "eval_loss" in log]
    eval_losses = [log["eval_loss"] for log in training_history["log_history"] if "eval_loss" in log]
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(eval_steps, eval_losses, 'o-', label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, "training_loss.png")
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")
    
    return {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "train_loss_history": train_losses,
        "eval_loss_history": eval_losses
    }

def interactive_demo(model, tokenizer):
    """Run an interactive demo of the model"""
    print("\n" + "="*50)
    print("Interactive Slang Detection Demo")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Enter text to check for slang: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        print("\nModel response:")
        print(response)
        print("-" * 50)

def main():
    # Load test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Analyze training history
    training_metrics = analyze_training_history()
    
    # Evaluate model
    metrics, examples = evaluate_model(model, tokenizer, test_data)
    
    # Print metrics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report'])
    
    # Save results
    results = {
        "metrics": metrics,
        "training_metrics": training_metrics,
        "examples": examples[:20]  # Save first 20 examples to keep file size reasonable
    }
    
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Run interactive demo
    interactive_demo(model, tokenizer)

if __name__ == "__main__":
    main() 