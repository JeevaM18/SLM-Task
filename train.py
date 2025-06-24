import sys
import os

def main():
    print("Launching training script...")
    
    script_path = os.path.join("src", "train_model.py")
    
    if not os.path.exists(script_path):
        print(f"Training script not found at: {script_path}")
        sys.exit(1)
    
    print(f"Running: {script_path}")
    os.system(f"python {script_path}")

if __name__ == "__main__":
    main()
