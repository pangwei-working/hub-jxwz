import subprocess
import os
from pathlib import Path

def list_ollama_models():
    try:
        # 调用 shell 命令 ollama list
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if not output:
            print("No Ollama models found.")
        else:
            print("Ollama models:")
            print(output)
    except FileNotFoundError:
        print("Ollama not found. Make sure it's installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error running 'ollama list': {e}")

    # 尝试打印默认模型目录
    default_path = Path.home() / "Library" / "Application Support" / "Ollama" / "models"
    if default_path.exists():
        print(f"\nModel files are stored in: {default_path}")
        for model_dir in default_path.iterdir():
            print(model_dir)
    else:
        print(f"\nDefault Ollama model directory does not exist yet: {default_path}")

if __name__ == "__main__":
    list_ollama_models()
