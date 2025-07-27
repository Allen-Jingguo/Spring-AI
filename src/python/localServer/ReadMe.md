1. command
   brew install python
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org