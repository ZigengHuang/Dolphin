# Dolphin 
 Dolphin is an intelligent medical conversation system.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [API Documentation](#api-documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Features


## Installation

~~~
git clone https://github.com/ZigengHuang/Dolphin.git
cd Dolphin

# Create conda environment
conda create -n dolphin python=3.8
conda activate dolphin
# Install core dependencies
pip install -r requirements.txt
# Download pretrained models
./scripts/download_models.sh
~~~

## Configuration
Edit config in file dolphin.py:
~~~
    config = {
        'RAG_MODEL_PATH': '',
        'RAG_KNOWLEDGE_PATH': '',
        'AUDIO_MODEL_PATH': '',
        'AUDIO_SAVE_PATH': ''
    }
~~~

## Quick Start
Basic Usage:
~~~
from dolphin_core import DolphinCore

# Initialize system
dolphin = DolphinCore(config_path="./config.json")

# Process conversation
results = dolphin.process_conversation(
    audio_path="patient_01.webm",
    clinical_context="Type 2 Diabetes Follow-up"
)

# Export report
print(f"Transcription: {results['transcription']}")
print(f"Full analysis:\n{results['model_a_output']}")
print(f"Final response:\n{results['model_b_response']}")
~~~

Webdemo: 
We also provide a webdemo for visual application.
~~~
cd webdemo
python gui.py
~~~
After finishing deployment, you can try the webdemo on the local web page 127.0.0.1:5000.

Note: ffmpeg is required for this project.

## Data Preparation
RAG Knowledge Base Format(.xlsx or .csv):

| Content|
|--------|
|Knowledge 1|
|Knowledge 2|
|......|

Audio Requirements:

| Parameter | Specification | 
|--------|------------|
| Format | WEBM/WAV |
| Sample Rate | 16000 Hz |
| Channels | Mono |
| Duration | ≤5 minutes |
| SNR | ≥40 dB |
	
## API Documentation
Dolphin project utilizes ChatAnywhere API to call the foundational large language models.
~~~
def call_chatanywhere(self, input_text, context=[]):
        """
        Call ChatAnywhere API
        
        Parameters:
            input_text: Input text
            context: Conversation context
            
        Returns:
            API response text
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': '/api-key',
            }

            # Build message format
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
            messages.append({"role": "user", "content": input_text})

            data = {
                "model": "gpt-4o-mini", #model name, such as "gpt-4o", "deepseek-chat", "deepseek-v3", etc.
                "messages": messages,
                "temperature": 0.7
            }

            response = requests.post(
                'https://api.chatanywhere.tech/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            print(f"API call failed, status code: {response.status_code}")
            return None
        except Exception as e:
            print(f"Error calling ChatAnywhere API: {e}")
            return None
~~~
Before using it, please fill in your own API key or use your own method to access the foundational large language model.

## Citation
If using this work in academic research, please cite:
~~~
@article{name,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
~~~

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For research collaborations or technical inquiries:
