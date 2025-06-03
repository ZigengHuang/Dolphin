# Dolphin 
  Dolphin is a bimodal large language model (LLM) designed for intra-hospital patient education, integrating audio and text inputs to enhance emotional and semantic alignment in medical communication. Its architecture comprises three core components:
### Multimodal Encoder: 
Fuses verbal (text) and non-verbal (audio) cues via cross-attention and gated fusion to generate unified contextual representations.
### Interpretable Reasoning Decoder:
Performs joint emotion classification (7 categories: neutral, happiness, sadness, surprise, fear, anger, disgust) and semantic interpretation, producing structured reasoning paths (emotion analysis → semantic intent → response guidelines).
### Clinical Responder: 
Synthesizes responses by integrating decoder outputs with site-specific knowledge via a retrieval-augmented generation (RAG) framework.

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

### Multimodal Emotional and Semantic Modeling
Inputs: Supports audio (16kHz mono WEBM/WAV, ≤5 minutes, SNR ≥40dB) and text, extracting:
#### Audio features: 
CLAP-based verbal embeddings + 88-dimensional eGeMAPS acoustic features (pitch, energy, speech tempo) .
#### Text features: 
512-dimensional CLAP embeddings, fused with audio via bidirectional cross-attention .
#### Emotion Classification: 
Achieves 80.9–90.3% accuracy across 7 emotion categories, with fear (90.3%) and anger (89.5%) showing highest precision .
#### Semantic Alignment: 
84.9% of responses rated “consistent” or “extremely consistent” with patient intent (expert panel) .
### Structured Reasoning Paths for Interpretability
Three-Stage Framework:
#### Emotion Interpretation: Outputs top 3 emotions with confidence scores (e.g., “fear: 82%, sadness: 15%”) .
#### Semantic Interpretation: Identifies communicative intent (e.g., “hesitation about treatment,” “request for clarification”) .
#### Response Guidelines: Provides tailored strategies (e.g., “use reassuring tone,” “address underlying anxiety about side effects”) .
### Clinical Adaptability and Knowledge Integration
Department-Specific Customization: Supports 6 clinical departments (pediatrics, radiology, endoscopy, etc.), with RAG-driven knowledge bases (e.g., vaccination protocols, pre-procedure counseling) .
Ethical Compliance: Aligns with Declaration of Helsinki principles, ensuring data anonymization, encryption, and GDPR-compliant explainability .
### Efficient Deployment and Scalability
Lightweight Architecture: Runs on low-power hardware (GPU ≥2060 with 6GB VRAM, 8GB RAM), achieving 11.1 ± 3.3-second response latency .
User-Friendly Tools: Web-based interface (Flask) and API compatible with EMR systems, enabling seamless integration into clinical workflows .
### Rigorous Empirical Validation
Double-Blinded RCT Results:
	Patient Outcomes: 23.7% higher satisfaction, 17.2% higher treatment acceptance, and 15.3% stronger willingness to continue care .
	Educator Feedback: 84.3% willingness to continue using Dolphin, vs. 50.5% for text-based LLMs (P=1.9×10^−16) .
	Large-Scale Retrospective Analysis: Evaluated on 64,200 utterances from 16,583 cases, demonstrating cross-departmental effectiveness .

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
python dolphin.py
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
