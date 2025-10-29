# Dolphin 
  Dolphin is a bimodal large language model (LLM) designed for intra-hospital patient education, integrating audio and text inputs to enhance emotional and semantic alignment in medical communication. Its architecture comprises three core components:
### Encoder: 
Fuses verbal (text) and non-verbal (audio) cues via cross-attention and gated fusion to generate unified contextual representations.
###  Decoder:
Performs joint emotion classification (7 categories: neutral, happiness, sadness, surprise, fear, anger, disgust) and semantic interpretation, producing structured reasoning paths (emotion analysis → semantic intent → response guidelines).
### Responder: 
Synthesizes responses by integrating structured reasoning paths with domain-specific knowledge via a retrieval-augmented generation (RAG) framework.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [API Documentation](#api-documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

In in-hospital patient education, traditional text-only LLMs often fail to capture non-verbal cues (e.g., tone, speech rate), leading to suboptimal communication outcomes. Dolphin introduces an innovative multimodal framework combined with structured reasoning to overcome these challenges:
### Emotional and Semantic Alignment: 
By fusing audio and text data, Dolphin better recognizes patients’ emotional states and interprets their underlying informational needs, reducing misalignment compared to text-only models.
### Clinically Validated Efficacy: 
A double-blinded randomized controlled trial (conducted in the Pain & Sleep Disorders Department) demonstrated that Dolphin improved patient satisfaction, treatment adherence, and healthcare providers’ willingness to adopt the tool—all key indicators of effective patient education.
### Lightweight and Deployable: 
Optimized for low-resource hardware, Dolphin can run locally without relying on cloud infrastructure, enabling seamless integration into existing clinical workflows (e.g., Electronic Medical Record (EMR) systems) with minimal latency.

## Key Features
### 1. Multimodal Emotional and Semantic Modeling
**Input Support: ** 
**Audio**: Accepts 16kHz mono audio files (WEBM/WAV format) with sufficient signal quality. Extracts dual audio features: verbal embeddings via Contrastive Language-Audio Pretraining (CLAP) and non-verbal acoustic features via the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS).  
**Text**: Generates text embeddings (via CLAP) that are fused with audio features using bidirectional cross-attention, ensuring unified contextual understanding.  
**Value Proposition**: Enables nuanced recognition of patients’ emotional states (e.g., anxiety, relief) and accurate interpretation of their communicative intent (e.g., requesting clarification, expressing concerns), which is critical for empathetic patient education.
### 2. Interpretable Structured Reasoning Paths
To address the "black-box" limitation of many AI models and build trust among healthcare providers, Dolphin generates transparent, three-stage reasoning paths:  
**Emotion Interpretation**: Identifies the most prominent emotional states of the patient, with clarity on relative confidence levels.  
**Semantic Interpretation**: Infers the patient’s core informational needs or communicative goals (e.g., understanding a procedure, seeking reassurance).  
**Response Guidelines**: Provides tailored recommendations for communication—including appropriate tone, key content to address, and strategies to enhance empathy.  
**Value Proposition**: Improves providers’ confidence in the model by making its decision-making process visible, aligning with clinical reasoning norms.
### 3. Clinical Adaptability and Ethical Compliance
**Department-Specific Customization**: Supports patient education across multiple clinical departments (e.g., pediatrics, radiology, endoscopy, pain management). Integrates a Retrieval-Augmented Generation (RAG) framework to pull in department-specific knowledge (e.g., procedural guidelines, post-treatment care tips).  
**Ethical and Regulatory Alignment**: Adheres to the Declaration of Helsinki and GDPR principles, with robust data anonymization, encryption, and algorithmic interpretability to protect patient privacy and ensure compliance with healthcare regulations.
### 4. Easy Deployment and Usability
**Low Hardware Requirements**: Runs on standard local hardware (no specialized high-performance computing needed), making it accessible for resource-constrained clinical settings (e.g., community health centers).  
**User-Friendly Tools**: Includes a Flask-based WebDemo for real-time interaction, supporting audio recording/upload and visualizing the model’s reasoning paths. Offers a standardized API for integration with EMR systems, minimizing disruption to clinical workflows.

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
~~~

## Configuration
Edit config in file DolphinInOnefile.py:
~~~
    config = {
        'RAG_MODEL_PATH': '', # fill in your own
        'RAG_KNOWLEDGE_PATH': '', # fill in your own
        'AUDIO_MODEL_PATH': '', # fill in your own
        'AUDIO_SAVE_PATH': '', # fill in your own
        'DOLPHIN_MODEL_PATH': 'dolphinencoder.pth',
        'CLAP_MODEL_PATH': '' # fill in your own
    }
~~~

## Quick Start
Basic Usage:
~~~
python DolphinInOnefile.py
~~~

Webdemo: 
We also provide a lightweight webdemo for visual application.
~~~
cd webdemo
python gui.py
~~~
After finishing deployment, you can try the webdemo on the local web page 127.0.0.1:5000.

Note: ffmpeg is required for this project.

## Data Preparation
RAG Knowledge Base Format (.xlsx or .csv):

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
                "model": "deepseek-chat", #model name, such as "gpt-4o", "gpt-4o-mini", "deepseek-r1", etc.
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
For research collaborations or technical inquiries, please feel free to contact us for any questions or comments: 
Zigeng Huang, E-mail: yuuko_huang@pumc.edu.cn; 
Erping Long, E-mail: erping.long@ibms.pumc.edu.cn;
Peixing Wan, E-mail: peixing@bjmu.edu.cn.
