import os
import torchaudio
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import librosa
import numpy as np
import opensmile
import soundfile as sf
import tempfile
from transformers import ClapModel, ClapProcessor

# Predefined constants
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
CLAP_FEAT_DIM = 512
CLAP_AUDIO_SR = 48000
CLAP_AUDIO_DURATION = 3  # seconds
EGEMAPS_DIM = 88         # Dimension of eGeMAPS functional features
FUSION_DIM = 512         # Unified embedding dimension after projection


class CrossAttentionModule(nn.Module):
    """Cross-attention layer using multi-head attention."""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query, key, value):
        """Compute attention output given query, key, and value sequences."""
        output, _ = self.multihead_attn(query, key, value)
        return output


class DolphinEncoder(nn.Module):
    """
    Hierarchical multimodal encoder that fuses CLAP audio/text embeddings 
    with segment-level eGeMAPS features via cross-attention.
    """
    def __init__(self, local_clap_path, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.d_model = FUSION_DIM

        # Load and freeze CLAP model
        self.clap_processor = ClapProcessor.from_pretrained(local_clap_path)
        self.clap_model = ClapModel.from_pretrained(local_clap_path)
        for param in self.clap_model.parameters():
            param.requires_grad = False

        # Linear projections to unified embedding space
        self.audio_proj = nn.Linear(CLAP_FEAT_DIM, self.d_model)
        self.text_proj = nn.Linear(CLAP_FEAT_DIM, self.d_model)
        self.egemaps_proj = nn.Linear(EGEMAPS_DIM, self.d_model)

        # Cross-attention modules
        self.audio_cross_attn = CrossAttentionModule(self.d_model)
        self.text_cross_attn = CrossAttentionModule(self.d_model)

        # Gating mechanism for intra-audio fusion
        self.audio_gate_proj = nn.Linear(self.d_model * 2, self.d_model)

        # Final projection (512 → 512) to match trained model checkpoint
        self.final_proj = nn.Linear(self.d_model, self.d_model)

    def extract_clap_features(self, texts, audios):
        """Extract CLAP text and audio embeddings."""
        # Text encoding
        text_inputs = self.clap_processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(self.device)
        text_feat = self.clap_model.get_text_features(**text_inputs)

        # Audio encoding: CLAP expects list of numpy arrays
        audio_list = [audios[i].cpu().numpy() for i in range(audios.shape[0])]
        audio_inputs = self.clap_processor(
            audios=audio_list,
            sampling_rate=CLAP_AUDIO_SR,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=CLAP_AUDIO_SR * CLAP_AUDIO_DURATION
        ).to(self.device)
        audio_feat = self.clap_model.get_audio_features(**audio_inputs)
        return text_feat, audio_feat

    def extract_egemaps_features(self, audio_path):
        """
        Extract eGeMAPS functional features from 3-second non-overlapping segments.
        Returns a tensor of shape [1, T, 88] on the model's device.
        """
        if audio_path is None:
            raise ValueError("audio_path is required for eGeMAPS extraction")

        # Load and resample audio to 16 kHz (eGeMAPS standard)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Segment audio into 3-second chunks
        seg_samples = int(3.0 * sr)
        total_len = len(audio)
        num_segs = max(1, total_len // seg_samples)

        # Initialize openSMILE extractor
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )

        features_list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(num_segs):
                start = i * seg_samples
                end = start + seg_samples
                if end > total_len:
                    seg = audio[start:]
                    if len(seg) == 0:
                        break
                    seg = np.pad(seg, (0, end - total_len), mode='constant')
                else:
                    seg = audio[start:end]

                # Save segment and extract features
                seg_path = os.path.join(tmp_dir, f"seg_{i}.wav")
                sf.write(seg_path, seg, sr)
                try:
                    df = smile.process_file(seg_path)
                    feat = df.values.squeeze()
                    features_list.append(feat)
                except Exception as e:
                    print(f"Warning: eGeMAPS extraction failed for segment {i}: {e}")
                    features_list.append(np.zeros(88))

        if not features_list:
            features_list = [np.zeros(88)]

        # Convert to tensor and add batch dimension
        egemaps_array = np.stack(features_list, axis=0)
        egemaps_tensor = torch.tensor(egemaps_array, dtype=torch.float32, device=self.device)
        return egemaps_tensor.unsqueeze(0)

    def temporal_expand(self, feat, T):
        """Expand a feature vector into a sequence of length T."""
        return feat.unsqueeze(1).expand(-1, T, -1)

    def forward(self, texts, audios, audio_path=None):
        batch_size = audios.shape[0]

        # Step 1: Extract CLAP features
        text_feat, audio_feat = self.extract_clap_features(texts, audios)

        # Step 2: Extract eGeMAPS features (segment-level)
        egemaps_feat = self.extract_egemaps_features(audio_path)
        egemaps_feat = egemaps_feat.to(self.device)

        # Step 3: Temporal expansion to align time steps
        T = egemaps_feat.shape[1]
        F_a = self.temporal_expand(audio_feat, T)
        F_t = self.temporal_expand(text_feat, T)

        # Step 4: Project to shared embedding space
        F_a_prime = self.audio_proj(F_a)
        F_t_prime = self.text_proj(F_t)
        F_e_prime = self.egemaps_proj(egemaps_feat)

        # Broadcast eGeMAPS to match batch size (supports batch_size=1)
        if F_e_prime.shape[0] == 1 and batch_size > 1:
            F_e_prime = F_e_prime.expand(batch_size, -1, -1)

        # Step 5: Intra-audio fusion (CLAP audio + eGeMAPS)
        A_a_to_e = self.audio_cross_attn(F_a_prime, F_e_prime, F_e_prime)
        A_e_to_a = self.audio_cross_attn(F_e_prime, F_a_prime, F_a_prime)
        combined_attn = torch.cat([A_a_to_e, A_e_to_a], dim=-1)
        gate_weights = torch.sigmoid(self.audio_gate_proj(combined_attn))
        Z_a = gate_weights * A_a_to_e + (1 - gate_weights) * A_e_to_a

        # Step 6: Cross-modal fusion (text queries attend to fused audio)
        Z_fused_seq = self.text_cross_attn(F_t_prime, Z_a, Z_a)
        Z_fused = Z_fused_seq.mean(dim=1)

        # Step 7: Final projection
        Z_prime = self.final_proj(Z_fused)
        return Z_prime


class DolphinEmotionClassifier(nn.Module):
    """Emotion classifier built on top of the Dolphin encoder."""
    def __init__(self, local_clap_path, device=torch.device("cpu")):
        super().__init__()
        self.encoder = DolphinEncoder(local_clap_path, device)
        self.classifier = nn.Linear(512, len(EMOTIONS))

    def forward(self, texts, audios, audio_path=None):
        fused_repr = self.encoder(texts, audios, audio_path)
        logits = self.classifier(fused_repr)
        return logits


class DolphinCore:
    """Main pipeline orchestrator for transcription, emotion analysis, and response generation."""
    def __init__(self, config):
        self.config = config
        self.init_models()
        self.load_knowledge_base()
        self.precompute_knowledge_embeddings()

    def init_models(self):
        """Initialize all submodules and move models to appropriate device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = DolphinEmotionClassifier(
            local_clap_path=self.config['CLAP_MODEL_PATH'],
            device=self.device
        )
        self.emotion_model = self.emotion_model.to(self.device)
        self.emotion_model.load_state_dict(
            torch.load(self.config['DOLPHIN_MODEL_PATH'], map_location=self.device)
        )
        self.emotion_model.eval()

        # Audio transcription model (FunASR)
        from funasr import AutoModel
        self.audio_model = AutoModel(
            model=self.config['AUDIO_MODEL_PATH'],
            disable_update=True,
            language="zh"
        )

        # RAG components
        print("Loading RAG model and tokenizer...")
        self.rag_tokenizer = AutoTokenizer.from_pretrained(self.config['RAG_MODEL_PATH'])
        self.rag_model = AutoModelForMaskedLM.from_pretrained(self.config['RAG_MODEL_PATH'])

    def load_knowledge_base(self):
        """Load knowledge base from Excel file and detect content column."""
        print(f"Loading knowledge base from {self.config['RAG_KNOWLEDGE_PATH']}...")
        self.knowledge_df = pd.read_excel(self.config['RAG_KNOWLEDGE_PATH'])

        # Identify content column
        content_col = None
        for col in ['Content', 'content', 'Text', 'text']:
            if col in self.knowledge_df.columns:
                content_col = col
                break
        if content_col is None:
            content_col = self.knowledge_df.columns[0]
            print(f"Warning: Using first column '{content_col}' as content")

        self.knowledge = self.knowledge_df[content_col].dropna().tolist()
        print(f"Loaded {len(self.knowledge)} knowledge entries")

    def precompute_knowledge_embeddings(self):
        """Precompute BERT embeddings for all knowledge chunks."""
        print("Precomputing knowledge embeddings...")
        self.knowledge_embeddings = []

        for text in self.knowledge[1:]:
            chunks = self.split_text_into_chunks(text)
            chunk_embeddings = []
            for chunk in chunks:
                inputs = self.rag_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.rag_model(**inputs, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
                chunk_embeddings.append(embedding)
            self.knowledge_embeddings.append(chunk_embeddings)

        print("Knowledge embeddings precomputation completed")

    def split_text_into_chunks(self, text, max_length=512):
        """Split long text into model-compatible chunks."""
        tokens = self.rag_tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            if current_length >= max_length:
                chunk_text = self.rag_tokenizer.convert_tokens_to_string(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunk_text = self.rag_tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
        return chunks

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using FunASR, with WebM support and chunking."""
        try:
            if audio_path.endswith('.webm'):
                wav_path = audio_path.replace('.webm', '.wav')
                audio = AudioSegment.from_file(audio_path, format="webm")
                audio.export(wav_path, format="wav")
                audio_path = wav_path

            audio = AudioSegment.from_wav(audio_path)
            chunk_length = 60 * 1000  # 60 seconds
            chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

            transcriptions = []
            for idx, chunk in enumerate(chunks):
                chunk_path = os.path.join(self.config['AUDIO_SAVE_PATH'], f"chunk_{idx}.wav")
                chunk.export(chunk_path, format="wav")
                res = self.audio_model.generate(input=chunk_path, batch_size_s=300, batch_size_threshold_s=60)
                transcriptions.append(res[0]['text'] if res else "")
                os.remove(chunk_path)

            return "\n".join(transcriptions)
        except Exception as e:
            print(f"Audio transcription error: {e}")
            return ""

    def analyze_emotion(self, audio_path, text):
        """Run multimodal emotion analysis using the trained Dolphin model."""
        try:
            # Load and preprocess audio to 3-second centered clip at 48kHz
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != CLAP_AUDIO_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_AUDIO_SR)
            target_length = CLAP_AUDIO_SR * CLAP_AUDIO_DURATION
            audio_len = len(audio)

            if audio_len < target_length:
                pad_before = (target_length - audio_len) // 2
                pad_after = target_length - audio_len - pad_before
                audio = np.pad(audio, (pad_before, pad_after), mode="constant")
            else:
                start_idx = (audio_len - target_length) // 2
                audio = audio[start_idx:start_idx + target_length]

            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.emotion_model([text], audio_tensor, audio_path)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # Format results
            emotions = [{"emotion": EMOTIONS[i], "score": round(probs[i], 4)} for i in range(len(EMOTIONS))]
            emotions.sort(key=lambda x: x["score"], reverse=True)
            return emotions[:3]
        except Exception as e:
            print(f"Dolphin emotion analysis error: {e}")
            return []

    def retrieve_knowledge(self, query, k=5):
        """Retrieve top-k relevant knowledge entries using cosine similarity."""
        try:
            inputs_query = self.rag_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs_query = self.rag_model(**inputs_query, output_hidden_states=True)
                query_embedding = outputs_query.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()

            similarities = []
            for chunk_embeddings in self.knowledge_embeddings:
                chunk_similarities = [
                    cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
                    for emb in chunk_embeddings
                ]
                max_similarity = max(chunk_similarities) if chunk_similarities else 0
                similarities.append(max_similarity)

            top_k_idx = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
            top_k_texts = [self.knowledge[idx + 1] for idx in top_k_idx]

            for i, idx in enumerate(top_k_idx):
                print(f"Top {i+1} relevant knowledge (similarity: {similarities[idx]:.2f}): {self.knowledge[idx + 1]}")

            return top_k_texts
        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return []

    def model_a_analysis(self, text, emotions, summary=None):
        """Perform semantic and emotional analysis using LLM (Model A)."""
        system_prompt = (
            "You are an insightful medical assistant specializing in analyzing emotions "
            "and underlying meanings in doctor-patient conversations. Analyze the emotional "
            "intent and deeper semantics based on discrepancies between text and audio emotions. "
            "Provide a structured analysis in the following Markdown format:\n"
            "Patient content: {patient's input}\n"
            "Emotions: {top 3 emotions with scores}\n"
            "Semantics: {semantic interpretation}\n"
            "Profile analysis: {patient profile and needs}\n"
            "Response guide: {concise response suggestion (≤50 words)}"
        )
        context = [{"role": "system", "content": system_prompt}]
        input_text = f"Background summary: {summary}\nPatient content: {text}\nEmotions: {emotions}" if summary else f"Patient content: {text}\nEmotions: {emotions}"

        try:
            return self.call_chatanywhere(input_text, context=context)
        except Exception as e:
            print(f"Model A analysis error: {e}")
            return None

    def model_b_response(self, text, model_a_output, knowledge):
        """Generate empathetic and knowledge-grounded response (Model B)."""
        context = [
            {"role": "system", "content": "You are an empathetic and professional medical assistant..."},
            {"role": "user", "content": f"Patient speech content: {text}"},
            {"role": "assistant", "content": f"Analysis results: {model_a_output}"},
            {"role": "user", "content": f"Relevant knowledge: {knowledge}"}
        ]
        try:
            return self.call_chatanywhere(text, context=context)
        except Exception as e:
            print(f"Model B response generation error: {e}")
            return None

    def call_chatanywhere(self, input_text, context=[]):
        """Call external LLM API (ChatAnywhere) with retry and error handling."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer /api-key'  # fill in your own api-key
            }
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
            messages.append({"role": "user", "content": input_text})
            data = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.7}

            response = requests.post(
                'https://api.chatanywhere.tech/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            print(f"API call failed, status code: {response.status_code}, response: {response.text}")
            return None
        except Exception as e:
            print(f"Error calling ChatAnywhere API: {e}")
            return None

    def process_conversation(self, audio_path, text_input=None, summary=None):
        """End-to-end processing pipeline: transcription → emotion → knowledge → response."""
        results = {}

        if not text_input:
            print("Transcribing audio...")
            text_input = self.transcribe_audio(audio_path)
            print(f"Transcription result: {text_input}")

        print("Analyzing emotions using Dolphin encoder...")
        emotions = self.analyze_emotion(audio_path, text_input)
        print(f"Dolphin emotion analysis results: {emotions}")

        print("Retrieving relevant knowledge...")
        knowledge = self.retrieve_knowledge(text_input)
        knowledge_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(knowledge)])

        print("Running Model A analysis...")
        model_a_output = self.model_a_analysis(text_input, emotions, summary)
        print(f"Model A analysis results:\n{model_a_output}")

        print("Generating Model B response...")
        model_b_output = self.model_b_response(text_input, model_a_output, knowledge_str)
        print(f"Model B response:\n{model_b_output}")

        results.update({
            'transcription': text_input,
            'emotions': emotions,
            'knowledge': knowledge,
            'model_a_output': model_a_output,
            'model_b_output': model_b_output
        })
        return results


# Example usage
if __name__ == "__main__":
    config = {
        'RAG_MODEL_PATH': '',
        'RAG_KNOWLEDGE_PATH': '',
        'AUDIO_MODEL_PATH': '',
        'AUDIO_SAVE_PATH': '',
        'DOLPHIN_MODEL_PATH': '',
        'CLAP_MODEL_PATH': ''
    }

    dolphin = DolphinCore(config)
    audio_file = "audio_demo.webm"
    results = dolphin.process_conversation(audio_file)

    print("\nFinal results:")
    print(f"Transcription: {results['transcription']}")
    print(f"Emotion analysis: {results['emotions']}")
    print(f"Structured reasoning path:\n{results['model_a_output']}")
    print(f"Dolphin response:\n{results['model_b_output']}")