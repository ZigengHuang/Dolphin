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
from transformers import ClapModel, ClapProcessor

# Model definitions consistent with your training code
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
CLAP_FEAT_DIM = 512
CLAP_AUDIO_SR = 48000
CLAP_AUDIO_DURATION = 3

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module: use text features as Query and audio features as Key/Value (or vice versa)"""
    def __init__(self, feat_dim=CLAP_FEAT_DIM, num_heads=4, dropout=0.3):
        super().__init__()
        # Cross-attention layer (text→audio)
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Cross-attention layer (audio→text)
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Fusion feature projection (dimension reduction + normalization)
        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),  # Concatenate text+audio fusion features
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, text_feat: torch.Tensor, audio_feat: torch.Tensor):
        """
        Forward propagation: dual cross-attention fusion
        :param text_feat: Text features, shape [batch_size, 1, feat_dim] (add seq dimension for attention compatibility)
        :param audio_feat: Audio features, shape [batch_size, 1, feat_dim]
        :return: Fused features, shape [batch_size, feat_dim]
        """
        # 1. Text→audio cross-attention (use text Query to attend to audio Key)
        text_attn_audio, _ = self.text_to_audio_attn(
            query=text_feat, key=audio_feat, value=audio_feat
        )  # Shape: [batch_size, 1, feat_dim]
        
        # 2. Audio→text cross-attention (use audio Query to attend to text Key)
        audio_attn_text, _ = self.audio_to_text_attn(
            query=audio_feat, key=text_feat, value=text_feat
        )  # Shape: [batch_size, 1, feat_dim]
        
        # 3. Concatenate fusion (text attention result + audio attention result)
        fused = torch.cat([text_attn_audio, audio_attn_text], dim=-1)  # [batch_size, 1, 2*feat_dim]
        fused = fused.squeeze(1)  # Remove seq dimension: [batch_size, 2*feat_dim]
        
        # 4. Projection dimension reduction (unify to original feature dimension)
        fused_feat = self.fusion_proj(fused)  # [batch_size, feat_dim]
        return fused_feat


class TransformerClassifier(nn.Module):
    """3-layer Transformer encoder classifier head (aligned with text model)"""
    def __init__(self, input_dim=CLAP_FEAT_DIM, num_heads=8, hidden_dim=2048, num_layers=3, num_emotions=len(EMOTIONS), dropout=0.3):
        super().__init__()
        # Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input feature dimension
            nhead=num_heads,    # Number of attention heads (must divide input_dim)
            dim_feedforward=hidden_dim,  # Feedforward network hidden dimension
            dropout=dropout,
            batch_first=True    # Batch dimension first
        )
        # 3-layer Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        # Final classifier layer
        self.classifier = nn.Linear(input_dim, num_emotions)
        # Positional encoding (add position info for single-element sequence)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, input_dim))  # [1, 1, input_dim]

    def forward(self, x: torch.Tensor):
        """
        :param x: Fused features, shape [batch_size, input_dim]
        :return: Emotion classification logits, shape [batch_size, num_emotions]
        """
        # Add sequence dimension (Transformer needs [batch, seq_len, dim])
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Add positional encoding
        x = x + self.pos_encoder  # [batch_size, 1, input_dim]
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch_size, 1, input_dim]
        
        # Take first element of sequence as final feature
        x = x.squeeze(1)  # [batch_size, input_dim]
        
        # Classification output
        return self.classifier(x)  # [batch_size, num_emotions]


class CLAPBimodalEmotionModel(nn.Module):
    def __init__(
        self,
        local_clap_path: str,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.feat_dim = CLAP_FEAT_DIM
        print(f"Loading CLAP bimodal model (device: {device})")
        
        # 1. Load CLAP processor (supports audio+text preprocessing)
        self.clap_processor = ClapProcessor.from_pretrained(local_clap_path)
        print("✅ CLAP Processor loaded (supports audio+text)")
        
        # 2. Load CLAP full model (contains audio+text encoders)
        self.clap_model = ClapModel.from_pretrained(local_clap_path).to(device)
        print("✅ CLAP Model loaded (contains audio/text encoders)")
        
        # 3. Freeze CLAP bottom layers, fine-tune top layers
        self._freeze_clap_bottom_layers()
        
        # 4. Bimodal fusion module (cross-attention)
        self.cross_attn_fusion = CrossAttentionFusion(
            feat_dim=self.feat_dim,
            num_heads=4,
            dropout=0.3
        ).to(device)
        
        # 5. Classifier head (replace with 3-layer Transformer, aligned with text model)
        self.classifier = TransformerClassifier(
            input_dim=self.feat_dim,
            num_heads=8,  # 512 dimension suitable for 8-head attention (512/8=64)
            hidden_dim=2048,
            num_layers=3,  # 3-layer Transformer
            num_emotions=len(EMOTIONS),
            dropout=0.3
        ).to(device)

    def _freeze_clap_bottom_layers(self):
        """Freeze CLAP bottom layers, only fine-tune top layers"""
        # Text encoder freeze (12 layers, unfreeze last 2)
        text_encoder = self.clap_model.text_model.encoder
        total_text_layers = len(text_encoder.layer)
        unfreeze_text_layers = [10, 11]
        for layer_idx, layer in enumerate(text_encoder.layer):
            for param in layer.parameters():
                param.requires_grad = (layer_idx in unfreeze_text_layers)
        print(f"✅ Text encoder frozen: total {total_text_layers} layers, unfreeze layers {unfreeze_text_layers}")

        # Audio encoder freeze (12 blocks, unfreeze last 2)
        audio_encoder = self.clap_model.audio_model.audio_encoder
        all_audio_blocks = []
        for stage in audio_encoder.layers:
            all_audio_blocks.extend(stage.blocks)
        total_audio_blocks = len(all_audio_blocks)
        unfreeze_audio_blocks = [10, 11]
        for block_idx, block in enumerate(all_audio_blocks):
            for param in block.parameters():
                param.requires_grad = (block_idx in unfreeze_audio_blocks)
        print(f"✅ Audio encoder frozen: total {total_audio_blocks} blocks, unfreeze blocks {unfreeze_audio_blocks}")

        # Freeze CLAP contrastive learning parameters
        self.clap_model.logit_scale_a.requires_grad = False
        self.clap_model.logit_scale_t.requires_grad = False
        print("✅ CLAP contrastive learning parameters (logit_scale_a/t) frozen")

    def extract_modal_features(self, texts: list, audios: torch.Tensor):
        """Extract CLAP audio+text features"""
        # Text feature extraction
        text_inputs = self.clap_processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(self.device)
        text_feat = self.clap_model.get_text_features(** text_inputs)  # [batch_size, feat_dim]

        # Audio feature extraction
        if audios is None or audios.shape[0] == 0:
            raise ValueError(f"Invalid audio data! audios shape: {audios.shape if audios is not None else 'None'}")

        audio_inputs = self.clap_processor(
            audios=audios.cpu().numpy(),
            sampling_rate=CLAP_AUDIO_SR,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=CLAP_AUDIO_SR * CLAP_AUDIO_DURATION
        ).to(self.device)

        if "input_features" not in audio_inputs:
            raise KeyError("CLAP Processor did not generate input_features! Check if audio format is correct")

        audio_feat = self.clap_model.get_audio_features(**audio_inputs)  # [batch_size, feat_dim]

        # Add seq dimension
        text_feat = text_feat.unsqueeze(1)  # [batch_size, 1, feat_dim]
        audio_feat = audio_feat.unsqueeze(1)  # [batch_size, 1, feat_dim]

        return text_feat, audio_feat

    def forward(self, texts: list, audios: torch.Tensor):
        """Forward propagation: feature extraction → cross-attention fusion → Transformer classification"""
        # 1. Extract bimodal features
        text_feat, audio_feat = self.extract_modal_features(texts, audios)
        
        # 2. Cross-attention fusion
        fused_feat = self.cross_attn_fusion(text_feat, audio_feat)  # [batch_size, feat_dim]
        
        # 3. Transformer classification (aligned with text model)
        logits = self.classifier(fused_feat)  # [batch_size, num_emotions]
        return logits


class DolphinCore:
    def __init__(self, config):
        """
        Initialize core functionality modules
        
        Parameters:
            config: Dictionary containing the following keys:
                - RAG_MODEL_PATH: Path to RAG model
                - RAG_KNOWLEDGE_PATH: Path to knowledge base file
                - AUDIO_MODEL_PATH: Path to audio transcription model
                - AUDIO_SAVE_PATH: Path to save audio files
                - BIMODAL_MODEL_PATH: Path to trained bimodal model
                - CLAP_MODEL_PATH: Path to CLAP model
        """
        self.config = config
        
        # Initialize models
        self.init_models()
        
        # Load knowledge base
        self.load_knowledge_base()
        
        # Precompute knowledge embeddings
        self.precompute_knowledge_embeddings()
        
    def init_models(self):
        """Initialize all required models"""
        print("Initializing bimodal emotion model and audio transcription model...")
        
        # Initialize bimodal emotion model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = CLAPBimodalEmotionModel(
            local_clap_path=self.config['CLAP_MODEL_PATH'],
            device=self.device
        )
        self.emotion_model.load_state_dict(
            torch.load(self.config['BIMODAL_MODEL_PATH'], map_location=self.device)
        )
        self.emotion_model.eval()
        
        # Initialize audio transcription model
        from funasr import AutoModel
        self.audio_model = AutoModel(
            model=self.config['AUDIO_MODEL_PATH'], 
            disable_update=True, 
            language="zh"
        )
        
        print("Loading RAG model and tokenizer...")
        self.rag_tokenizer = AutoTokenizer.from_pretrained(self.config['RAG_MODEL_PATH'])
        self.rag_model = AutoModelForMaskedLM.from_pretrained(self.config['RAG_MODEL_PATH'])
        
    def load_knowledge_base(self):
        """Load knowledge base"""
        print(f"Loading knowledge base from {self.config['RAG_KNOWLEDGE_PATH']}...")
        self.knowledge_df = pd.read_excel(self.config['RAG_KNOWLEDGE_PATH'])
        
        # Check column names and select appropriate column
        if 'Content' in self.knowledge_df.columns:
            self.knowledge = self.knowledge_df['Content'].tolist()
        elif 'content' in self.knowledge_df.columns:
            self.knowledge = self.knowledge_df['content'].tolist()
        elif 'text' in self.knowledge_df.columns:
            self.knowledge = self.knowledge_df['text'].tolist()
        elif 'Text' in self.knowledge_df.columns:
            self.knowledge = self.knowledge_df['Text'].tolist()
        else:
            # If no suitable column found, use first column
            first_col = self.knowledge_df.columns[0]
            self.knowledge = self.knowledge_df[first_col].tolist()
            print(f"⚠️  No 'Content' column found, using first column '{first_col}' as knowledge base content")
        
        print(f"✅ Loaded {len(self.knowledge)} knowledge base entries")
        
    def precompute_knowledge_embeddings(self):
        """Precompute embeddings for knowledge base"""
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
        """Split long text into chunks suitable for model processing"""
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
        """
        Transcribe audio file to text
        
        Parameters:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Handle webm format audio
            if audio_path.endswith('.webm'):
                wav_path = audio_path.replace('.webm', '.wav')
                audio = AudioSegment.from_file(audio_path, format="webm")
                audio.export(wav_path, format="wav")
                audio_path = wav_path

            # Use pydub to process audio file in chunks
            audio = AudioSegment.from_wav(audio_path)
            chunk_length = 60 * 1000  # 60 seconds
            chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

            transcriptions = []
            for idx, chunk in enumerate(chunks):
                # Save chunked audio
                chunk_path = os.path.join(self.config['AUDIO_SAVE_PATH'], f"chunk_{idx}.wav")
                chunk.export(chunk_path, format="wav")

                # Use funasr to transcribe each chunk
                res = self.audio_model.generate(input=chunk_path, batch_size_s=300, batch_size_threshold_s=60)
                transcriptions.append(res[0]['text'] if res else "")

                # Delete temporary file
                os.remove(chunk_path)

            return "\n".join(transcriptions)
        except Exception as e:
            print(f"Audio transcription error: {e}")
            return ""
            
    def analyze_emotion(self, audio_path, text):
        """
        Analyze emotions in audio using bimodal model
        
        Parameters:
            audio_path: Path to audio file
            text: Corresponding text
            
        Returns:
            List of emotion analysis results
        """
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            # Resample to 48000Hz
            if sr != CLAP_AUDIO_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_AUDIO_SR)
            # Unify length to 3 seconds
            target_length = CLAP_AUDIO_SR * CLAP_AUDIO_DURATION
            audio_len = len(audio)
            
            if audio_len < target_length:
                # Short audio: pad with zeros to keep centered
                pad_before = (target_length - audio_len) // 2
                pad_after = target_length - audio_len - pad_before
                audio = np.pad(audio, (pad_before, pad_after), mode="constant")
            else:
                # Long audio: take middle 3 seconds
                start_idx = (audio_len - target_length) // 2
                audio = audio[start_idx: start_idx + target_length]
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Use bimodal model for prediction
            with torch.no_grad():
                logits = self.emotion_model([text], audio_tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Organize results
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_emotion = EMOTIONS[pred_idx]
            emotion_probs = {EMOTIONS[i]: round(probs[i], 4) for i in range(len(EMOTIONS))}
            
            # Return formatted results
            emotions = []
            for emotion, prob in emotion_probs.items():
                emotions.append({"emotion": emotion, "score": prob})
                
            # Sort by probability and return top 3
            emotions.sort(key=lambda x: x["score"], reverse=True)
            return emotions[:3]
            
        except Exception as e:
            print(f"Bimodal emotion analysis error: {e}")
            return []
            
    def retrieve_knowledge(self, query, k=5):
        """
        Retrieve relevant information from knowledge base
        
        Parameters:
            query: Query text
            k: Number of relevant knowledge items to return
            
        Returns:
            List of relevant knowledge texts
        """
        try:
            inputs_query = self.rag_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs_query = self.rag_model(**inputs_query, output_hidden_states=True)
                query_embedding = outputs_query.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()

            similarities = []
            for chunk_embeddings in self.knowledge_embeddings:
                chunk_similarities = []
                for embedding in chunk_embeddings:
                    similarity_score = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        embedding.reshape(1, -1)
                    )[0][0]
                    chunk_similarities.append(similarity_score)
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
        """
        Model A analysis - Emotion and semantic analysis
        Parameters:
            text: Input text
            emotions: Emotion analysis results
            summary: Background summary (optional)
        Returns:
            Analysis result text
        """
        context = [{
            "role": "system",
            "content": """
                You are an insightful medical assistant working in a hospital, specializing in analyzing emotions and underlying meanings in doctor-patient conversations. When patients communicate, models will identify the emotions (and probabilities) represented in their audio and text. Your task is to analyze the emotional intent and deeper semantics of the patient's speech based on the discrepancy between the text meaning and audio emotions. After analyzing emotions and semantics, provide brief and accurate suggestions (within 50 words) on how to respond to the patient's questions.
                Your response must strictly follow this Markdown format:
                Patient content: {patient's input content}
                Emotions: {emotion analysis results, select top 3 most probable emotions to explain}
                Semantics: {semantic analysis results}
                Profile analysis: {analyze patient's profile characteristics and needs, provide comprehensive understanding}
                Response guide: {suggestions for responding to the patient}
            """
        }]
        
        input_text = f"Background summary: {summary}\nPatient content: {text}\nEmotions: {emotions}" if summary else f"Patient content: {text}\nEmotions: {emotions}"
        
        try:
            response = self.call_chatanywhere(input_text, context=context)
            return response
        except Exception as e:
            print(f"Model A analysis error: {e}")
            return None
            
    def model_b_response(self, text, model_a_output, knowledge):
        """
        Model B response generation
        
        Parameters:
            text: Original input text
            model_a_output: Model A analysis results
            knowledge: Relevant knowledge
            
        Returns:
            Generated response text
        """
        context = [
            {"role": "system", "content": "You are an empathetic and professional medical assistant working in a hospital, supporting outpatient doctors by providing emotional support and professional medical consultation to patients. After receiving emotion analysis results, you need to engage in targeted dialogue responses that align with medical knowledge and demonstrate good medical literacy, based on the patient's emotional state and potential psychological needs. Your responses should be as concise and clear as possible."},
            {"role": "user", "content": f"Patient speech content: {text}"},
            {"role": "assistant", "content": f"Analysis results: {model_a_output}"},
            {"role": "user", "content": f"Relevant knowledge: {knowledge}"}
        ]
        
        try:
            response = self.call_chatanywhere(text, context=context)
            return response
        except Exception as e:
            print(f"Model B response generation error: {e}")
            return None
            
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
                'Authorization': 'Bearer YOUR_API_KEY_HERE', #fill in your own api-key
            }

            # Build message format
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
            messages.append({"role": "user", "content": input_text})

            data = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.7
            }

            response = requests.post(
                'https://api.chatanywhere.tech/v1/chat/completions',  # Fixed: removed trailing space
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
        """
        Process complete conversation workflow
        
        Parameters:
            audio_path: Path to audio file
            text_input: Optional text input (if not provided, will transcribe from audio)
            summary: Optional background summary
            
        Returns:
            Dictionary containing all processing results
        """
        results = {}
        
        # 1. Transcribe audio if text not provided
        if not text_input:
            print("Transcribing audio...")
            text_input = self.transcribe_audio(audio_path)
            print(f"Transcription result: {text_input}")
        
        # 2. Emotion analysis
        print("Analyzing emotions using bimodal model...")
        emotions = self.analyze_emotion(audio_path, text_input)
        print(f"Bimodal emotion analysis results: {emotions}")
        
        # 3. Knowledge retrieval
        print("Retrieving relevant knowledge...")
        knowledge = self.retrieve_knowledge(text_input)
        knowledge_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(knowledge)])
        
        # 4. Model A analysis
        print("Running Model A analysis...")
        model_a_output = self.model_a_analysis(text_input, emotions, summary)
        print(f"Model A analysis results:\n{model_a_output}")
        
        # 5. Model B response
        print("Generating Model B response...")
        model_b_output = self.model_b_response(text_input, model_a_output, knowledge_str)
        print(f"Model B response:\n{model_b_output}")
        
        # Collect all results
        results['transcription'] = text_input
        results['emotions'] = emotions
        results['knowledge'] = knowledge
        results['model_a_output'] = model_a_output
        results['model_b_output'] = model_b_output
        
        return results

# Example usage
if __name__ == "__main__":
    # Configuration parameters
    config = {
        'RAG_MODEL_PATH': '', # fill in the model location
        'RAG_KNOWLEDGE_PATH': '', # fill in the xlsx file location
        'AUDIO_MODEL_PATH': '', # fill in the model location
        'AUDIO_SAVE_PATH': '', # fill in the audio save location
        'BIMODAL_MODEL_PATH': 'dolphinemo.pth',
        'CLAP_MODEL_PATH': 'clap-htsat-unfused' # fill in the model location
    }
    
    # Initialize core module
    dolphin = DolphinCore(config)
    
    # Example audio file path
    audio_file = "audio_demo.webm"
    
    # Process conversation
    results = dolphin.process_conversation(audio_file)
    
    # Print results
    print("\nFinal results:")
    print(f"Transcription: {results['transcription']}")
    print(f"Emotion analysis: {results['emotions']}")
    print(f"Structured reasoning path:\n{results['model_a_output']}")
    print(f"Dolphin response:\n{results['model_b_output']}")