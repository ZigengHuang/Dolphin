import os
import torchaudio
from pydub import AudioSegment
from funasr import AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests

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
        print("Initializing emotion model and audio transcription model...")
        self.emotion_model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
        self.audio_model = AutoModel(model=self.config['AUDIO_MODEL_PATH'], disable_update=True, language="zh")
        
        print("Loading RAG model and tokenizer...")
        self.rag_tokenizer = AutoTokenizer.from_pretrained(self.config['RAG_MODEL_PATH'])
        self.rag_model = AutoModelForMaskedLM.from_pretrained(self.config['RAG_MODEL_PATH'])
        
    def load_knowledge_base(self):
        """Load knowledge base"""
        print(f"Loading knowledge base from {self.config['RAG_KNOWLEDGE_PATH']}...")
        self.knowledge_df = pd.read_excel(self.config['RAG_KNOWLEDGE_PATH'])
        self.knowledge = self.knowledge_df['内容'].tolist()
        
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
            
    def analyze_emotion(self, audio_path):
        """
        Analyze emotions in audio
        
        Parameters:
            audio_path: Path to audio file
            
        Returns:
            List of emotion analysis results
        """
        try:
            # Handle webm format audio
            if audio_path.endswith('.webm'):
                wav_path = audio_path.replace('.webm', '.wav')
                audio = AudioSegment.from_file(audio_path, format="webm")
                audio.export(wav_path, format="wav")
                audio_path = wav_path

            waveform, sample_rate = torchaudio.load(audio_path)
            res = self.emotion_model.generate(waveform, granularity="utterance", extract_embedding=False)
            
            emotions = []
            for item in res:
                labels = item['labels']
                scores = item['scores']
                emotions.extend([{"emotion": lbl, "score": scr} for lbl, scr in zip(labels, scores)])
                
            return emotions
        except Exception as e:
            print(f"Emotion analysis error: {e}")
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
                'Authorization': 'Bearer sk-yhGP21PPtDMiB8ptAow3zDn3pHQk46j5D2N34iXyspqvfCOG',
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
        
        # 1. Transcribe audio
        if not text_input:
            print("Transcribing audio...")
            text_input = self.transcribe_audio(audio_path)
            print(f"Transcription result: {text_input}")
        
        # 2. Emotion analysis
        print("Analyzing emotions...")
        emotions = self.analyze_emotion(audio_path)
        print(f"Emotion analysis results: {emotions}")
        
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
        results['model_b_response'] = model_b_output
        
        return results

# Example usage
if __name__ == "__main__":
    # Configuration parameters
    config = {
        'RAG_MODEL_PATH': 'E:/00Graduate/Dolphin/代码/Erlangshen-SimCSE-110M-Chinese',
        'RAG_KNOWLEDGE_PATH': 'E:/00Graduate/Dolphin/数据/rag-2024.12.29/knowledge.xlsx',
        'AUDIO_MODEL_PATH': 'E:/00Graduate/Dolphin/数据/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'AUDIO_SAVE_PATH': 'E:/00Graduate/Dolphin/数据/音频'
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
    print(f"Model A analysis:\n{results['model_a_output']}")
    print(f"Model B response:\n{results['model_b_response']}")