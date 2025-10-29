from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import os
import torchaudio
from funasr import AutoModel
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from sqlalchemy import JSON
import config
import requests


rag_model_path = config.RAG_MODEL_PATH
rag_knowledge_path = config.RAG_KNOWLEDGE_PATH
audio_model_path = config.AUDIO_MODEL_PATH
audio_save_path = config.AUDIO_SAVE_PATH
db_path = config.DB_PATH
api_key = config.API_KEY

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
UPLOAD_FOLDER = config.AUDIO_SAVE_PATH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置 SQLite 数据库
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{config.DB_PATH}/data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 定义数据模型
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 角色：user 或 assistant
    content = db.Column(db.Text, nullable=False)  # 对话内容
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    dialogue_type = db.Column(db.String(50), nullable=True)
    scene_type = db.Column(db.String(20), nullable=False, default="outpatient") 

class Emotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    emotions = db.Column(JSON, nullable=False)  # 存储情绪列表
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class ModelAOutput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    feedback = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class FeedbackRating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    rating_type = db.Column(db.String(50), nullable=False)  # 评分类型：patient_satisfaction, doctor_satisfaction, follow_up_intention, doctor_burden
    rating = db.Column(db.Integer, nullable=False)  # 1-5 星评分
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class FeedbackSuggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    suggestion = db.Column(db.Text, nullable=False)  # 改进建议
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AudioRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)  # 关联的对话 ID
    audio_filename = db.Column(db.String(100), nullable=False)  # 音频文件名
    transcription = db.Column(db.Text, nullable=False)  # 最终输入的文本
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # 记录时间
    
class DemographicInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
class BackgroundAudioRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(50), nullable=False)  # 关联的对话 ID
    audio_filename = db.Column(db.String(100), nullable=False)  # 音频文件名
    transcription = db.Column(db.Text, nullable=False)  # 最终输入的文本
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # 记录时间

# 创建数据库表
with app.app_context():
    db.create_all()

# 初始化 DeepSeek 客户端
deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 加载情感模型与音频转录模型
emotion_model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
audio_model = AutoModel(model= audio_model_path,disable_update=True,language="zh")

# 加载 RAG 模型和分词器
rag_model_path = config.RAG_MODEL_PATH  # RAG 模型路径
rag_tokenizer = AutoTokenizer.from_pretrained(rag_model_path)
rag_model = AutoModelForMaskedLM.from_pretrained(rag_model_path)

# 读取知识库 Excel 文件
knowledge_file_path = config.RAG_KNOWLEDGE_PATH  # 知识库文件路径
knowledge_df = pd.read_excel(knowledge_file_path)
knowledge = knowledge_df['内容'].tolist()  # 知识库的列名为 "内容"

def split_text_into_chunks(text, max_length=512):
    tokens = rag_tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        if current_length >= max_length:
            chunk_text = rag_tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunk_text = rag_tokenizer.convert_tokens_to_string(current_chunk)
        chunks.append(chunk_text)

    return chunks

def precompute_knowledge_embeddings(knowledge):
    embeddings = []
    for text in knowledge[1:]:
        chunks = split_text_into_chunks(text)
        chunk_embeddings = []
        for chunk in chunks:
            inputs = rag_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = rag_model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
            chunk_embeddings.append(embedding)
        embeddings.append(chunk_embeddings)
    return embeddings

def get_high_knowledge(query, knowledge, k):
    inputs_query = rag_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs_query = rag_model(**inputs_query, output_hidden_states=True)
        query_embedding = outputs_query.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()

    similarities = []
    for chunk_embeddings in knowledge_embeddings:
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
    top_k_texts = [knowledge[idx + 1] for idx in top_k_idx]

    for i, idx in enumerate(top_k_idx):
        print(f"Top {i+1} Text: {knowledge[idx + 1]} (Similarity: {similarities[idx]})")

    return top_k_texts

knowledge_embeddings = precompute_knowledge_embeddings(knowledge)
print(f"已完成知识库向量转换")

def call_chatanywhere(input_text, context=[]):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer sk-yhGP21PPtDMiB8ptAow3zDn3pHQk46j5D2N34iXyspqvfCOG',
        }

        # 构建消息格式
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
        print(f"API调用失败，状态码：{response.status_code}")
        return None
    except Exception as e:
        print(f"调用ChatAnywhere API时发生错误：{e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

def transcribe_audio(file_path):
    try:
        if file_path.endswith('.webm'):
            wav_path = file_path.replace('.webm', '.wav')
            audio = AudioSegment.from_file(file_path, format="webm")
            audio.export(wav_path, format="wav")
            file_path = wav_path

        # 使用 pydub 对音频文件进行分块处理
        audio = AudioSegment.from_wav(file_path)
        chunk_length = 60 * 1000  # 60 秒
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

        transcriptions = []
        for idx, chunk in enumerate(chunks):
            # 保存分块后的音频
            chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"chunk_{idx}.wav")
            chunk.export(chunk_path, format="wav")

            # 使用 funasr 转录每个片段
            res = audio_model.generate(input=chunk_path, batch_size_s=300, batch_size_threshold_s=60)
            transcriptions.append(res[0]['text'] if res else "")

            # 删除临时文件
            os.remove(chunk_path)

        return "\n".join(transcriptions)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def process_audio(file_path):
    if file_path.endswith('.webm'):
        wav_path = file_path.replace('.webm', '.wav')
        audio = AudioSegment.from_file(file_path, format="webm")
        audio.export(wav_path, format="wav")
        file_path = wav_path

    waveform, sample_rate = torchaudio.load(file_path)
    res = emotion_model.generate(waveform, granularity="utterance", extract_embedding=False)
    for item in res:
        labels = item['labels']
        scores = item['scores']
        return [{"emotion": lbl, "score": scr} for lbl, scr in zip(labels, scores)]
    return []

def call_deepseek(input_text, role="user", context=[]):
    try:
        context.append({"role": role, "content": input_text})

        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=context,
            stream=False
        )

        model_reply = response.choices[0].message.content
        context.append({"role": "assistant", "content": model_reply})

        return model_reply
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return None

@app.route('/save-demographic-info', methods=['POST'])
def save_demographic_info():
    data = request.json
    print("收到人口学信息数据:", data)

    if not data:
        return jsonify({"status": "error", "message": "请求数据为空"}), 400

    if 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400
    
    if 'name' not in data:
        return jsonify({"status": "error", "message": "缺少 name 字段"}),    

    if 'age' not in data:
        return jsonify({"status": "error", "message": "缺少 age 字段"}), 400

    if 'gender' not in data:
        return jsonify({"status": "error", "message": "缺少 gender 字段"}), 400

    new_demographic_info = DemographicInfo(
        conversation_id=data['conversation_id'],
        name=data['name'],
        age=data['age'],
        gender=data['gender']
    )
    db.session.add(new_demographic_info)
    db.session.commit()
    print("人口学信息已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/get-demographic-info', methods=['POST'])
def get_demographic_info():
    data = request.json
    print("收到请求数据:", data)

    if not data:
        return jsonify({"status": "error", "message": "请求数据为空"}), 400

    if 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

    demographic_info = DemographicInfo.query.filter_by(conversation_id=data['conversation_id']).first()
    if demographic_info:
        return jsonify({
            "status": "success",
            "age": demographic_info.age,
            "gender": demographic_info.gender
        })
    else:
        return jsonify({"status": "error", "message": "未找到人口学信息"}), 404

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    audio = request.files.get('audio')
    conversation_id = request.form.get('conversation_id')
    
    if not audio:
        return jsonify({"error": "未上传音频文件"}), 400

    # 保存音频文件
    audio_filename = f"background_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    audio.save(audio_path)

    # 转录音频
    transcription = transcribe_audio(audio_path)
    
    # 调用 chatanywhere API 生成总结
    context = [{
        "role": "system",
        "content": """
            你是一个专业医疗助手，负责根据患者和医生的对话文本总结背景信息。请根据音频内容，生成一个简洁的总结作为对话的开头。请注意，对话文本没有标点符号，并且无法识别是患者还是医生说的话，你需要仔细甄别，并给出一个50到100字的总结。
        """
    }]
    
    summary = call_chatanywhere(f"音频内容: {transcription}", context=context)
    
    if not summary:
        return jsonify({"error": "生成总结失败"}), 500
    
    # 保存总结到数据库
    new_conversation = Conversation(
        conversation_id=conversation_id,
        role='user',
        content=summary,
        dialogue_type='summary'
    )
    # 保存背景音频记录
    new_background_audio = BackgroundAudioRecord(
        conversation_id=conversation_id,
        audio_filename=audio_filename,
        transcription=transcription
    )
    db.session.add(new_background_audio)
    db.session.add(new_conversation)
    db.session.commit()

    return jsonify({
        "summary": summary,
        "audio_filename": audio_filename
    })

@app.route('/save-conversation', methods=['POST'])
def save_conversation():
    try:
        data = request.json
        print("收到数据:", data)

        if not data:
            return jsonify({"status": "error", "message": "请求数据为空"}), 400

        if 'conversation_id' not in data:
            return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

        if 'role' not in data:
            return jsonify({"status": "error", "message": "缺少 role 字段"}), 400

        if 'content' not in data:
            return jsonify({"status": "error", "message": "缺少 content 字段"}), 400

        new_conversation = Conversation(
            conversation_id=data['conversation_id'],
            role=data['role'],
            content=data['content'],
            dialogue_type=data.get('dialogue_type'),
            scene_type=data.get('scene_type')
        )
        db.session.add(new_conversation)
        db.session.commit()
        print("数据已保存到数据库")
        return jsonify({"status": "success"})

    except Exception as e:
        print(f"保存对话内容时发生错误: {e}")
        return jsonify({"status": "error", "message": "服务器内部错误"}), 500

@app.route('/save-emotion', methods=['POST'])
def save_emotion():
    data = request.json
    print("收到数据:", data)
    new_emotion = Emotion(
        conversation_id=data['conversation_id'],
        emotions=data['emotions']
    )
    db.session.add(new_emotion)
    db.session.commit()
    print("数据已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/save-model-a-output', methods=['POST'])
def save_model_a_output():
    data = request.json
    print("收到数据:", data)

    if not data or 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

    if 'content' not in data:
        return jsonify({"status": "error", "message": "缺少 content 字段"}), 400

    new_output = ModelAOutput(
        conversation_id=data['conversation_id'],
        content=data['content']
    )
    db.session.add(new_output)
    db.session.commit()
    print("数据已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/save-audio-record', methods=['POST'])
def save_audio_record():
    data = request.json
    print("收到音频记录数据:", data)

    if not data or 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

    if 'audio_filename' not in data:
        return jsonify({"status": "error", "message": "缺少 audio_filename 字段"}), 400

    if 'transcription' not in data:
        return jsonify({"status": "error", "message": "缺少 transcription 字段"}), 400

    new_audio_record = AudioRecord(
        conversation_id=data['conversation_id'],
        audio_filename=data['audio_filename'],
        transcription=data['transcription']
    )
    db.session.add(new_audio_record)
    db.session.commit()
    print("音频记录已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/save-feedback', methods=['POST'])
def save_feedback():
    data = request.json
    print("收到数据:", data)
    new_feedback = Feedback(
        conversation_id=data['conversation_id'],
        content=data['content'],
        feedback=data['feedback']
    )
    db.session.add(new_feedback)
    db.session.commit()
    print("数据已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/save-rating', methods=['POST'])
def save_rating():
    data = request.json
    print("收到评分数据:", data)

    if not data:
        return jsonify({"status": "error", "message": "请求数据为空"}), 400

    if 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

    if 'rating' not in data:
        return jsonify({"status": "error", "message": "缺少 rating 字段"}), 400

    new_rating = FeedbackRating(
        conversation_id=data['conversation_id'],
        rating_type=data['rating_type'],
        rating=data['rating']
    )
    db.session.add(new_rating)
    db.session.commit()
    print("评分已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/save-suggestion', methods=['POST'])
def save_suggestion():
    data = request.json
    print("收到改进建议数据:", data)

    if not data:
        return jsonify({"status": "error", "message": "请求数据为空"}), 400

    if 'conversation_id' not in data:
        return jsonify({"status": "error", "message": "缺少 conversation_id 字段"}), 400

    if 'suggestion' not in data:
        return jsonify({"status": "error", "message": "缺少 suggestion 字段"}), 400

    new_suggestion = FeedbackSuggestion(
        conversation_id=data['conversation_id'],
        suggestion=data['suggestion']
    )
    db.session.add(new_suggestion)
    db.session.commit()
    print("改进建议已保存到数据库")
    return jsonify({"status": "success"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio = request.files.get('audio')
    if not audio:
        return jsonify({"error": "未上传音频文件"}), 400

    audio_filename = audio.filename
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    audio.save(audio_path)
    print(f"音频文件已保存到: {audio_path}")

    transcription = transcribe_audio(audio_path)
    if not transcription:
        return jsonify({"error": "语音转写失败"}), 500

    return jsonify({
        "transcription": transcription,
        "audio_filename": audio_filename
    })

@app.route('/chat', methods=['POST'])
def chat():
    audio = request.files.get('audio')
    message = request.form.get('message')
    conversation_id = request.form.get('conversation_id')
    if not audio or not message or not conversation_id:
        return jsonify({"error": "音频、文本和对话ID均为必填项"}), 400

    summary = request.form.get('summary')
    
    audio_filename = f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    audio.save(audio_path)

    emotions = process_audio(audio_path)
    if not emotions:
        return jsonify({"error": "情感识别失败"}), 500

    new_emotion = Emotion(
        conversation_id=conversation_id,
        emotions=emotions
    )
    db.session.add(new_emotion)
    db.session.commit()

    # 修改这里：固定为 dolphin 模式，移除双盲逻辑
    dialogue_type = "dolphin"
    print(f"对话类型: {dialogue_type}")

    Conversation.query.filter_by(conversation_id=conversation_id).update({"dialogue_type": dialogue_type})
    db.session.commit()

    rag_knowledge = get_high_knowledge(message, knowledge, k=10)
    rag_knowledge_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(rag_knowledge)])

    context_a = [{
        "role": "system",
        "content": """
            你是一个富有洞察力的医疗助手，你在医院工作，专门分析医患对话中的情感和潜在含义。每当患者进行交流时，会有模型来识别其音频和文字中代表的情感（及可能性），你的任务是根据其文字含义与音频文字情感的差距，来分析患者话语所要表达的情感以及其深层的语义。在分析情感和语义后，简短准确地给出如何回复患者问题的建议（50字以内）。
            你的回复必须严格按照以下 Markdown 格式输出：
            患者内容:{患者输入的内容}
            情感:{情感分析结果，挑选概率最高的3个情感去说明}
            语义:{语义分析结果}
            形象分析:{分析患者的形象特点以及其需求，给出综合理解}
            回复指南:{对回复患者这个问题建议}
            示例：
            患者内容:你好
            情感:中性: 0.85，快乐：0.10，悲伤：0.02
            语义:患者想打招呼询问问题。
            形象分析：患者希望引起你的注意
            回复指南:请及时回复并给予支持。
        """
    }]
    
    # 修改这里：统一使用带情感分析的 dolphin 模式
    a1 = call_chatanywhere(f"背景总结: {summary}，患者内容: {message}, 情感: {emotions}", context=context_a)

    new_audio_record = AudioRecord(
        conversation_id=conversation_id,
        audio_filename=audio_filename,
        transcription=message
    )
    db.session.add(new_audio_record)
    db.session.commit()

    return jsonify({
        "status": "success",
        "model_a_output": a1,
        "rag_knowledge": rag_knowledge_str,
        "audio_filename": audio_filename,
        "dialogue_type": dialogue_type
    })

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.json
    conversation_id = data.get('conversation_id')
    model_a_output = data.get('model_a_output')
    feedback = data.get('feedback')
    original_message = data.get('original_message')
    summary = request.form.get('summary')

    if not conversation_id or not model_a_output or not feedback or not original_message:
        return jsonify({"error": "缺少必要参数"}), 400

    history = Conversation.query.filter_by(conversation_id=conversation_id).order_by(Conversation.timestamp).all()
    messages = []
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
        
    rag_knowledge = get_high_knowledge(original_message, knowledge, k=5)
    rag_knowledge_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(rag_knowledge)])

    # 修改这里：固定为 dolphin 模式
    dialogue_type = "dolphin"

    if feedback == 'like':
        # 修改这里：统一使用带情感分析的 dolphin 模式
        messages.append({"role": "user", "content": f"背景总结: {summary};理解: {model_a_output};RAG知识: {rag_knowledge_str}"})

        context_b = [
            {"role": "system", "content": "你是一名在医院工作的具有同理心和专业素养的医疗助手，你是帮助门诊医生的，专门为患者提供情感支持和专业医疗咨询。在接收到情感分析结果后，你需要根据患者的情绪状态和潜在心理需求，进行具有针对性且符合医学常识、具有良好医学素养的对话回应。并且，你的回复应该尽可能简洁明确。请注意，你目前就是一名医生的助手，所以绝对不应该说出类似请及时就医请尽快前往医院这样的话，更应该去给它们专业的医学建议。"}
        ]
        context_b.extend(messages)

        b1 = call_chatanywhere(f"患者说话内容: {original_message}", context=context_b)

        new_assistant_message = Conversation(
            conversation_id=conversation_id,
            role='assistant',
            content=b1,
            dialogue_type=dialogue_type
        )
        db.session.add(new_assistant_message)
        db.session.commit()

        return jsonify({
            "model_b_output": b1
        })
    elif feedback == 'dislike':
        context_a = [{
            "role": "system",
            "content": """
                （此处内容保持不变）
            """
        }]
        context_a.extend(messages)

        a1 = call_chatanywhere(f"背景总结: {summary};患者内容: {model_a_output}", context=context_a)

        return jsonify({
            "model_a_output": a1
        })
    else:
        return jsonify({"error": "无效的反馈类型"}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)