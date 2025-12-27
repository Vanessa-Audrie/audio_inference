import streamlit as st
import torch
import torch.nn as nn
from transformers import HubertModel
from huggingface_hub import hf_hub_download
import librosa
import numpy as np
import pickle
import json
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import librosa.display

# Page config
st.set_page_config(
    page_title="Audio Emotion & Stress Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_REPO_ID = "vanessaaudrie/audio-emotion-stress-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get HF token from secrets
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("❌ HF_TOKEN not found in secrets!")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    h1 {
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 600;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: #1e88e5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    [data-testid="stFileUploader"] {
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: 600;
    }
    
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .result-title {
        color: white;
        margin: 0;
        font-size: 2em;
        font-weight: 600;
    }
    
    .result-subtitle {
        color: rgba(255, 255, 255, 0.9);
        margin: 10px 0 0 0;
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Emotion to emoji mapping
EMOTION_EMOJI = {
    'marah': '😡',
    'takut': '😨',
    'senang': '😁',
    'sedih': '😢',
    'netral': '😐',
}

# Stress to emoji mapping
STRESS_EMOJI = {
    'tidakstress': '😀',
    'stress': '😞',
}

# Model class
class FinalHuBERT(nn.Module):
    def __init__(self, hubert_model, num_emotion_classes, num_stress_classes, config):
        super(FinalHuBERT, self).__init__()
        self.hubert = hubert_model
        self.hidden_size = hubert_model.config.hidden_size
        
        self.shared_layer = nn.Sequential(
            nn.Linear(self.hidden_size, config['hidden_dim_1']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim_1'], config['hidden_dim_2']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(config['hidden_dim_2'], 128),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.7),
            nn.Linear(128, num_emotion_classes)
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(config['hidden_dim_2'], 128),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.7),
            nn.Linear(128, num_stress_classes)
        )
    
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        shared_repr = self.shared_layer(pooled)
        emotion_logits = self.emotion_head(shared_repr)
        stress_logits = self.stress_head(shared_repr)
        return emotion_logits, stress_logits

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        progress_container = st.empty()
        
        with progress_container.container():
            st.info("Loading model")
            progress_bar = st.progress(0)
            
            progress_bar.progress(10)
            st.text("Downloading encoders...")
            encoders_path = hf_hub_download(repo_id=HF_REPO_ID, filename="encoders.pkl", token=HF_TOKEN)
            
            progress_bar.progress(20)
            st.text("Downloading config...")
            config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_config.json", token=HF_TOKEN)
            
            progress_bar.progress(30)
            st.text("Downloading class info...")
            class_info_path = hf_hub_download(repo_id=HF_REPO_ID, filename="class_info.json", token=HF_TOKEN)
            
            progress_bar.progress(40)
            st.text("Downloading model weights (1.06 GB)")
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_hubert_model.pth", token=HF_TOKEN)
            
            progress_bar.progress(60)
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            emotion_encoder = encoders['emotion_encoder']
            stress_encoder = encoders['stress_encoder']
            
            progress_bar.progress(70)
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            
            progress_bar.progress(80)
            hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            
            progress_bar.progress(90)
            model = FinalHuBERT(
                hubert_model=hubert_model,
                num_emotion_classes=class_info['num_emotion_classes'],
                num_stress_classes=class_info['num_stress_classes'],
                config=config
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            progress_bar.progress(100)
        
        progress_container.empty()
        return model, emotion_encoder, stress_encoder, config
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

def load_audio_resampled(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr

def chunk_audio(audio, sr=16000, chunk_duration=5.0, overlap=1.0):
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    for start in range(0, len(audio), step_samples):
        end = start + chunk_samples
        if end > len(audio):
            chunk = audio[start:]
            padding = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, padding), mode='constant')
        else:
            chunk = audio[start:end]
        chunks.append(chunk)
        if end > len(audio):
            break
    return chunks

def predict_audio(model, audio, emotion_encoder, stress_encoder, device):
    chunks = chunk_audio(audio)
    emotion_preds, stress_preds = [], []
    emotion_probs_list, stress_probs_list = [], []
    
    with torch.no_grad():
        for chunk in chunks:
            audio_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
            emotion_logits, stress_logits = model(audio_tensor)
            
            emotion_probs = torch.softmax(emotion_logits, dim=1)
            stress_probs = torch.softmax(stress_logits, dim=1)
            
            emotion_preds.append(torch.argmax(emotion_probs, dim=1).cpu().numpy()[0])
            stress_preds.append(torch.argmax(stress_probs, dim=1).cpu().numpy()[0])
            emotion_probs_list.append(emotion_probs.cpu().numpy()[0])
            stress_probs_list.append(stress_probs.cpu().numpy()[0])
    
    emotion_final = max(set(emotion_preds), key=emotion_preds.count)
    stress_final = max(set(stress_preds), key=stress_preds.count)
    
    return {
        'emotion': emotion_encoder.inverse_transform([emotion_final])[0],
        'stress': stress_encoder.inverse_transform([stress_final])[0],
        'emotion_probs': np.mean(emotion_probs_list, axis=0),
        'stress_probs': np.mean(stress_probs_list, axis=0),
        'emotion_classes': emotion_encoder.classes_,
        'stress_classes': stress_encoder.classes_,
        'num_chunks': len(chunks)
    }

def main():
    # Title
    st.markdown("<h1>Audio Emotion & Stress Classifier</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Klasifikasi emosi and deteksi stres melalui audio menggunakan model HuBERT</p>",
        unsafe_allow_html=True
    )
    
    # Load model
    model, emotion_encoder, stress_encoder, config = load_model()
    
    if model is None:
        st.stop()
    
    # Model info in main area
    st.success("Model ready for inference")
    st.metric("Compute Device", str(device).upper())
    
    # Display classes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kelas Emotion")
        st.markdown("""
        - Marah
        - Senang
        - Sedih
        - Takut
        - Netral
        """)
    
    with col2:
        st.subheader("Kelas Stress")
        st.markdown("""
        - Stress
        - Tidak Stress
        """)
    
    # Main content
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Pilih audio file yang Ingin di analisis:",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Supported formats: WAV, MP3, OGG, FLAC"
        )
    
    with col2:
        st.markdown("### Tips")
        st.markdown("""
        - Gunakan audio yang jernih dan kualitasnya bagus 
        - Minimize background noise
        - Pastikan suara terdengar jelas
        """)
    
    if uploaded_file:
        st.markdown("---")
        
        # File information
        st.markdown("### Informasi File Yang Diupload")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filename", uploaded_file.name)
        col2.metric("Format", uploaded_file.type.split('/')[-1].upper())
        col3.metric("Size", f"{uploaded_file.size/1024:.2f} KB")
        
        # Audio player
        st.markdown("### Audio Preview")
        st.audio(uploaded_file)
        
        # Analyze button
        st.markdown("---")
        if st.button("Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load and predict
                    audio, sr = load_audio_resampled(tmp_path)
                    results = predict_audio(model, audio, emotion_encoder, stress_encoder, device)
                    
                    # Display results
                    st.success("Analysis Selesai")
                    st.markdown("---")

                    st.markdown("### Detail")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Durasi", f"{len(audio)/sr:.2f}s")
                    col2.metric("Sample Rate", f"{sr} Hz")
                    col3.metric("Chunks", results['num_chunks'])

                    st.markdown("---")
                    st.markdown("## Hasil Klasifikasi")
                    
                    # Main predictions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Emosi")
                        
                        # Get emoji for emotion
                        emotion_emoji = EMOTION_EMOJI.get(results['emotion'].lower(), '')
                        
                        st.markdown(f"""
                            <div class='result-card' style='background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
                                <h2 class='result-title'>{emotion_emoji} {results['emotion'].capitalize()}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### Confidence Distribution")
                        for cls, prob in zip(results['emotion_classes'], results['emotion_probs']):
                            cls_emoji = EMOTION_EMOJI.get(cls.lower(), '')
                            st.progress(float(prob), text=f"{cls_emoji} {cls.capitalize()}: {prob*100:.1f}%")
                    
                    with col2:
                        st.markdown("### Stress")
                        
                        # Get emoji for stress
                        stress_emoji = STRESS_EMOJI.get(results['stress'].lower(), '')
                        
                        st.markdown(f"""
                            <div class='result-card' style='background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
                                <h2 class='result-title'>{stress_emoji} {results['stress'].capitalize()}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### Confidence Distribution")
                        for cls, prob in zip(results['stress_classes'], results['stress_probs']):
                            cls_emoji = STRESS_EMOJI.get(cls.lower(), '')
                            st.progress(float(prob), text=f"{cls_emoji} {cls.capitalize()}: {prob*100:.1f}%")
                    
                    
                finally:
                    # Cleanup temp file
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass

if __name__ == "__main__":
    main()