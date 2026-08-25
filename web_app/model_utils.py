"""
PyTorch Neural Network Models and Inference Utilities for CDSS Web App
Loads TransMIL God Mode, BiLSTM Multimodal, and performs real-time PAM50 diagnosis.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# 1. TRANSMIL SOTA ARCHITECTURE (NYSTROM ATTENTION)
# =========================================================================
class NystromAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_landmarks=256, pinv_iterations=6):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = self.dim_head ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.num_heads
        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, self.dim_head).transpose(1, 2)
        m = min(self.num_landmarks, n)
        q_landmarks = F.adaptive_avg_pool1d(q.reshape(b*h, self.dim_head, n), m).reshape(b, h, self.dim_head, m).transpose(2, 3)
        k_landmarks = F.adaptive_avg_pool1d(k.reshape(b*h, self.dim_head, n), m).reshape(b, h, self.dim_head, m).transpose(2, 3)
        
        kernel_1 = F.softmax(torch.matmul(q, k_landmarks.transpose(-1, -2)) * self.scale, dim=-1)
        kernel_2 = F.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)) * self.scale, dim=-1)
        kernel_3 = F.softmax(torch.matmul(q_landmarks, k.transpose(-1, -2)) * self.scale, dim=-1)
        
        z = kernel_2
        v_norm = torch.norm(z, p=float('inf'), dim=(-2, -1), keepdim=True) * torch.norm(z, p=1, dim=(-2, -1), keepdim=True)
        v_mat = z.transpose(-1, -2) / (v_norm + 1e-6)
        for _ in range(self.pinv_iterations):
            v_mat = 2 * v_mat - torch.matmul(v_mat, torch.matmul(z, v_mat))
        
        out = torch.matmul(kernel_1, torch.matmul(v_mat, torch.matmul(kernel_3, v)))
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)

class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim)
    def forward(self, x):
        return x + self.attn(self.norm(x))

class TransMIL_Vision_Extractor(nn.Module):
    def __init__(self, input_dim=2048, out_dim=512):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        
    def forward(self, x):
        h = self.fc1(x.float())
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        return h[:, 0], h[:, 1:] # Return CLS token and patch sequence representations

class Multimodal_GenomicsFusion(nn.Module):
    def __init__(self, genomics_dim=500, num_classes=4):
        super().__init__()
        self.vision_net = TransMIL_Vision_Extractor()
        self.genomics_net = nn.Sequential(
            nn.Linear(genomics_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, img_feat, gen_feat):
        v_img, patch_embeds = self.vision_net(img_feat)
        v_gen = self.genomics_net(gen_feat)
        v_fusion = torch.cat((v_img, v_gen), dim=1)
        logits = self.classifier(v_fusion)
        return logits, v_fusion, v_img, v_gen, patch_embeds


# =========================================================================
# 2. BILSTM MULTIMODAL ARCHITECTURE
# =========================================================================
class BiLSTM_Vision_Extractor(nn.Module):
    def __init__(self, input_dim=2048, out_dim=512):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
    def forward(self, x):
        h = self.fc1(x.float())
        lstm_out, _ = self.lstm(h)
        return lstm_out.mean(dim=1)

class Multimodal_LSTM_Genomics(nn.Module):
    def __init__(self, genomics_dim=500, num_classes=4):
        super().__init__()
        self.vision_net = BiLSTM_Vision_Extractor()
        self.genomics_net = nn.Sequential(
            nn.Linear(genomics_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, img_feat, gen_feat):
        v_img = self.vision_net(img_feat)
        v_gen = self.genomics_net(gen_feat)
        v_fusion = torch.cat((v_img, v_gen), dim=1)
        return self.classifier(v_fusion)


# =========================================================================
# 3. RESOURCE CACHING & LOADERS
# =========================================================================
LABEL_MAP = {0: 'Luminal A', 1: 'Luminal B', 2: 'Basal-like', 3: 'HER2-enriched'}
SUBTYPE_KEY_MAP = {'BRCA_LumA': 'Luminal A', 'BRCA_LumB': 'Luminal B', 'BRCA_Basal': 'Basal-like', 'BRCA_Her2': 'HER2-enriched'}

@st.cache_resource
def load_models_and_scalers(base_dir):
    """Loads all PyTorch, Scikit-learn, and Lifelines models into cache."""
    model_dir = os.path.join(base_dir, 'model')
    
    # 1. TransMIL God Mode
    model_transmil = Multimodal_GenomicsFusion().to(device)
    w_transmil = os.path.join(model_dir, 'multimodal_genomics_best.pth')
    if os.path.exists(w_transmil):
        st_dict = torch.load(w_transmil, map_location=device)
        model_transmil.load_state_dict({k.replace('module.', ''): v for k, v in st_dict.items()}, strict=False)
    model_transmil.eval()
    
    # 2. BiLSTM Multimodal
    model_bilstm = Multimodal_LSTM_Genomics().to(device)
    w_bilstm = os.path.join(model_dir, 'multimodal_bilstm_best.pth')
    if os.path.exists(w_bilstm):
        st_dict_lstm = torch.load(w_bilstm, map_location=device)
        model_bilstm.load_state_dict({k.replace('module.', ''): v for k, v in st_dict_lstm.items()}, strict=False)
    model_bilstm.eval()
    
    # 3. PCA & CoxPH & Scaler
    pca = joblib.load(os.path.join(model_dir, 'pca_16_multimodal.joblib'))
    cph = joblib.load(os.path.join(model_dir, 'coxph_survival_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler_genomics_500.joblib'))
    
    # 4. ResNet50 Feature Extractor for on-the-spot SVS patch embedding
    import torchvision.models as tv_models
    resnet_extractor = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
    resnet_extractor.fc = nn.Identity()
    resnet_extractor.eval().to(device)
    
    return {
        'transmil': model_transmil,
        'bilstm': model_bilstm,
        'pca': pca,
        'coxph': cph,
        'scaler': scaler,
        'resnet': resnet_extractor
    }


def extract_resnet50_features_from_patches(patches, resnet_model):
    """
    Extracts a (1, N, 2048) feature tensor from a list of PIL patch images in memory.
    """
    import torchvision.transforms as tv_transforms
    transform = tv_transforms.Compose([
        tv_transforms.Resize(224),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch = torch.stack([transform(p) for p in patches]).to(device)
    with torch.no_grad():
        wsi_features = resnet_model(batch).unsqueeze(0) # (1, N, 2048)
    return wsi_features

@st.cache_data
def load_cohort_data(base_dir):
    """Loads clinical records, RNA-Seq vectors, manifest, and t-SNE coordinates."""
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    # Clinical
    df_clin = pd.read_csv(os.path.join(data_dir, 'clinical', 'tcga_brca_master_matched_cohort.csv'))
    df_overview = pd.read_csv(os.path.join(data_dir, 'clinical', 'cohort_inference_overview.csv'))
    
    # Genomics
    rna_500 = pd.read_csv(os.path.join(data_dir, 'genomics', 'tcga_brca_rna_500_scaled.csv'), index_col=0)
    with open(os.path.join(data_dir, 'genomics', 'top_500_gene_names.json'), 'r', encoding='utf-8') as f:
        top_500_genes = json.load(f)
    with open(os.path.join(data_dir, 'genomics', 'gene_biomarkers_summary.json'), 'r', encoding='utf-8') as f:
        gene_stats = json.load(f)
        
    # Manifest
    with open(os.path.join(output_dir, 'demo_cases_manifest.json'), 'r', encoding='utf-8') as f:
        demo_manifest = json.load(f)
        
    # t-SNE
    df_tsne = pd.read_csv(os.path.join(output_dir, 'cohort_tsne_coordinates.csv'))
    
    return {
        'clinical': df_clin,
        'overview': df_overview,
        'rna_500': rna_500,
        'gene_names': top_500_genes,
        'gene_stats': gene_stats,
        'manifest': demo_manifest,
        'tsne': df_tsne
    }


# =========================================================================
# 4. INFERENCE ENGINE
# =========================================================================
def run_transmil_inference(model, img_source, gen_vector):
    """
    Runs real-time inference on a patient's WSI (either .pt path or direct torch.Tensor) and 500-gene vector.
    Returns:
        pred_label (int), pred_name (str), confidence (float), probs (list),
        v_fusion (np.ndarray 1024D), patch_attentions (np.ndarray)
    """
    if isinstance(img_source, torch.Tensor):
        img_tensor = img_source.to(device)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        img_tensor = torch.load(img_source, map_location=device)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
    if isinstance(gen_vector, torch.Tensor):
        gen_tensor = gen_vector.to(device)
    else:
        gen_tensor = torch.tensor(gen_vector, dtype=torch.float32).to(device)
        
    while gen_tensor.ndim < 2:
        gen_tensor = gen_tensor.unsqueeze(0)
    while gen_tensor.ndim > 2:
        gen_tensor = gen_tensor.squeeze(1)
    
    with torch.no_grad():
        logits, v_fusion, v_img, v_gen, patch_embeds = model(img_tensor, gen_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        
        # Calculate Attention Similarity (CLS token vs. Patch representations)
        # Cosine similarity gives the patch attention saliency
        cls_vec = v_img # (1, 512)
        p_embeds = patch_embeds.squeeze(0) # (num_patches, 512)
        sim = F.cosine_similarity(cls_vec, p_embeds, dim=1).cpu().numpy()
        # Min-max scale to [0, 1]
        attn_scores = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        
    return {
        'pred_label': pred_label,
        'pred_name': LABEL_MAP[pred_label],
        'confidence': confidence,
        'probs': {LABEL_MAP[i]: float(probs[i]) for i in range(4)},
        'fusion_vector': v_fusion.cpu().numpy()[0],
        'patch_attentions': attn_scores,
        'num_patches': img_tensor.shape[1]
    }


def preprocess_raw_20k_rna(df_raw, top_500_genes, scaler):
    """
    Takes a 1-row DataFrame of ~20k raw RSEM gene counts, extracts Top 500 genes,
    and standard-scales with the pre-fitted cohort Scaler (Z-Score).
    Returns:
        scaled_vector (np.ndarray of shape (500,)), scaled_series (pd.Series)
    """
    raw_500 = np.zeros((1, len(top_500_genes)))
    col_map = {str(c).strip().upper(): c for c in df_raw.columns}
    
    for i, g in enumerate(top_500_genes):
        g_upper = g.strip().upper()
        if g_upper in col_map:
            actual_col = col_map[g_upper]
            try:
                raw_500[0, i] = float(df_raw[actual_col].values[0])
            except Exception:
                raw_500[0, i] = 0.0
                
    df_500 = pd.DataFrame(raw_500, columns=top_500_genes)
    scaled_500 = scaler.transform(df_500)
    return scaled_500[0], pd.Series(scaled_500[0], index=top_500_genes)


def get_external_demo_pairs(base_dir):
    """
    Finds available external SVS slides and their matched 1-row 20k RNA-Seq CSVs.
    """
    slides_dir = r"C:\Users\huynh\Desktop\slides"
    demo_dir = os.path.join(base_dir, 'data', 'demo_external_pairs')
    
    pairs = []
    if os.path.exists(slides_dir):
        svs_files = [f for f in os.listdir(slides_dir) if f.lower().endswith('.svs')]
        for f in svs_files:
            # Extract PID
            import re
            m = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', f)
            pid = m.group(1) if m else f[:12]
            
            # Check matching RNA CSV
            csv_name = f"{pid}_rna_20k.csv"
            csv_path1 = os.path.join(slides_dir, "demo_rna_csv", csv_name)
            csv_path2 = os.path.join(demo_dir, csv_name)
            
            chosen_csv = csv_path1 if os.path.exists(csv_path1) else (csv_path2 if os.path.exists(csv_path2) else None)
            
            pairs.append({
                'patient_id': pid,
                'svs_filename': f,
                'svs_path': os.path.join(slides_dir, f),
                'csv_filename': csv_name,
                'csv_path': chosen_csv,
                'has_rna': chosen_csv is not None
            })
    return pairs
