# Multimodal Deep Learning for Breast Cancer Molecular Subtyping and Survival Prognosis (TCGA-BRCA)

## Technical Report and System Documentation

---

### Executive Summary

This repository documents the end-to-end development, empirical benchmarking, and clinical software deployment of a multimodal deep learning framework for breast invasive carcinoma (*TCGA-BRCA*). The system integrates gigapixel Whole Slide Images (WSI) and high-dimensional transcriptomic profiles (20,518-gene RNA-Seq) alongside clinical variables to address two core clinical objectives:

1. **PAM50 Molecular Subtyping (4-Class Classification):** Accurate identification of the four intrinsic breast cancer subtypes: Luminal A, Luminal B, HER2-enriched, and Basal-like (Triple-Negative).
2. **5-Year Overall Survival Prognosis & Risk Stratification:** Estimation of patient hazard ratios and generation of 5-year survival probability curves using a Multimodal Cox Proportional Hazards framework.

The experimental progression spans 22 research notebooks, tracking models from single-modality baseline classifiers (Traditional Machine Learning, Naive Deep MIL, TransMIL) to multimodal late-fusion architectures (TransMIL + Genomics MLP, BiLSTM + Genomics MLP). To enable practical evaluation, the trained models are integrated into an interactive Clinical Decision Support System (CDSS) built with Streamlit, supporting direct ingestion and on-the-spot inference of raw Aperio `.svs` gigapixel slides and 20k-gene RNA-Seq expression matrices.

---

### 1. Dataset & Cohort Characterization

The primary data source is the **The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA)** cohort, curated through GDC and cBioPortal.

#### 1.1. Matched Cohort Distribution
From the total TCGA-BRCA collection, **945 unique patients** possess complete, multi-way matched data (diagnostic WSI, RNA-Seq expression, and curated clinical survival endpoints).

| Characteristic | Cohort Value / Breakdown | Clinical Relevance |
| :--- | :--- | :--- |
| **Total Matched Patients** | 945 patients | Multi-way matched across all modalities |
| **PAM50 Subtype Breakdown** | • **Luminal A:** 499 (52.8%)<br>• **Luminal B:** 197 (20.8%)<br>• **Basal-like:** 171 (18.1%)<br>• **HER2-enriched:** 78 (8.3%) | Significant class imbalance reflecting real-world clinical epidemiological prevalence |
| **Median Patient Age** | 58.0 years (range: 26 – 90) | Standard post- and peri-menopausal breast cancer demographics |
| **Median Follow-up Time** | 28.5 months (range: 0.1 – 282.7 months) | Right-censored survival endpoint |
| **Overall Censoring Rate** | 85.2% censored (140 deaths observed) | Standard TCGA long-term follow-up censoring profile |
| **Histological Types** | • Infiltrating Ductal Carcinoma (IDC): ~78%<br>• Infiltrating Lobular Carcinoma (ILC): ~18%<br>• Other rare histologies: ~4% | Major histological sub-categories represented |

---

### 2. Data Processing Pipelines

```
                                  [ RAW MODALITIES ]
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
    [ Diagnostic WSI (.svs) ]   [ Transcriptomics (RNA-Seq) ]   [ Clinical Data ]
            │                             │                             │
    • Otsu Tissue Masking         • 20,518 Gene RSEM Counts     • Missing Imputation
    • Background Filtration       • Variance Ranking (Top 500)  • Follow-up / Censoring
    • 256x256 Grid Tiling         • StandardScaler (Z-Score)    • Stage Harmonization
    • ResNet-50 Feature Extraction        │                             │
            │                             │                             │
            ▼                             ▼                             ▼
    [ Bag: N x 2048D Tensor ]     [ 500D Gene Vector ]          [ Tabular Covariates ]
```

#### 2.1. Histopathology Pipeline (Whole Slide Images)
- **Slide Ingestion:** Aperio format (`.svs`) Whole Slide Images scanned at 20x (0.50 um/px) and 40x (0.25 um/px).
- **Tissue Segmentation:** RGB-to-HSV conversion, Otsu thresholding on the Saturation channel to eliminate glass background, marker pen artifacts, and mounting media defects.
- **Patch Extraction:** Non-overlapping tiling into 256x256 pixel patches. Patches with <40% tissue content or standard deviation <10 (blank/fat tissue) were discarded.
- **Visual Feature Encoding:** Patches were normalized using ImageNet channel parameters (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) and processed through a **ResNet-50** backbone truncated at the penultimate layer (`AdaptiveAvgPool2d`), yielding a 2048-dimensional embedding vector per patch. A slide is thus represented as a bag: `X_WSI in R^(N x 2048)`, where `N` varies between 500 and 4,000 patches depending on tumor specimen size.

#### 2.2. Transcriptomics Pipeline (RNA-Seq)
- **Input Matrix:** `data_mrna_seq_v2_rsem.txt` containing 20,518 genes across 1,082 samples (RNA-Seq V2 RSEM normalized counts).
- **Feature Selection:** Variance ranking was computed across the cohort. The top 500 genes with highest variance across the population were extracted. These capture key biological variance in intrinsic breast cancer oncogenes (e.g., *ESR1*, *PGR*, *ERBB2*, *MKI67*, *FOXA1*, *GATA3*, *KRT5*, *EGFR*) while discarding noisy invariant housekeeping genes (*ACTB*, *GAPDH*).
- **Standardization:** Evaluated both log2(x+1) transform and direct Z-score standardization. The optimal empirical separation was obtained via direct feature-wise Z-score scaling (`StandardScaler` fitted on the 500 genes), yielding a dense 500-dimensional vector `x_gen in R^500`.

#### 2.3. Clinical Covariates
- Features retained for baseline comparison and survival adjustment: age at diagnosis, AJCC pathologic tumor stage (Stage I, II, III, IV), surgical margin status, and lymph node involvement. Categorical variables were one-hot encoded; numerical variables were median-imputed and standardized.

---

### 3. Model Methodologies & Architectures

#### 3.1. Phase 1: Unimodal Baselines
1. **Traditional Machine Learning (WSI alone):** Slide-level mean-pooling of the `N x 2048` patch vectors into a single 2048D vector, followed by Logistic Regression, Random Forest, Support Vector Machines with RBF kernel, and XGBoost.
2. **Naive Deep MIL:** Deep multiple instance learning with simple permutation-invariant aggregation operators (Global Mean Pooling, Global Max Pooling).
3. **TransMIL (Transformer-based MIL):**
   - Applies Correlated Nyström Self-Attention ($O(N)$ computational complexity) to model long-range morphological dependencies between disparate tissue regions.
   - Prepends a learnable `[CLS]` token `z_cls` to aggregate slide-level representation.
   - Outputs a 512D morphological latent vector `v_img` alongside patch-level attention weights:

$$
a_i = \text{sim}(\mathbf{v}_{\text{img}}, \mathbf{h}_i), \quad i \in \{1, \dots, N\}
$$

#### 3.2. Phase 2: Multimodal Late Fusion Architecture
The primary production model integrates both modalities via late feature fusion:

```
[ WSI Patches: N x 2048D ] ────► TransMIL Backbone ────► 512D Latent Vector ──┐
                                                                              ├──► [ Concatenation: 1024D ] ────► Classifier (Dense 256 -> 4)
[ RNA-Seq: 500D Vector ]   ────► Genomics MLP      ────► 512D Latent Vector ──┘
```

- **Vision Stream:** TransMIL vision network processing the bag of `N` patch embeddings:
  `v_img = TransMIL(X) in R^512`
- **Genomics Stream:** Multi-Layer Perceptron (Dense 500 -> 256, LayerNorm, ReLU, Dropout 0.3, Dense 256 -> 512, ReLU) processing the 500-gene profile:
  `v_gen = MLP(x) in R^512`
- **Multimodal Fusion Layer:** Direct concatenation of histological and transcriptomic embeddings:
  `v_fusion = [v_img || v_gen] in R^1024`
- **Classification Head:** Dense 1024 -> 256, ReLU, Dropout 0.3, Dense 256 -> 4 (Softmax).
- **Loss Function:** Label-smoothed Cross-Entropy with class weighting to penalize minority-class misclassifications (HER2-enriched and Basal-like).

#### 3.3. Phase 3: BiLSTM Multimodal Comparison
To assess sequence modeling versus attention mechanisms for histological patch bags, an alternative architecture was evaluated:
- Patches were linearly projected to 512D and fed into a 2-layer Bidirectional LSTM (`hidden_dim = 256`), producing a 512D bidirectional sequence representation concatenated with the 512D genomic vector.

#### 3.4. Phase 4: Multimodal Survival Prognosis (Cox Proportional Hazards)
- The 1024-dimensional multimodal latent representations (`v_fusion`) were extracted from the trained model.
- Dimensionality reduction via Principal Component Analysis reduced the feature space to 16 orthogonal components capturing >88% of latent variance.
- A semi-parametric **Cox Proportional Hazards (CoxPH)** model with L2 regularization was fitted against overall survival $(T, E)$:

$$
h(t \mid \mathbf{z}) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{z})
$$

- Risk scores $\eta = \boldsymbol{\beta}^\top \mathbf{z}$ were stratified into three clinical tiers:
  - **Low Risk:** hazard score < 0.90 (Green)
  - **Borderline / Moderate Risk:** 0.90 <= hazard score <= 1.15 (Amber)
  - **High Risk:** hazard score > 1.15 (Red)
- Survival probability over a 60-month horizon was modeled using Breslow's estimator of the cumulative baseline hazard:

$$
S(t \mid \mathbf{z}) = \exp\left(-H_0(t) e^{\eta}\right)
$$

---

### 4. Experimental Results & Quantitative Benchmarking

All models were evaluated under stratified 5-fold cross-validation on identical patient splits to eliminate data leakage.

#### 4.1. Comparative Performance Table

| Model Family | Modalities | Architecture Details | Accuracy | Macro F1 | Clinical Takeaway |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Traditional ML** | WSI Alone | ResNet-50 Mean-Pool + Logistic Regression | 51.4% | 0.412 | Histology alone struggles to resolve LumA vs. LumB without marker data |
| **Traditional ML** | WSI Alone | ResNet-50 Mean-Pool + SVM (RBF) | 52.8% | 0.431 | Morphology lacks proliferation boundary signals |
| **Traditional ML** | WSI Alone | ResNet-50 Mean-Pool + XGBoost | 53.6% | 0.438 | Patch averaging loses focal tumor signals |
| **Deep MIL Baseline** | WSI Alone | Naive Attention-MIL (Ilse et al.) | 54.2% | 0.440 | Attention pooling provides slight gain over global mean pooling |
| **Transformer MIL** | WSI Alone | TransMIL (Nyström Attention) | 54.2% | 0.425 | Transformer attention on morphology alone still plateaus at ~0.44 F1 |
| **Clinical Baseline** | Clinical Alone | Age, Stage, Histology + Logistic Regression | 61.2% | 0.518 | Clinical staging provides marginal predictive power for molecular subtypes |
| **Multimodal Fusion** | WSI + Clinical | ResNet-50 Mean-Pool + Clinical + SVM | 69.1% | 0.586 | Adding clinical covariates improves accuracy to ~69% |
| **Multimodal Fusion** | WSI + Clinical | ResNet-50 Mean-Pool + Clinical + LogReg | 69.0% | 0.594 | Clear improvement over unimodal WSI |
| **Genomics Unimodal** | RNA-Seq Alone | 500-Gene MLP (Dense 500 -> 256 -> 512) | 85.3% | 0.832 | Gene expression directly reflects molecular PAM50 taxonomy |
| **Multimodal Fusion** | WSI + RNA-Seq | ResNet-50 Mean-Pool + RNA-Seq + XGBoost | 87.5% | 0.859 | High accuracy, but lacks spatial patch interpretability |
| **Multimodal Fusion** | WSI + RNA-Seq | BiLSTM (2-layer 512D) + Genomics MLP | 75.9% | 0.655 | Sequential ordering of patches adds noise compared to attention sets |
| **Multimodal TransMIL** | **WSI + RNA-Seq** | **TransMIL + Genomics MLP (Late Fusion)** | **88.4%** | **0.875** | **Optimal balance:** Highest F1, robust across all 4 classes, spatial attention retained |

#### 4.2. Survival Analysis Benchmark (C-Index)

| Prognostic Model | Features Used | Harrell's C-Index | 95% Confidence Interval | p-value |
| :--- | :--- | :---: | :---: | :---: |
| Clinical Stage Only (AJCC) | Stage I, II, III, IV | 0.612 | [0.558, 0.666] | p = 0.003 |
| Pathological Grade + Age | Histologic Grade + Age | 0.641 | [0.589, 0.693] | p < 0.001 |
| Unimodal WSI Latent (TransMIL) | 16 PCA Components of `v_img` | 0.654 | [0.601, 0.707] | p < 0.001 |
| Unimodal RNA Latent (500 Genes) | 16 PCA Components of `v_gen` | 0.718 | [0.667, 0.769] | p < 0.001 |
| **Multimodal Super-Vector + CoxPH** | **16 PCA Components of `v_fusion`** | **0.767** | **[0.720, 0.814]** | **p < 0.0001** |

*Key finding:* The combined multimodal latent space provides a **+15.5% absolute increase in concordance index** over clinical staging alone, confirming that paired morphology and transcriptomics encode independent prognostic risk factors.

---

### 5. Explainable AI (XAI) & Interpretability

To satisfy clinical verification standards, the framework operates dual interpretability mechanisms:

#### 5.1. Spatial Morphological Attention (WSI Domain)
- Extracted using cosine similarity between the TransMIL `[CLS]` token representation and each individual patch representation:

$$
\alpha_i = \frac{\mathbf{v}_{\text{img}}^\top \mathbf{h}_i}{\|\mathbf{v}_{\text{img}}\| \|\mathbf{h}_i\|}, \quad \alpha_i \in [0, 1]
$$

- **Pathological Correlation:** High-attention patches ($\alpha_i \ge 0.75$, highlighted with red diagnostic borders) correspond to areas of high cellularity, pleomorphic tumor cell clusters, and active mitotic figures. Low-attention patches ($\alpha_i < 0.25$, green borders) correspond to benign stroma, adipose connective tissue, and acellular necrosis.

#### 5.2. Transcriptomic Feature Attribution (Genomics Domain)
- Computed via **Integrated Gradients (Sundararajan et al.)** across the 500 input gene expressions relative to the predicted PAM50 logit $F_c(\mathbf{x})$:

$$
\text{Attr}_j(\mathbf{x}) = (x_j - x'_j) \times \int_{0}^{1} \frac{\partial F_c(\mathbf{x}' + \alpha (\mathbf{x} - \mathbf{x}'))}{\partial x_j} \, d\alpha
$$

approximated using a 20-step Riemann summation against a neutral zero-expression baseline $\mathbf{x}' = \mathbf{0}$.
- **Top Attributed Biomarkers by Subtype:**
  - **Luminal A:** Strong positive attribution on *ESR1*, *PGR*, *FOXA1*, *GATA3*; negative attribution on proliferation genes.
  - **Luminal B:** Positive attribution on *ESR1* and *MKI67* (elevated proliferation marker).
  - **HER2-enriched:** High positive attribution on *ERBB2* (HER2 amplification) and adjacent 17q12 amplicon genes (*GRB7*, *PGAP3*).
  - **Basal-like:** High positive attribution on cytokeratins (*KRT5*, *KRT14*, *KRT17*), *EGFR*, and *SOX10*; negative attribution on *ESR1* and *ERBB2*.

---

### 6. Clinical Decision Support System (CDSS) Web Architecture

The interactive user interface is implemented in Python via Streamlit (`web_app/`).

#### 6.1. System Module Overview

| Component | File Path | Functional Responsibility |
| :--- | :--- | :--- |
| **Main Orchestrator** | `web_app/app.py` | UI layout, tab navigation, state management, threshold controls |
| **Model Ingestion & Inference** | `web_app/model_utils.py` | PyTorch model loading, ResNet-50 extractor caching, RNA scaler transformation, forward pass execution |
| **WSI Processing Engine** | `web_app/wsi_utils.py` | Gigapixel SVS reading (`tifffile`), Otsu tissue segmentation, on-the-fly 256x256 patch slicing, metadata extraction |
| **Survival Prognosis Engine** | `web_app/survival_utils.py` | Multimodal CoxPH hazard score computation, 3-tier risk stratification, Kaplan-Meier curve generation |
| **Explainable AI Engine** | `web_app/xai_utils.py` | Integrated Gradients computation, Top 15 driver gene waterfall chart, 8-biomarker radar diagram |
| **Medical Design Styling** | `web_app/styles.py` | Clean CSS stylesheet, metrics cards, clinical badge palettes, responsive layout |

#### 6.2. On-the-Spot Ingestion Workflow for External Cases
The CDSS supports direct ingestion of novel external patient data:
1. **User Input:** Operator uploads one Whole Slide Image (`.svs`) and one 1-row CSV file containing raw RSEM counts for all 20,518 genes.
2. **On-the-Fly Patch Slicing (~1.0s):** `wsi_utils.extract_real_patches_from_svs` opens Series 0 of the `.svs` slide via `tifffile`, filters out background glass, ranks tissue candidates by texture variance, and crops 24 genuine 256x256 biopsy patches.
3. **On-the-Fly Feature Extraction (~2.1s):** The 24 patches are passed through the cached PyTorch ResNet-50 backbone, generating a `(1, 24, 2048)` vision tensor in memory.
4. **On-the-Fly RNA Standardization (~0.2s):** `model_utils.preprocess_raw_20k_rna` maps the 20,518 genes to the pre-fitted top 500 features and applies the cohort `StandardScaler`, yielding a `(1, 500)` tensor.
5. **Forward Inference (~0.05s):** The TransMIL late fusion model outputs PAM50 subtype probabilities, confidence score, patch attention saliency, survival hazard index, and gene attribution waterfall.
6. **Session-State Caching:** Extracted ResNet-50 features are cached in `st.session_state` so subsequent UI interactions (tab transitions, threshold adjustments) execute in <10 ms.

---

### 7. Repository Structure

```
breast-cancer-cdss/
├── data/                                 # Datasets & reference tables (gitignored)
│   ├── clinical/                         # Master matched clinical cohort (945 cases)
│   ├── genomics/                         # Top 500 gene names JSON & scaled references
│   ├── demo_external_pairs/              # 12 ready-to-test 1-row 20k RNA-Seq CSVs
│   ├── wsi_patches/                      # Cached 256x256 biopsy patch PNGs
│   └── raw_svs/                          # Uploaded and local SVS whole slide images
├── model/                                # Pretrained model weights & scalers (< 100MB)
│   ├── multimodal_genomics_best.pth      # TransMIL + Genomics Late Fusion PyTorch model
│   ├── multimodal_bilstm_best.pth        # BiLSTM + Genomics PyTorch model
│   ├── coxph_survival_model.joblib       # Lifelines Cox Proportional Hazards model
│   ├── pca_16_multimodal.joblib          # 16-component PCA on 1024D multimodal latent space
│   └── scaler_genomics_500.joblib        # Fitted StandardScaler for top 500 genes
├── notebooks/                            # 22 Jupyter research notebooks
│   ├── 00_wsi_svs_to_patches_preprocessing.ipynb
│   ├── 01_TCGA_EDA_and_Clinical_Processing.ipynb
│   ├── 02_CNN_Feature_Extraction_and_Baseline.ipynb
│   ├── 03_Classification_Traditional_ML.ipynb
│   ├── 03.1_Classification_Traditional_ML_Advanced.ipynb
│   ├── 03.2_Classification_ML_Clinical_Fusion.ipynb
│   ├── 03.3_Classification_ML_Genomics_Fusion.ipynb
│   ├── 04_Classification_Naive_DeepMIL.ipynb
│   ├── 05_Classification_TransMIL_SOTA.ipynb
│   ├── 05.1_Classification_TransMIL_Ablation.ipynb
│   ├── 06.0_Multimodal_EDA_and_Genomics_Processing.ipynb
│   ├── 06.0.1_Multimodal_Cross_Correlation_and_Clinical_Rationale.ipynb
│   ├── 06.1_Multimodal_WSI_Clinical.ipynb
│   ├── 06.2_Multimodal_WSI_Genomics.ipynb
│   ├── 06.4_Multimodal_Architecture_DeepDive_WSI_Genomics_for_explain.ipynb
│   ├── 06.6_Multimodal_WSI_BiLSTM_Genomics_Train.ipynb
│   ├── 07_Survival_Analysis_Multimodal.ipynb
│   ├── 07.1_LSTM_Survival_Prediction.ipynb
│   ├── 08_explainable_ai_multimodal.ipynb
│   ├── 09_Digital_Twin_Treatment_Simulation.ipynb
│   ├── 10_Prepare_Web_Demo_Assets_Package.ipynb
│   └── 11_Extract_WSI_Patches_and_SVS_Assets.ipynb
├── output/                               # Precomputed figures, metrics & evaluation logs
│   └── rna_preprocessing_outputs/        # RNA-Seq variance distributions, PCA & t-SNE charts
├── web_app/                              # Interactive Streamlit CDSS application
│   ├── app.py                            # Streamlit entrypoint (7 clinical modules)
│   ├── model_utils.py                    # PyTorch architectures, cached loaders & inference
│   ├── wsi_utils.py                      # SVS reader, patch slicer & attention mapper
│   ├── survival_utils.py                 # CoxPH risk model & Kaplan-Meier curve generator
│   ├── xai_utils.py                      # Integrated Gradients & Radar chart generator
│   └── styles.py                         # Medical-grade CSS stylesheet
├── CHAY_WEB_DEMO.bat                     # Windows one-click launch batch script
├── .gitignore                            # Git exclusion rules (data, svs, pt, scripts)
└── README.md                             # Comprehensive technical documentation (this file)
```

---

### 8. Installation & Execution Guide

#### 8.1. Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Environment:** Python 3.10, 3.11, or 3.13
- **Hardware:** 8 GB RAM minimum (16 GB recommended for direct SVS decoding); CUDA-compatible GPU optional (CPU inference supported).

#### 8.2. Environment Setup
```bash
# 1. Clone repository
git clone git@github.com:Cheesenoice/breast-cancer-cdss.git
cd breast-cancer-cdss

# 2. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit pandas numpy scipy scikit-learn lifelines plotly pillow tifffile imagecodecs joblib
```

#### 8.3. Launching the Web Application
```bash
# Direct Streamlit command:
streamlit run web_app/app.py --server.port 8501

# Or on Windows, double-click:
CHAY_WEB_DEMO.bat
```
Navigate to `http://localhost:8501` in your browser.

---

### 9. Technical Limitations & Discussion

1. **Retrospective Cohort Bias:** All training and validation cohorts derive from TCGA, which exhibits demographic skew toward Caucasian populations and overrepresents surgical candidates from tertiary academic centers. External generalization requires prospective multicenter validation.
2. **Histological Resolution Trade-off:** Whole-slide inference currently uses 24 tissue-dense patches sampled across the slide rather than all N > 2000 patches to enable low-latency (<3s) CPU inference during live clinical demonstrations. While this preserves dominant tumor morphology, micro-focal invasion may be under-sampled.
3. **RNA-Seq Platform Dependence:** The 500-gene scaler assumes count distributions approximately aligned with Illumina HiSeq RNA-Seq V2 RSEM outputs. Microarray data or targeted panels (e.g., Nanostring nCounter) require platform-specific calibration before ingestion.

---

### References & Foundational Literature
1. **Perou, C. M., et al. (2000).** *Molecular portraits of human breast tumours.* Nature, 406(6797), 747–752.
2. **Parker, J. S., et al. (2009).** *Supervised risk predictor of breast cancer based on intrinsic subtypes.* Journal of Clinical Oncology, 27(8), 1160–1167.
3. **Shao, Z., et al. (2021).** *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.* Advances in Neural Information Processing Systems (NeurIPS 2021).
4. **Sundararajan, M., et al. (2017).** *Axiomatic Attribution for Deep Networks.* International Conference on Machine Learning (ICML 2017).
5. **Cox, D. R. (1972).** *Regression models and life-tables.* Journal of the Royal Statistical Society: Series B, 34(2), 187–202.
