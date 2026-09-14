# Multimodal Deep Learning for Breast Cancer Molecular Subtyping and Overall Survival Prognosis (TCGA-BRCA)

## Comprehensive Technical Report, Methodological Breakdown, and Clinical System Architecture

---

### Executive Summary

This technical report provides complete architectural documentation, mathematical formulation, empirical benchmarking, and software implementation details for an end-to-end Multimodal Artificial Intelligence Clinical Decision Support System (CDSS) developed for breast invasive carcinoma (*TCGA-BRCA*). The framework synthesizes gigapixel histopathological Whole Slide Images (WSI), high-dimensional transcriptomic profiles (20,518-gene RNA-Seq), and structured clinical records into a unified representation space to address two fundamental oncology objectives:

1. **PAM50 Molecular Subtyping (4-Class Classification):** Discrimination between the intrinsic biological subtypes of breast cancer: Luminal A, Luminal B, HER2-enriched, and Basal-like (Triple-Negative Breast Cancer).
2. **Long-Term Overall Survival Prognosis (Time-to-Event Modeling):** Estimation of patient hazard ratios and generation of 5-year survival probability curves using a Multimodal Cox Proportional Hazards framework.

The project encompasses a sequential 22-notebook experimental trajectory spanning five distinct modeling paradigms: Unimodal Traditional Machine Learning, Unimodal Multiple Instance Learning (Naive MIL, TransMIL), Cross-Modal Tabular Baselines, Deep Multimodal Late Fusion Networks (TransMIL + Genomics MLP), and Deep Sequence Fusion (BiLSTM + Genomics MLP). The validated models are integrated into a production-grade Streamlit application capable of on-the-spot ingestion and real-time inference on raw Aperio `.svs` whole slide images and full 20,518-gene raw RNA-Seq CSV files within approximately 3.5 seconds on standard CPU hardware.

---

### 1. Clinical Problem Formulation & Biomedical Background

#### 1.1. Biological Heterogeneity of Breast Cancer
Breast cancer is a heterogeneous group of malignancies displaying widely divergent histological patterns, genomic alterations, clinical trajectories, and therapeutic responses. Under the landmark Perou-Sørlie classification and the 50-gene PAM50 molecular assay (Parker et al.), breast invasive carcinomas are categorized into four primary intrinsic subtypes:

| Subtype | Receptor Status (IHC Proxy) | Proliferation Index (*MKI67*) | Clinical Behavior & Standard Care | Cohort Prevalence (TCGA-BRCA) |
| :--- | :--- | :--- | :--- | :---: |
| **Luminal A (LumA)** | ER+ and/or PR+, HER2- | Low (<14% Ki-67) | Indolent course, favorable prognosis; treated primarily with endocrine therapy (tamoxifen, aromatase inhibitors). | 52.8% (499 / 945) |
| **Luminal B (LumB)** | ER+ and/or PR+, HER2+/- | High (>=14% Ki-67) | Aggressive clinical course, higher recurrence risk; requires adjuvant cytotoxic chemotherapy alongside endocrine therapy. | 20.8% (197 / 945) |
| **HER2-enriched (Her2)** | ER-, PR-, HER2+ (amplified) | Variable to High | Rapid growth driven by 17q12 *ERBB2* amplicon; responsive to targeted anti-HER2 monoclonal antibodies (trastuzumab, pertuzumab) and tyrosine kinase inhibitors. | 8.3% (78 / 945) |
| **Basal-like (TNBC)** | ER-, PR-, HER2- ("Triple-Negative") | Markedly Elevated | Highly aggressive, early visceral metastasis; lacks targeted endocrine/HER2 options; treated with intensive systemic chemotherapy and PARP inhibitors. | 18.1% (171 / 945) |

#### 1.2. The Histopathological Diagnostic Bottleneck
In clinical practice, routine Hematoxylin and Eosin (H&E) stained histopathology slides allow pathologists to identify architectural tumor patterns (e.g., ductal vs. lobular), histological grading (Nottingham histological score), and nuclear pleomorphism. 

However, **H&E morphology alone cannot reliably separate Luminal A from Luminal B**. Both subtypes frequently present with identical moderately-differentiated glandular structures. Differentiating LumA from LumB hinges on measuring cellular proliferation rates (*MKI67*) and hormone receptor saturation, which are molecular traits inaccessible to pure visual morphology. Consequently, single-modality vision models applied to H&E slides invariably plateau at a Macro F1 score of ~0.43. Resolving this boundary demands paired transcriptomic profiling.

---

### 2. TCGA-BRCA Cohort Curation & Data Architecture

The experimental cohort was derived from The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) project, accessible via the GDC Data Portal and cBioPortal.

#### 2.1. Multi-Way Matched Cohort Filtering
Patients were subjected to strict multi-omic matching criteria:

```
[ All TCGA-BRCA Enrolled Patients: N = 1,098 ]
                        │
                        ▼ (Exclude cases lacking primary diagnostic WSI in Aperio .svs format)
[ Patients with Diagnostic WSI: N = 1,061 ]
                        │
                        ▼ (Exclude cases lacking matched Illumina HiSeq RNA-Seq V2 RSEM counts)
[ Patients with Matched WSI + RNA-Seq: N = 982 ]
                        │
                        ▼ (Exclude cases with ambiguous/missing PAM50 ground truth or survival data)
[ Final Multi-Omic Analytical Cohort: N = 945 ]
```

#### 2.2. Cohort Baseline Demographics & Clinical Characteristics

| Clinical / Genomic Variable | Cohort Value / Distribution | Notes & Methodological Handling |
| :--- | :--- | :--- |
| **Total Cohort Size** | 945 unique patients | Strict 1:1:1 multi-way matched |
| **Age at Initial Diagnosis** | Median: 58.0 years (IQR: 49.0 – 67.0, range: 26 – 90) | Standard post-/perimenopausal distribution; continuous covariate |
| **AJCC Pathologic Stage** | Stage I: 16.2% (153)<br>Stage II: 56.4% (533)<br>Stage III: 23.6% (223)<br>Stage IV: 1.8% (17)<br>Stage X / Unstaged: 2.0% (19) | One-hot encoded into discrete ordinal clinical risk factors |
| **Histological Type** | Infiltrating Ductal Carcinoma (IDC): 78.4% (741)<br>Infiltrating Lobular Carcinoma (ILC): 18.2% (172)<br>Mixed / Other: 3.4% (32) | Pathological morphology stratification |
| **Overall Survival (OS)** | Median follow-up: 28.5 months (range: 0.1 – 282.7 months) | Right-censored time-to-event outcome |
| **Vital Status / Censoring** | Censored (Alive): 85.2% (805)<br>Events (Deceased): 14.8% (140) | Standard TCGA long-term survival censoring profile |
| **PAM50 Ground Truth** | LumA: 499 (52.8%) • LumB: 197 (20.8%) • Basal: 171 (18.1%) • Her2: 78 (8.3%) | Significant 4-class imbalance reflecting natural biology |

---

### 3. Detailed Experimental Trajectory (The 22 Research Notebooks)

The research progression spans 22 standalone notebooks systematically numbered across analytical phases:

```
[ Phase 1: WSI & Clinical Foundation ]
  ├── 00_wsi_svs_to_patches_preprocessing.ipynb
  ├── 01_TCGA_EDA_and_Clinical_Processing.ipynb
  └── 02_CNN_Feature_Extraction_and_Baseline.ipynb

[ Phase 2: Unimodal Machine Learning & Multiple Instance Learning ]
  ├── 03_Classification_Traditional_ML.ipynb
  ├── 03.1_Classification_Traditional_ML_Advanced.ipynb
  ├── 03.2_Classification_ML_Clinical_Fusion.ipynb
  ├── 03.3_Classification_ML_Genomics_Fusion.ipynb
  ├── 04_Classification_Naive_DeepMIL.ipynb
  ├── 05_Classification_TransMIL_SOTA.ipynb
  └── 05.1_Classification_TransMIL_Ablation.ipynb

[ Phase 3: Transcriptomic Feature Engineering & Multimodal Deep Learning ]
  ├── 06.0_Multimodal_EDA_and_Genomics_Processing.ipynb
  ├── 06.0.1_Multimodal_Cross_Correlation_and_Clinical_Rationale.ipynb
  ├── 06.1_Multimodal_WSI_Clinical.ipynb
  ├── 06.2_Multimodal_WSI_Genomics.ipynb
  ├── 06.4_Multimodal_Architecture_DeepDive_WSI_Genomics_for_explain.ipynb
  └── 06.6_Multimodal_WSI_BiLSTM_Genomics_Train.ipynb

[ Phase 4: Long-Term Survival Prognosis & Deep Time-to-Event Modeling ]
  ├── 07_Survival_Analysis_Multimodal.ipynb
  └── 07.1_LSTM_Survival_Prediction.ipynb

[ Phase 5: Interpretability, Digital Twin Simulation & Production Assets ]
  ├── 08_explainable_ai_multimodal.ipynb
  ├── 09_Digital_Twin_Treatment_Simulation.ipynb
  ├── 10_Prepare_Web_Demo_Assets_Package.ipynb
  └── 11_Extract_WSI_Patches_and_SVS_Assets.ipynb
```

---

### 4. Mathematical Formulations & Architectural Blueprints

#### 4.1. Stage 1: Histopathological Tiling & Feature Encoding (NB 00, 02)
Diagnostic Whole Slide Images in Aperio `.svs` format represent gigapixel tissue matrices (typically 80,000 x 60,000 pixels at 40x optical magnification). Direct end-to-end convolutional training is computationally intractable on modern GPUs.

##### Step 1: Otsu Tissue-Background Segmentation
The thumbnail image is transformed from RGB to the HSV color space. Tissue regions exhibit higher Saturation ($S$) than transparent glass slides. The optimal threshold $\tau$ maximizes inter-class variance:

$$
\sigma_B^2(\tau) = \omega_0(\tau)\omega_1(\tau)\left[\mu_0(\tau) - \mu_1(\tau)\right]^2
$$

Generating a binary foreground mask $M(x, y) \in \{0, 1\}$.

##### Step 2: Patch Extraction & Artifact Filtering
Non-overlapping tiles of dimension 256 x 256 pixels are extracted across the foreground mask at 20x optical magnification. A candidate tile $P_k$ is retained if and only if it satisfies both cellularity and texture variance criteria:

$$
\frac{1}{256^2} \sum_{(x,y) \in P_k} M(x, y) \ge 0.40 \quad \text{and} \quad \text{std}(P_k) \ge 10.0
$$

This dual criterion filters out empty glass, mounting resin, folded edges, and acellular adipose bubbles.

##### Step 3: Feature Encoding via ResNet-50 Backbone
Retained patches are normalized using ImageNet channel parameters (mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]`) and forwarded through a pretrained **ResNet-50** backbone truncated after the global average pooling layer (`AdaptiveAvgPool2d`). The entire patient biopsy is represented as a permutation-invariant bag of $N$ embedding vectors:

$$
\mathbf{X} = \{ \mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N \}, \quad \mathbf{h}_i \in \mathbb{R}^{2048}
$$

---

#### 4.2. Stage 2: Transcriptomic Feature Selection & Normalization (NB 06.0)
The raw transcriptomic data (`data_mrna_seq_v2_rsem.txt`) contains RSEM normalized counts for M = 20,518 genes. Directly feeding 20,518 features into a multimodal network causes severe overfitting and curse-of-dimensionality degradation.

##### Step 1: Population Variance Gene Ranking
For each gene $j$ in the transcriptomic panel, unbiased population sample variance is computed across the patient cohort ($N = 945$):

$$
s_j^2 = \frac{1}{N - 1} \sum_{i=1}^{N} (x_{ij} - \bar{x}_j)^2
$$

Genes are ranked in descending order:

$$
s_{(1)}^2 \ge s_{(2)}^2 \ge \dots \ge s_{(M)}^2
$$

##### Step 2: Top 500 Informative Biomarker Cutoff
The top 500 genes ($K = 500$) capture the vast majority of biological variance in breast cancer oncogenesis. This subset naturally isolates intrinsic PAM50 drivers (*ESR1*, *PGR*, *ERBB2*, *MKI67*, *FOXA1*, *GATA3*, *KRT5*, *KRT14*, *EGFR*, *SOX10*) while excluding non-informative housekeeping genes (*ACTB*, *GAPDH*, *B2M*).

##### Step 3: StandardScaler Z-Score Transformation
Extensive testing revealed that direct Z-score standardization on raw counts preserves relative linear expression amplitude better than log2 transforms:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

where cohort mean and standard deviation per gene:

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}, \quad \sigma_j = \sqrt{s_j^2}
$$

are fitted across the training cohort and saved in `scaler_genomics_500.joblib`. This produces the dense standardized genomic vector:

$$
\mathbf{x}_{\text{gen}} \in \mathbb{R}^{500}
$$

---

#### 4.3. Stage 3: Multiple Instance Learning Evolution (NB 04, 05, 05.1)

```
[ ResNet-50 Bag: N x 2048D ]
           │
           ▼
[ Linear Projection: N x 512D ]
           │
           ▼
[ Correlated Nyström Multi-Head Self-Attention: N x 512D ] (O(N) Complexity)
           │
           ▼
[ Class Token [CLS] Pooling: 1 x 512D ]
           │
           ▼
[ Morphological Latent Embedding: v_img in R^512 ]
```

##### Paradigm 1: Naive Deep MIL (NB 04)
Aggregates patch embeddings via static symmetric pooling operators:

$$
\mathbf{v}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i, \quad \mathbf{v}_{\text{max}} = \max_{i=1}^N (\mathbf{h}_i)
$$

While permutation-invariant, mean-pooling dilutes focal malignant signals across non-neoplastic tissue, while max-pooling discards tumor microenvironment context.

##### Paradigm 2: TransMIL - Transformer-based Correlated MIL (NB 05)
Standard Softmax self-attention has quadratic complexity $\mathcal{O}(N^2)$, which is prohibitive when bags contain up to $N = 3,000$ patches. TransMIL utilizes the **Nyström approximation** of self-attention to reduce complexity to linear $\mathcal{O}(N)$:

$$
\hat{\mathbf{A}} = \text{Softmax}\left(\frac{\mathbf{Q} \tilde{\mathbf{K}}^\top}{\sqrt{d}}\right) \left[\text{Softmax}\left(\frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d}}\right)\right]^+ \text{Softmax}\left(\frac{\tilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d}}\right)
$$

where $\tilde{\mathbf{Q}}$ and $\tilde{\mathbf{K}}$ represent selected landmark approximations ($m = 64$ landmarks).

A learnable classification token [CLS] is prepended to the patch sequence. Through multi-head Nyström attention layers, morphological correlations between distant tissue regions are learned, outputting a slide representation:

$$
\mathbf{v}_{\text{img}} \in \mathbb{R}^{512}
$$

---

#### 4.4. Stage 4: SOTA Multimodal Late Fusion Architecture (NB 06.2)

```
[ WSI Patches: N x 2048D ] ────► TransMIL Backbone ────► v_img in R^512 ──┐
                                                                          ├──► [ Concatenation ] ──► v_fusion in R^1024 ──► Classifier ──► Softmax
[ RNA-Seq: 500D Vector ]   ────► Genomics MLP      ────► v_gen in R^512 ──┘
```

The primary production model integrates both modalities via late feature fusion:

##### Stream 1: Vision Stream (Histopathology)
The WSI patch bag is processed by the pretrained TransMIL encoder:

$$
\mathbf{v}_{\text{img}} = \text{TransMIL}(\mathbf{X}) \in \mathbb{R}^{512}
$$

##### Stream 2: Genomics Stream (Transcriptomics)
A specialized deep Multi-Layer Perceptron projects the 500-dimensional continuous expression vector into the same latent dimensionality:

$$
\mathbf{h}_{\text{gen}}^{(1)} = \text{ReLU}\left(\text{LayerNorm}\left(\mathbf{W}_1 \mathbf{x}_{\text{gen}} + \mathbf{b}_1\right)\right), \quad \mathbf{W}_1 \in \mathbb{R}^{256 \times 500}
$$

$$
\mathbf{v}_{\text{gen}} = \text{ReLU}\left(\mathbf{W}_2 \cdot \text{Dropout}_{0.3}\left(\mathbf{h}_{\text{gen}}^{(1)}\right) + \mathbf{b}_2\right), \quad \mathbf{W}_2 \in \mathbb{R}^{512 \times 256}
$$

##### Stream 3: Multimodal Late Feature Fusion
The morphological latent vector and transcriptomic latent vector are concatenated into a 1024-dimensional joint representation:

$$
\mathbf{v}_{\text{fusion}} = \left[ \mathbf{v}_{\text{img}} \,,\, \mathbf{v}_{\text{gen}} \right] \in \mathbb{R}^{1024}
$$

##### Stream 4: Subtype Classification Head
A 2-layer MLP head projects the fused embedding to the 4 PAM50 class logits:

$$
\hat{\mathbf{y}} = \text{Softmax}\left(\mathbf{W}_4 \cdot \text{Dropout}_{0.3}\left(\text{ReLU}\left(\mathbf{W}_3 \mathbf{v}_{\text{fusion}} + \mathbf{b}_3\right)\right) + \mathbf{b}_4\right)
$$

$$
\mathbf{W}_3 \in \mathbb{R}^{256 \times 1024}, \quad \mathbf{W}_4 \in \mathbb{R}^{4 \times 256}
$$

##### Optimization: Label-Smoothed Class-Weighted Cross-Entropy Loss
To mitigate the 6.4:1 class imbalance between Luminal A and HER2-enriched subtypes, training uses weighted cross-entropy with label smoothing ($\epsilon = 0.05$):

$$
\mathcal{L}_{\text{CE}} = -\sum_{c=1}^4 w_c \left[ (1 - \epsilon) y_c + \frac{\epsilon}{4} \right] \log(\hat{y}_c)
$$

where class weights are calibrated inversely to training frequency:

$$
w_{\text{Her2}} = 3.03, \quad w_{\text{Basal}} = 1.38, \quad w_{\text{LumB}} = 1.20, \quad w_{\text{LumA}} = 0.47
$$

---

#### 4.5. Stage 5: Multimodal Survival Analysis & Risk Stratification (NB 07, 07.1)

```
[ Multimodal Representation: v_fusion in R^1024 ]
                      │
                      ▼
[ Orthogonal PCA: 16 Components (>88% Variance) ]
                      │
                      ▼
[ Regularized Cox Proportional Hazards Model ]
                      │
                      ├──► Hazard Score (eta = beta^T * z) ──► 3-Tier Clinical Risk Group
                      └──► Breslow Cumulative Hazard ───────► 5-Year Survival Probability Curve
```

##### Step 1: Latent Space Orthogonal Dimensionality Reduction (PCA-16)
Fitting a survival model directly on 1024 features across 945 samples induces severe collinearity. Principal Component Analysis (PCA) reduces the joint representation to $d = 16$ orthogonal components, retaining over 88.2% of cumulative variance:

$$
\mathbf{z} = \mathbf{U}_{16}^\top (\mathbf{v}_{\text{fusion}} - \boldsymbol{\mu}_{\text{fusion}}) \in \mathbb{R}^{16}
$$

##### Step 2: Regularized Cox Proportional Hazards Formulation
The hazard rate of death at time $t$ given covariates $\mathbf{z}$ is parameterized as:

$$
h(t \mid \mathbf{z}) = h_0(t) \exp\left(\boldsymbol{\beta}^\top \mathbf{z}\right), \quad \boldsymbol{\beta} \in \mathbb{R}^{16}
$$

where baseline hazard function $h_0(t)$ and coefficient vector $\boldsymbol{\beta}$ are estimated by maximizing Cox's partial log-likelihood with L2 penalty:

$$
\ell(\boldsymbol{\beta}) = \sum_{i: E_i = 1} \left[ \boldsymbol{\beta}^\top \mathbf{z}_i - \log\left(\sum_{j \in R(T_i)} \exp\left(\boldsymbol{\beta}^\top \mathbf{z}_j\right)\right) \right] - \lambda \|\boldsymbol{\beta}\|_2^2
$$

where the risk set $R(t)$ denotes patients surviving immediately prior to failure time $t$.

##### Step 3: Prognostic Discrimination (Harrell's C-Index)
Model discriminative capability is quantified by Harrell's Concordance Index, evaluating all evaluable patient pairs $(i, j)$:

$$
C = \frac{\sum_{i \ne j} \mathbb{I}(T_i < T_j) \cdot \mathbb{I}(\hat{\eta}_i > \hat{\eta}_j) \cdot E_i}{\sum_{i \ne j} \mathbb{I}(T_i < T_j) \cdot E_i}
$$

##### Step 4: Clinical 3-Tier Risk Stratification
Patient prognostic hazard scores are computed from the linear predictor:

$$
\eta_i = \boldsymbol{\beta}^\top \mathbf{z}_i
$$

Calibrated against clinical overall survival outcomes, the continuous hazard score is stratified into three actionable clinical risk tiers:
- **Low Risk ($\eta \lt 0.90$):** Indolent prognosis, 5-year survival probability > 88% (Green badge).
- **Borderline / Moderate Risk ($0.90 \le \eta \le 1.15$):** Intermediate prognosis, 5-year survival probability 70% - 85% (Amber badge).
- **High Risk ($\eta \gt 1.15$):** Aggressive prognosis, 5-year survival probability < 65% (Red badge).

##### Step 5: 5-Year Survival Curve Projection (Breslow Estimator)
The cumulative baseline hazard function:

$$
H_0(t) = \int_0^t h_0(u) \, du
$$

is estimated non-parametrically using Breslow's method:

$$
\hat{H}_0(t) = \sum_{t_i \le t} \frac{d_i}{\sum_{j \in R(t_i)} \exp(\hat{\eta}_j)}
$$

The time-dependent survival function for any new patient with risk score $\eta$ over a 60-month timeline is computed as:

$$
S(t \mid \mathbf{z}) = \exp\left(-\hat{H}_0(t) \exp(\eta)\right)
$$

---

#### 4.6. Stage 6: Dual-Domain Explainable AI (XAI) (NB 08)

```
                                  [ DUAL EXPLAINABILITY ]
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
    [ Spatial Attention (WSI) ]                               [ Integrated Gradients (Genomics) ]
               │                                                           │
    • Cosine Saliency: CLS vs Patch                           • 20-Step Riemann Path Integral
    • Invasive Core vs Stroma Separation                      • Top 15 Driver Gene Waterfall
    • Color-Coded Heatmap Borders                             • 8-Biomarker Expression Radar
```

##### Domain 1: Histopathological Saliency Mapping (CLS Attention Similarity)
To localize the histological regions driving the TransMIL decision, the cosine similarity between the slide-level [CLS] token representation and each individual patch representation is computed:

$$
s_i = \frac{\mathbf{v}_{\text{img}}^\top \mathbf{h}_i}{\|\mathbf{v}_{\text{img}}\| \|\mathbf{h}_i\|}, \quad \alpha_i = \frac{s_i - \min(\mathbf{s})}{\max(\mathbf{s}) - \min(\mathbf{s}) + \epsilon} \in [0, 1]
$$

- **Attention >= 0.75 (Red Border):** Core invasive neoplastic epithelial nests, high nuclear pleomorphism, atypical mitotic figures.
- **Attention 0.50 – 0.75 (Orange Border):** Infiltrating tumor margins and ductal carcinoma in situ (DCIS) components.
- **Attention 0.25 – 0.50 (Yellow Border):** Tumor-infiltrating lymphocytes and desmoplastic reactive stroma.
- **Attention < 0.25 (Green Border):** Acellular dense collagenous stroma, benign adipose tissue, and normal lobules.

##### Domain 2: Genomic Attribution via Integrated Gradients
To identify which of the 500 genes contributed most significantly to the predicted PAM50 logit, path integrals are computed along the straight line from a neutral baseline $\mathbf{x}' = \mathbf{0}$ (average expression across normalized cohort) to the patient's actual expression vector $\mathbf{x}$:

$$
\text{Attr}_j(\mathbf{x}) = (x_j - x'_j) \times \int_{0}^{1} \frac{\partial F_c\left(\mathbf{x}' + \alpha (\mathbf{x} - \mathbf{x}')\right)}{\partial x_j} \, d\alpha
$$

Approximated via a 20-step Riemann summation:

$$
\text{Attr}_j(\mathbf{x}) \approx (x_j - x'_j) \times \frac{1}{20} \sum_{k=1}^{20} \frac{\partial F_c\left(\mathbf{x}' + \frac{k}{20}(\mathbf{x} - \mathbf{x}')\right)}{\partial x_j}
$$

---

### 5. Benchmark Results & Quantitative Evaluation

All models were evaluated using identical 5-fold cross-validation splits stratified by PAM50 subtype.

#### 5.1. Comprehensive Model Comparison Matrix

| Model Architecture | Input Modality | Accuracy | Macro Precision | Macro Recall | Macro F1 | Survival C-Index | Computational Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean-Pool Logistic Regression** | WSI Alone | 51.4% | 0.442 | 0.408 | 0.412 | 0.582 | < 1 ms |
| **Mean-Pool Random Forest** | WSI Alone | 52.1% | 0.458 | 0.415 | 0.420 | 0.591 | ~ 5 ms |
| **Mean-Pool SVM (RBF Kernel)** | WSI Alone | 52.8% | 0.461 | 0.421 | 0.431 | 0.604 | ~ 3 ms |
| **Mean-Pool XGBoost** | WSI Alone | 53.6% | 0.473 | 0.428 | 0.438 | 0.612 | ~ 8 ms |
| **PCA-128 Engineered ML (NB 03.1)** | WSI Alone | 53.1% | 0.465 | 0.430 | 0.446 | 0.618 | ~ 12 ms |
| **Naive Deep MIL (Mean-Pool)** | WSI Alone | 53.7% | 0.459 | 0.419 | 0.421 | 0.620 | ~ 15 ms |
| **Naive Attention-MIL (Ilse et al.)** | WSI Alone | 54.2% | 0.468 | 0.432 | 0.440 | 0.635 | ~ 22 ms |
| **TransMIL (Nyström Attention)** | WSI Alone | 54.2% | 0.472 | 0.429 | 0.425 | 0.654 | ~ 45 ms |
| **Clinical Tabular Logistic Regression**| Clinical Alone | 61.2% | 0.540 | 0.505 | 0.518 | 0.612 | < 1 ms |
| **WSI + Clinical Fusion (SVM)** | WSI + Clinical | 69.1% | 0.615 | 0.572 | 0.586 | 0.662 | ~ 10 ms |
| **WSI + Clinical Fusion (LogReg)** | WSI + Clinical | 69.0% | 0.621 | 0.580 | 0.594 | 0.658 | ~ 5 ms |
| **Genomics MLP (500 Genes)** | RNA-Seq Alone | 85.3% | 0.841 | 0.828 | 0.832 | 0.718 | ~ 2 ms |
| **WSI + RNA-Seq Late Fusion (XGBoost)**| WSI + RNA-Seq | 87.5% | 0.865 | 0.854 | 0.859 | 0.732 | ~ 15 ms |
| **BiLSTM + Genomics MLP (NB 06.6)** | WSI + RNA-Seq | 75.9% | 0.685 | 0.643 | 0.655 | 0.684 | ~ 85 ms |
| **TransMIL + Genomics MLP (SOTA)** | **WSI + RNA-Seq** | **88.4%** | **0.881** | **0.870** | **0.875** | **0.767** | **~ 48 ms** |

#### 5.2. Confusion Matrix Analysis of the SOTA Model (TransMIL + Genomics)
Evaluation across the test fold reveals strong per-class performance:
- **Basal-like:** Precision 94.2%, Recall 96.5%, F1 95.3% (Distinct genomic profile dominated by cytokeratin expression and *TP53* loss).
- **HER2-enriched:** Precision 88.5%, Recall 82.1%, F1 85.2% (Clear *ERBB2* amplicon signal).
- **Luminal A:** Precision 89.2%, Recall 91.4%, F1 90.3% (Strong estrogen/progesterone receptor network).
- **Luminal B:** Precision 80.4%, Recall 78.2%, F1 79.3% (Slight residual misclassification with Luminal A at the proliferation threshold boundary).

---

### 6. Production Clinical Decision Support System (CDSS) Architecture

The validated models are bundled into an interactive software platform built with Streamlit (`web_app/`).

```
[ Clinical Decision Support System (Streamlit CDSS) ]
  ├── Tab 1: Multimodal Diagnosis & Risk Stratification (Confidence Gauge, Subtype Distribution)
  ├── Tab 2: Histopathology & Whole Slide Interactive Studio (SVS Decoding, Multi-Zoom 4x-40x)
  ├── Tab 3: Explainable AI & Biomarker Radar (Top 15 Drivers, 8-Biomarker Profile)
  ├── Tab 4: 5-Year Survival Prognosis & Kaplan-Meier Curve (CoxPH Hazard Index)
  ├── Tab 5: 2D Cohort Landscape & Patient Locator (t-SNE Embedding Map)
  ├── Tab 6: Digital Twin In Silico Treatment Simulator (Counterfactual Response Modeling)
  └── Tab 7: Comprehensive Clinical Audit & Export (JSON / CSV Diagnostic Reports)
```

#### 6.1. On-the-Spot Ingestion of Novel External Patients
The application supports direct evaluation of unseen external patients without prior database ingestion:

1. **Dual Ingestion Interface:** The operator uploads:
   - One Whole Slide Image in Aperio format (`.svs`) or histological image.
   - One 1-row CSV containing raw RSEM expression counts across all 20,518 genes.
2. **On-the-Fly Patch Slicing (~1.0s):**
   `wsi_utils.extract_real_patches_from_svs` accesses the highest-resolution series (`Series 0`) of the `.svs` container via `tifffile`, executes tissue-background segmentation, ranks tissue candidates by texture variance, and extracts 24 genuine 256 x 256 biopsy patches.
3. **On-the-Fly Feature Extraction (~2.1s):**
   The 24 patches are converted into normalized tensors and passed through the in-memory cached ResNet-50 network, producing a (1, 24, 2048) vision tensor in RAM.
4. **On-the-Fly RNA Standardization (~0.2s):**
   `model_utils.preprocess_raw_20k_rna` parses the 20,518 columns, filters to the top 500 features, and standardizes them using the cohort `StandardScaler`, yielding a (1, 500) tensor.
5. **Real-Time Multimodal Inference (~0.05s):**
   The TransMIL late-fusion model executes a single forward pass, generating the predicted subtype, confidence gauge, 4-class probability distribution, spatial attention weights, and 1024D joint representation.
6. **Session-State Caching:**
   Extracted ResNet-50 features are cached in `st.session_state[f"{selected_pid}_resnet_features"]`. Switching between the 7 tabs or altering clinical thresholds executes instantly in < 10 ms.

---

### 7. Repository Organization

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

### 8. Verification & Quick Start Guide

#### 8.1. Environment Setup
```bash
# Clone the repository
git clone git@github.com:Cheesenoice/breast-cancer-cdss.git
cd breast-cancer-cdss

# Initialize Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install PyTorch and application dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit pandas numpy scipy scikit-learn lifelines plotly pillow tifffile imagecodecs joblib
```

#### 8.2. Launching the Web CDSS
```bash
# Run via Streamlit:
streamlit run web_app/app.py --server.port 8501

# Or on Windows, double-click:
CHAY_WEB_DEMO.bat
```
Open `http://localhost:8501` in your browser.

---

### 9. Methodological Limitations & Future Scope

1. **Retrospective Cohort Bias:** The training data derives entirely from TCGA, which features a demographic skew toward Caucasian patients and overrepresents surgical resections from academic medical centers. Prospective validation across multi-ethnic community cohorts is necessary.
2. **Computational Sampling vs. Micro-Invasion:** For real-time CPU demonstration (<3.5s latency), the slide viewer samples 24 high-cellularity patches. While this reliably captures predominant tumor grade, micro-focal vascular invasion or sparse tertiary lymphoid structures may be missed unless the entire bag (N > 2,000) is processed via batch inference.
3. **Assay Calibration:** The 500-gene standardization assumes RSEM count normalization. Ingesting raw unnormalized counts or alternate profiling technologies (e.g., Nanostring nCounter, Affymetrix arrays) requires prior assay-specific calibration.

---

### 10. References & Foundational Literature

1. **Perou, C. M., et al. (2000).** *Molecular portraits of human breast tumours.* Nature, 406(6797), 747–752.
2. **Parker, J. S., et al. (2009).** *Supervised risk predictor of breast cancer based on intrinsic subtypes.* Journal of Clinical Oncology, 27(8), 1160–1167.
3. **Shao, Z., et al. (2021).** *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.* Advances in Neural Information Processing Systems (NeurIPS 2021).
4. **Sundararajan, M., et al. (2017).** *Axiomatic Attribution for Deep Networks.* International Conference on Machine Learning (ICML 2017).
5. **Ilse, M., et al. (2018).** *Attention-based Deep Multiple Instance Learning.* International Conference on Machine Learning (ICML 2018).
6. **Cox, D. R. (1972).** *Regression models and life-tables.* Journal of the Royal Statistical Society: Series B, 34(2), 187–202.
7. **Harrell, F. E., et al. (1982).** *Evaluating the yield of medical tests.* Journal of the American Medical Association, 247(18), 2543–2546.
8. **Goldhirsch, A., et al. (2013).** *Personalizing the treatment of women with early breast cancer: highlights of the St Gallen International Expert Consensus on the Primary Therapy of Early Breast Cancer 2013.* Annals of Oncology, 24(9), 2206–2223.
