"""
Explainable AI (XAI) and Genomics Biomarker Visualization Utilities for CDSS Web App
Computes Integrated Gradients gene attributions and generates Biomarker Radar Charts.
"""

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_gene_integrated_gradients(model, img_source, gen_vector, target_class, top_500_genes, n_steps=20):
    """
    Computes Integrated Gradients attribution scores for the 500 genes with respect to target PAM50 class.
    Returns:
        top_genes (list), top_scores (list), and interactive Plotly horizontal bar figure.
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
        if gen_tensor.ndim == 1:
            gen_tensor = gen_tensor.unsqueeze(0)
    else:
        gen_tensor = torch.tensor(gen_vector, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Baseline is zero vector (average expression after standard scaling)
    baseline_gen = torch.zeros_like(gen_tensor)
    
    # Linear interpolation steps
    alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
    accum_grads = torch.zeros_like(gen_tensor)
    
    model.eval()
    for alpha in alphas:
        interpolated_gen = baseline_gen + alpha * (gen_tensor - baseline_gen)
        interpolated_gen.requires_grad_(True)
        
        logits, _, _, _, _ = model(img_tensor, interpolated_gen)
        target_logit = logits[0, target_class]
        
        model.zero_grad()
        target_logit.backward(retain_graph=True)
        
        accum_grads += interpolated_gen.grad
        
    # Riemann approximation of the integral
    avg_grads = accum_grads / n_steps
    attributions = ((gen_tensor - baseline_gen) * avg_grads).squeeze(0).cpu().detach().numpy()
    
    # Find Top 15 most influential genes (by absolute magnitude)
    top_indices = np.argsort(np.abs(attributions))[-15:]
    selected_genes = [top_500_genes[i] for i in top_indices]
    selected_scores = [float(attributions[i]) for i in top_indices]
    
    # Sort for clean waterfall visualization
    sorted_pairs = sorted(zip(selected_genes, selected_scores), key=lambda x: x[1])
    plot_genes = [p[0] for p in sorted_pairs]
    plot_scores = [p[1] for p in sorted_pairs]
    colors = ['#ef4444' if s > 0 else '#3b82f6' for s in plot_scores]
    
    # Generate Interactive Plotly Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_genes,
        x=plot_scores,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#0f172a', width=0.5)),
        hovertemplate='<b>Gen:</b> %{y}<br><b>Attribution Score:</b> %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Top 15 Gen Quyết Định (Integrated Gradients Attribution)</b><br><span style='font-size:12px; color:#64748b;'>Đỏ: Gen thúc đẩy khẳng định nhãn | Xanh: Gen ức chế/phủ định</span>",
            x=0.02, y=0.95
        ),
        xaxis=dict(title="<b>Điểm Cống Hiến Quyết Định (Attribution Score)</b>", gridcolor="#f1f5f9", zerolinecolor="#64748b", zerolinewidth=1.5),
        yaxis=dict(title="", tickfont=dict(size=12, family="monospace", color="#0f172a")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=80, r=30, t=60, b=40)
    )
    
    return {
        'top_genes': plot_genes,
        'top_scores': plot_scores,
        'fig': fig
    }


def create_biomarker_radar_chart(patient_scaled_series, gene_stats):
    """
    Creates an 8-axis Biomarker Radar Chart comparing patient expression to population baseline.
    Key Biomarkers: ERBB2, ESR1, PGR, MKI67, TP53, BRCA1, BRCA2, EGFR.
    """
    biomarkers = ['ERBB2', 'ESR1', 'PGR', 'MKI67', 'TP53', 'BRCA1', 'BRCA2', 'EGFR']
    labels = ['ERBB2 (HER2)', 'ESR1 (ER)', 'PGR (PR)', 'MKI67 (Ki-67)', 'TP53', 'BRCA1', 'BRCA2', 'EGFR']
    
    patient_vals = []
    cohort_mean_vals = [0.0] * len(biomarkers) # Z-score mean is 0.0
    
    for g in biomarkers:
        if g in patient_scaled_series:
            val = float(patient_scaled_series[g])
            # Bound within [-3, 3] for visual aesthetics
            patient_vals.append(np.clip(val, -3.0, 3.5))
        else:
            patient_vals.append(0.0)
            
    # Close the radar polygon
    radar_labels = labels + [labels[0]]
    radar_patient = patient_vals + [patient_vals[0]]
    radar_mean = cohort_mean_vals + [cohort_mean_vals[0]]
    
    fig = go.Figure()
    
    # Population Mean reference
    fig.add_trace(go.Scatterpolar(
        r=radar_mean,
        theta=radar_labels,
        name='Trung Bình Quần Thể TCGA (Mean = 0)',
        line=dict(color='#94a3b8', dash='dash', width=2),
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.1)',
        hoverinfo='text',
        hovertext=['Quần thể TCGA: Z-Score = 0.0'] * len(radar_labels)
    ))
    
    # Patient trace
    fig.add_trace(go.Scatterpolar(
        r=radar_patient,
        theta=radar_labels,
        name='Bệnh Nhân Hiện Tại',
        line=dict(color='#0284c7', width=3),
        fill='toself',
        fillcolor='rgba(2, 132, 199, 0.25)',
        hovertemplate='<b>%{theta}:</b> %{r:.2f} Z-Score<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Mạng Nhện Gen Chỉ Dấu Lâm Sàng (Key Biomarkers Profile)</b>",
            x=0.02, y=0.95
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[-3, 3.5], tickfont=dict(size=9), gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=11, weight="bold", color="#0f172a"), gridcolor="#e2e8f0")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor="white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=60)
    )
    
    return fig
