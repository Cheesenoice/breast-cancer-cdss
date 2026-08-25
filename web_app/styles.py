"""
Custom CSS Styles and Theming for Multimodal AI CDSS Web Application
Provides modern, clinical-grade UI components, badges, cards, and responsive layout.
"""

def get_custom_css():
    return """
    <style>
        /* Main background and fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
        }
        
        /* Clinical Dashboard Header */
        .cdss-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
            color: #ffffff;
            padding: 24px 30px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .cdss-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .cdss-subtitle {
            font-size: 14px;
            color: #94a3b8;
            margin-top: 6px;
            font-weight: 400;
        }
        
        .badge-sota {
            background: linear-gradient(135deg, #0284c7, #0369a1);
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
        }
        
        /* Patient Summary Card */
        .patient-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        
        .patient-meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }
        
        .meta-item {
            background: #f8fafc;
            border-radius: 10px;
            padding: 10px 12px;
            border: 1px solid #f1f5f9;
        }
        
        .meta-label {
            font-size: 11px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .meta-value {
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 2px;
        }
        
        /* PAM50 Subtype Badges */
        .badge-luma {
            background-color: #dbeafe;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
        }
        
        .badge-lumb {
            background-color: #f3e8ff;
            color: #7e22ce;
            border: 1px solid #e9d5ff;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
        }
        
        .badge-basal {
            background-color: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
        }
        
        .badge-her2 {
            background-color: #ffedd5;
            color: #c2410c;
            border: 1px solid #fed7aa;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
        }
        
        /* Metric Box Container */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }
        
        .metric-card-title {
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
        }
        
        .metric-card-value {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            margin-top: 4px;
        }
        
        /* Diagnosis Alert Box */
        .diag-box {
            padding: 16px 20px;
            border-radius: 12px;
            margin-top: 15px;
            margin-bottom: 15px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .diag-box-success {
            background-color: #f0fdf4;
            border-left: 5px solid #22c55e;
            color: #166534;
        }
        
        .diag-box-warning {
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
        }
        
        .diag-box-danger {
            background-color: #fef2f2;
            border-left: 5px solid #ef4444;
            color: #991b1b;
        }
        
        .diag-box-info {
            background-color: #f0f9ff;
            border-left: 5px solid #0284c7;
            color: #075985;
        }
        
        /* Mosaic Patch Tile */
        .patch-tile {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .patch-tile:hover {
            transform: scale(1.05);
            z-index: 10;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8fafc;
            padding: 8px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 13px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #0284c7 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
    </style>
    """
