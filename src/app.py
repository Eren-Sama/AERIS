import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import os
from scipy.ndimage import gaussian_filter, sobel

import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

st.set_page_config(
    page_title="AERIS - Satellite Cloud Detection",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #0a0e27 0%, #16213e 50%, #0a0e27 100%); }
    h1 { color: #e8eaf6 !important; font-weight: 200 !important; letter-spacing: 4px !important; 
         font-size: 2.5em !important; text-shadow: 0 0 20px rgba(121, 134, 203, 0.3); text-align: center !important; }
    .subtitle { color: #9fa8da; font-size: 1.05em; text-align: center; letter-spacing: 1px; 
                margin-bottom: 2em; line-height: 1.6; }
    h2 { color: #c5cae9 !important; font-weight: 200 !important; font-size: 1.5em !important; 
         margin-top: 2.5em !important; border-bottom: 1px solid rgba(121, 134, 203, 0.2); 
         padding-bottom: 0.8em; letter-spacing: 2px; }
    h3 { color: #9fa8da !important; font-weight: 300 !important; letter-spacing: 1px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1f3a 0%, #0f1419 100%);
        border-right: 1px solid rgba(121, 134, 203, 0.15); box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5); }
    [data-testid="stSidebar"] h4 { color: #7986cb !important; font-weight: 400 !important; 
        letter-spacing: 2px; font-size: 0.9em !important; text-transform: uppercase; 
        border-bottom: 1px solid rgba(121, 134, 203, 0.2); padding-bottom: 0.5em; margin-bottom: 1em; }
    .stButton>button { background: linear-gradient(135deg, #3949ab 0%, #5e35b1 100%); color: #fff;
        border: none; border-radius: 8px; padding: 0.9rem 2.5rem; font-weight: 400; letter-spacing: 2px;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(57, 73, 171, 0.4); 
        text-transform: uppercase; font-size: 0.9em; }
    .stButton>button:hover { background: linear-gradient(135deg, #5e35b1 0%, #3949ab 100%);
        box-shadow: 0 6px 25px rgba(94, 53, 177, 0.6); transform: translateY(-2px); }
    .metric-card { background: linear-gradient(135deg, rgba(26, 31, 58, 0.9) 0%, rgba(15, 20, 25, 0.9) 100%);
        border: 1px solid rgba(121, 134, 203, 0.2); border-radius: 12px; padding: 1.8rem; text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px); transition: all 0.3s ease; }
    .metric-card:hover { border-color: rgba(121, 134, 203, 0.4); 
        box-shadow: 0 12px 40px rgba(57, 73, 171, 0.3); transform: translateY(-4px); }
    .metric-card h3 { color: #5c6bc0; font-size: 0.7em; font-weight: 400; text-transform: uppercase;
        letter-spacing: 2px; margin: 0 0 0.8rem 0; }
    .metric-card h1 { color: #e8eaf6; font-size: 2.2em; font-weight: 200; margin: 0; letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(232, 234, 246, 0.3); }
    .metric-card p { color: #7986cb; font-size: 0.85em; margin: 0.5rem 0 0 0; letter-spacing: 1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background: rgba(26, 31, 58, 0.6); 
        border: 1px solid rgba(121, 134, 203, 0.2); color: #7986cb; padding: 0.7rem 1.8rem;
        font-size: 0.85em; letter-spacing: 1px; border-radius: 8px 8px 0 0; transition: all 0.2s ease; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(57, 73, 171, 0.3) 0%, 
        rgba(94, 53, 177, 0.3) 100%); color: #e8eaf6; border-color: rgba(121, 134, 203, 0.4); }
    .stAlert { background: rgba(26, 31, 58, 0.8); border-left: 3px solid #5c6bc0; 
        color: #c5cae9; border-radius: 8px; }
    .stSuccess { background: rgba(46, 125, 50, 0.15); border-left: 3px solid #66bb6a; color: #a5d6a7; }
    .stInfo { background: rgba(25, 118, 210, 0.15); border-left: 3px solid #42a5f5; color: #90caf9; }
    .stProgress > div > div { background: linear-gradient(90deg, #5e35b1 0%, #3949ab 100%); }
    .streamlit-expanderHeader { background: rgba(26, 31, 58, 0.6); 
        border: 1px solid rgba(121, 134, 203, 0.2); border-radius: 8px; color: #9fa8da; 
        transition: all 0.2s ease; }
    .streamlit-expanderHeader:hover { background: rgba(57, 73, 171, 0.2); 
        border-color: rgba(121, 134, 203, 0.3); }
    [data-testid="stMetricValue"] { font-size: 2em !important; color: #e8eaf6 !important; 
        font-weight: 200 !important; }
    [data-testid="stMetricLabel"] { color: #7986cb !important; font-size: 0.75em !important; 
        text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stFileUploader"] { background: rgba(26, 31, 58, 0.5); 
        border: 1px dashed rgba(121, 134, 203, 0.3); border-radius: 8px; padding: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# AERIS")
st.markdown('<p class="subtitle">🛰️ Advanced Cloud Detection System for Satellite Imagery<br/>Leveraging Deep Learning for Precise Cloud Segmentation with Uncertainty Quantification</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### 🎯 PERFORMANCE")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("IoU", "92.2%")
        st.metric("Dice", "94.3%")
    with col2:
        st.metric("ECE", "0.0070")
        st.metric("Acc", "95.8%")
    
    st.markdown("---")
    st.markdown("#### 🏗️ ARCHITECTURE")
    st.markdown("""<div style='color: #9fa8da; font-size: 0.85em; line-height: 1.6;'>
    <b style='color: #c5cae9;'>Encoder:</b> ResNet34 (ImageNet)<br/>
    <b style='color: #c5cae9;'>Decoder:</b> U-Net<br/>
    <b style='color: #c5cae9;'>Input:</b> 4-band (R,G,B,NIR)<br/>
    <b style='color: #c5cae9;'>Output:</b> Probability (0-1)<br/>
    <b style='color: #c5cae9;'>Uncertainty:</b> MC Dropout (30 iter)
    </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📊 DATASET")
    st.markdown("""<div style='color: #9fa8da; font-size: 0.85em; line-height: 1.6;'>
    <b style='color: #c5cae9;'>38-Cloud</b> | Landsat 8<br/>
    Training: 8,400 patches<br/>
    Validation: 9,201 patches<br/>
    Resolution: 256×256 px
    </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🔗 RESOURCES")
    st.markdown("🤗 [Model Hub](https://huggingface.co/Eklavya16/aeris-cloud-detection)")
    st.markdown("📦 [Dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images)")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'Aeris_Model.pth'
    
    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model from HuggingFace..."):
            try:
                model_path = hf_hub_download(
                    repo_id="Eklavya16/aeris-cloud-detection",
                    filename="Aeris_Model.pth"
                )
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None, device
    
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=4,
        classes=1,
        activation=None
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

def predict_with_uncertainty(model, image, device, n_iter=30):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    model.eval()
    enable_dropout(model)
    
    preds = []
    progress = st.progress(0, text="🔄 Running MC Dropout predictions...")
    for i in range(n_iter):
        with torch.no_grad():
            output = model(image)
            pred = torch.sigmoid(output)
            preds.append(pred.cpu().numpy())
        progress.progress((i + 1) / n_iter, text=f"Iteration {i+1}/{n_iter}")
    
    progress.empty()
    preds = np.array(preds)
    return preds.mean(axis=0).squeeze(), preds.std(axis=0).squeeze()

def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = np.clip(img_tensor[:3] * std + mean, 0, 1)
    return np.transpose(img_np, (1, 2, 0))

def process_image(image_array, model, device):
    img_tensor = torch.from_numpy(image_array).float()
    
    if img_tensor.max() > 1:
        img_tensor = img_tensor / 255.0
    
    if img_tensor.shape[0] != 4:
        st.error(f"Expected 4 channels, got {img_tensor.shape[0]}")
        return None
    
    if img_tensor.shape[1] != 256 or img_tensor.shape[2] != 256:
        import torch.nn.functional as F
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.squeeze(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.225]).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    pred, unc = predict_with_uncertainty(model, img_tensor, device)
    rgb = denormalize(img_tensor.numpy())
    
    return pred, unc, rgb

def calculate_metrics(pred, unc):
    binary = (pred > 0.5).astype(float)
    coverage = binary.mean() * 100
    
    cloud_mask = binary == 1
    clear_mask = binary == 0
    
    if cloud_mask.sum() > 0:
        cloud_conf = pred[cloud_mask].mean() * 100
    else:
        cloud_conf = 0
        
    if clear_mask.sum() > 0:
        clear_conf = (1 - pred[clear_mask]).mean() * 100
    else:
        clear_conf = 0
    
    confidence_map = np.where(binary == 1, pred, 1 - pred)
    overall_conf = confidence_map.mean() * 100
    
    mean_unc = unc.mean()
    quality = ((unc < 0.1).sum() / unc.size) * 100
    reliability = max(0, min(100, (1 - mean_unc * 2) * 100))
    
    if coverage < 10:
        category = "Clear Sky"
    elif coverage < 35:
        category = "Few Clouds"
    elif coverage < 65:
        category = "Partly Cloudy"
    elif coverage < 85:
        category = "Mostly Cloudy"
    else:
        category = "Overcast"
    
    quality_label = "Excellent" if quality >= 90 else "Very Good" if quality >= 75 else "Good" if quality >= 60 else "Fair" if quality >= 40 else "Poor"
    reliability_label = "Excellent" if reliability >= 95 else "Very Good" if reliability >= 85 else "Good" if reliability >= 70 else "Fair" if reliability >= 50 else "Poor"
    
    return {
        'coverage': coverage,
        'category': category,
        'cloud_conf': cloud_conf,
        'clear_conf': clear_conf,
        'overall_conf': overall_conf,
        'quality': quality,
        'quality_label': quality_label,
        'reliability': reliability,
        'reliability_label': reliability_label,
        'mean_unc': mean_unc,
        'max_unc': unc.max(),
        'binary': binary,
        'cloud_pixels': int(cloud_mask.sum()),
        'clear_pixels': int(clear_mask.sum())
    }

def create_visualization(rgb, binary, pred, unc, metrics):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    fig.patch.set_facecolor('#0a0e27')
    
    axes = [fig.add_subplot(gs[i//4, i%4]) for i in range(8)]
    for ax in axes:
        ax.set_facecolor('#16213e')
    
    axes[0].imshow(rgb)
    axes[0].set_title('Satellite Image (RGB)', fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[0].axis('off')
    
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Cloud Mask\n{metrics["category"]} ({metrics["coverage"]:.1f}%)', 
                      fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[1].axis('off')
    
    im2 = axes[2].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[2].set_title('Cloud Probability', fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors='#9fa8da')
    
    overlay = rgb.copy()
    overlay[pred > 0.5] = overlay[pred > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[3].imshow(overlay)
    axes[3].set_title('Cloud Overlay', fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[3].axis('off')
    
    im4 = axes[4].imshow(unc, cmap='hot', vmin=0, vmax=unc.max())
    axes[4].set_title(f'Uncertainty Map\nMean: {metrics["mean_unc"]:.3f}', 
                      fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[4].axis('off')
    cbar4 = plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(colors='#9fa8da')
    
    confidence_map = np.where(pred > 0.5, pred, 1 - pred)
    im5 = axes[5].imshow(confidence_map, cmap='viridis', vmin=0.5, vmax=1)
    axes[5].set_title(f'Confidence Map\n{metrics["overall_conf"]:.1f}% avg', 
                      fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[5].axis('off')
    cbar5 = plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    cbar5.ax.tick_params(colors='#9fa8da')
    
    edges = sobel(binary)
    edge_overlay = rgb.copy()
    edge_overlay[edges > 0] = [1, 0.84, 0]
    axes[6].imshow(edge_overlay)
    axes[6].set_title('Cloud Boundaries', fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[6].axis('off')
    
    quality_map = np.zeros_like(pred)
    quality_map[unc < 0.05] = 3
    quality_map[(unc >= 0.05) & (unc < 0.1)] = 2
    quality_map[(unc >= 0.1) & (unc < 0.2)] = 1
    
    im7 = axes[7].imshow(quality_map, cmap='RdYlGn', vmin=0, vmax=3)
    axes[7].set_title(f'Quality Zones\n{metrics["quality_label"]}', 
                      fontsize=11, color='#c5cae9', pad=10, weight='light')
    axes[7].axis('off')
    cbar7 = plt.colorbar(im7, ax=axes[7], fraction=0.046, pad=0.04)
    cbar7.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar7.set_ticklabels(['Poor', 'Fair', 'Good', 'Excellent'])
    cbar7.ax.tick_params(colors='#9fa8da')
    
    return fig

def load_separate_bands(red, green, blue, nir):
    try:
        bands = []
        for f in [red, green, blue, nir]:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            band = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if band is None:
                return None
            bands.append(band)
        return np.stack(bands, axis=0)
    except Exception as e:
        st.error(f"Error loading bands: {e}")
        return None

def main():
    model, device = load_model()
    if model is None:
        st.stop()
    
    st.markdown("## 📡 Upload Satellite Imagery")
    
    tab1, tab2 = st.tabs(["4-Channel File", "Separate Bands"])
    
    with tab1:
        st.caption("Single TIF/TIFF file containing R, G, B, NIR bands")
        uploaded = st.file_uploader("", type=['tif', 'tiff', 'npy'], key="single")
        method = "single" if uploaded else None
    
    with tab2:
        st.caption("Upload 4 separate single-band TIFF files")
        col1, col2 = st.columns(2)
        with col1:
            red = st.file_uploader("Red Band", type=['tif', 'tiff'], key="red")
            blue = st.file_uploader("Blue Band", type=['tif', 'tiff'], key="blue")
        with col2:
            green = st.file_uploader("Green Band", type=['tif', 'tiff'], key="green")
            nir = st.file_uploader("NIR Band", type=['tif', 'tiff'], key="nir")
        
        if all([red, green, blue, nir]):
            method = "separate"
            st.success("✓ All bands loaded successfully")
        elif any([red, green, blue, nir]):
            method = None
            st.warning("⚠ Please upload all 4 bands to proceed")
        else:
            method = None
    
    st.markdown("")
    if st.button("🔍 DETECT CLOUDS", use_container_width=True, type="primary"):
        if method is None:
            st.warning("⚠ Please upload imagery to proceed")
            return
        
        with st.spinner("🛰️ Analyzing satellite data..."):
            try:
                if method == "separate":
                    image_array = load_separate_bands(red, green, blue, nir)
                    if image_array is None:
                        return
                elif method == "single":
                    if uploaded.name.endswith('.npy'):
                        image_array = np.load(uploaded)
                    else:
                        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                        if img is None or img.ndim != 3 or img.shape[2] != 4:
                            st.error("❌ Invalid file format. Expected 4-channel image.")
                            return
                        image_array = np.transpose(img, (2, 0, 1))
                
                result = process_image(image_array, model, device)
                if result is None:
                    return
                
                pred, unc, rgb = result
                m = calculate_metrics(pred, unc)
                
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Coverage</h3>
                        <h1>{m['coverage']:.1f}%</h1>
                        <p>{m['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence</h3>
                        <h1>{m['overall_conf']:.1f}%</h1>
                        <p>Overall</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Quality</h3>
                        <h1>{m['quality_label']}</h1>
                        <p>{m['quality']:.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Reliability</h3>
                        <h1>{m['reliability_label']}</h1>
                        <p>{m['reliability']:.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Cloud Pixels</h3>
                        <h1>{m['cloud_pixels']:,}</h1>
                        <p>of {m['cloud_pixels']+m['clear_pixels']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown("### 🎨 Comprehensive Analysis")
                fig = create_visualization(rgb, m['binary'], pred, unc, m)
                st.pyplot(fig)
                
                st.markdown("### 📈 Detailed Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**☁️ Cloud Analysis**")
                    st.metric("Cloud Confidence", f"{m['cloud_conf']:.1f}%")
                    st.metric("Clear Confidence", f"{m['clear_conf']:.1f}%")
                    st.metric("Total Pixels", f"{m['cloud_pixels']+m['clear_pixels']:,}")
                
                with col2:
                    st.markdown("**📊 Uncertainty Metrics**")
                    st.metric("Mean Uncertainty", f"{m['mean_unc']:.4f}")
                    st.metric("Maximum Uncertainty", f"{m['max_unc']:.4f}")
                    interp = "Very confident" if m['mean_unc'] < 0.05 else "Confident" if m['mean_unc'] < 0.1 else "Moderate" if m['mean_unc'] < 0.2 else "Review needed"
                    st.caption(f"*{interp} predictions*")
                
                with col3:
                    st.markdown("**⚙️ System Info**")
                    st.metric("Compute Device", str(device).upper())
                    st.metric("Architecture", "U-Net + ResNet34")
                    st.metric("MC Iterations", "30")
                
                with st.expander("🔬 Technical Details"):
                    st.markdown("""
                    **Architecture:** U-Net decoder + ResNet34 encoder (4-channel: R,G,B,NIR)
                    
                    **Uncertainty:** MC Dropout (30 iterations) provides pixel-level confidence
                    
                    **Performance:** IoU 92.2% | Dice 94.3% | ECE 0.0070 | Acc 95.8%
                    """)
                
                with st.expander("📖 Interpretation Guide"):
                    st.markdown(f"""
                    **Analysis: {m['category']}** with {m['coverage']:.1f}% cloud coverage
                    
                    **Quality:** {m['quality_label']} ({m['quality']:.0f}/100) | **Reliability:** {m['reliability_label']} ({m['reliability']:.0f}/100)
                    
                    **Confidence:** {m['overall_conf']:.1f}% overall | Mean uncertainty: {m['mean_unc']:.4f}
                    
                    ✅ **High quality:** Quality > 75%, Uncertainty < 0.1
                    
                    ⚠️ **Review needed:** Quality < 60%, Uncertainty > 0.15
                    
                    **Check:** Uncertainty map (hot spots), Quality zones (green = reliable), Cloud boundaries
                    """)
                
                st.success("✅ Analysis complete")
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)

if __name__ == "__main__":

    main()
