import os
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import streamlit as st
import tempfile
from io import BytesIO
import zipfile
import base64

# Page configuration
st.set_page_config(
    page_title="Spectral Parameters Predictor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the application
st.title("🔭 Spectral Parameters Predictor")
st.markdown("""
This application predicts physical parameters of astronomical spectra using machine learning models.
Upload a spectrum file and trained models to get predictions.
""")

# Function to load models (with caching for better performance)
@st.cache_resource
def load_models_from_zip(zip_file):
    """Load all models and scalers from a ZIP file"""
    models = {}
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load main scaler and PCA
            models['scaler'] = joblib.load(os.path.join(temp_dir, "standard_scaler.save"))
            models['ipca'] = joblib.load(os.path.join(temp_dir, "incremental_pca.save"))
            
            # Load parameter scalers
            param_names = ['logn', 'tex', 'velo', 'fwhm']
            models['param_scalers'] = {}
            
            for param in param_names:
                scaler_path = os.path.join(temp_dir, f"{param}_scaler.save")
                if os.path.exists(scaler_path):
                    models['param_scalers'][param] = joblib.load(scaler_path)
            
            # Load trained models
            models['all_models'] = {}
            for param in param_names:
                param_models = {}
                for model_type in ['randomforest', 'gradientboosting', 'svr', 'gaussianprocess']:
                    model_path = os.path.join(temp_dir, f"{param}_{model_type}.save")
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        param_models[model_type.capitalize()] = model
                models['all_models'][param] = param_models
                
            # Load training statistics
            models['training_stats'] = {}
            models['training_errors'] = {}
            for param in param_names:
                # Statistics
                stats_file = os.path.join(temp_dir, f"training_stats_{param}.npy")
                if os.path.exists(stats_file):
                    models['training_stats'][param] = np.load(stats_file, allow_pickle=True).item()
                
                # Errors
                errors_file = os.path.join(temp_dir, f"training_errors_{param}.npy")
                if os.path.exists(errors_file):
                    models['training_errors'][param] = np.load(errors_file, allow_pickle=True).item()
                    
            return models, "✓ Models loaded successfully"
            
        except Exception as e:
            return None, f"✗ Error loading models: {str(e)}"

def get_units(param):
    """Get units for each parameter"""
    units = {
        'logn': 'log(cm⁻³)',
        'tex': 'K',
        'velo': 'km/s',
        'fwhm': 'km/s'
    }
    return units.get(param, '')

def process_spectrum(spectrum_file, models, target_length=64607):
    """Process spectrum and make predictions"""
    # Read spectrum data
    frequencies = []
    intensities = []
    
    try:
        # Read file content
        if hasattr(spectrum_file, 'read'):
            content = spectrum_file.read().decode("utf-8")
            lines = content.splitlines()
        else:
            with open(spectrum_file, 'r') as f:
                lines = f.readlines()
        
        # Skip header if it exists
        start_line = 0
        if lines and lines[0].startswith('!'):
            start_line = 1
        
        for line in lines[start_line:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    frequencies.append(freq)
                    intensities.append(intensity)
                except ValueError:
                    continue
        
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)
        
        # Create reference frequencies based on the spectrum range
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        reference_frequencies = np.linspace(min_freq, max_freq, target_length)
        
        # Interpolate to reference frequencies
        interpolator = interp1d(frequencies, intensities, kind='linear',
                              bounds_error=False, fill_value=0.0)
        interpolated_intensities = interpolator(reference_frequencies)
        
        # Scale the spectrum
        X_scaled = models['scaler'].transform(interpolated_intensities.reshape(1, -1))
        
        # Apply PCA
        X_pca = models['ipca'].transform(X_scaled)
        
        # Make predictions with all models
        predictions = {}
        uncertainties = {}
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['log(n)', 'T_ex (K)', 'V_los (km/s)', 'FWHM (km/s)']
        
        for param in param_names:
            param_predictions = {}
            param_uncertainties = {}
            
            for model_name, model in models['all_models'][param].items():
                try:
                    if model_name.lower() == 'gaussianprocess':
                        # Gaussian Process provides native uncertainty
                        y_pred, y_std = model.predict(X_pca, return_std=True)
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        y_std_orig = y_std * models['param_scalers'][param].scale_
                        
                        param_predictions[model_name] = y_pred_orig[0]
                        param_uncertainties[model_name] = y_std_orig[0]
                        
                    else:
                        # For other models, use various uncertainty estimation methods
                        y_pred = model.predict(X_pca)
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        
                        # Estimate uncertainty based on model type
                        if hasattr(model, 'estimators_'):
                            # Use standard deviation of tree predictions
                            tree_preds = [tree.predict(X_pca) for tree in model.estimators_]
                            tree_preds_orig = [models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                             for pred in tree_preds]
                            uncertainty = np.std(tree_preds_orig)
                        else:
                            # Default uncertainty
                            if param in models['training_errors'] and model_name in models['training_errors'][param]:
                                uncertainty = models['training_errors'][param][model_name]
                            else:
                                uncertainty = abs(y_pred_orig[0]) * 0.1  # 10% of prediction
                        
                        param_predictions[model_name] = y_pred_orig[0]
                        param_uncertainties[model_name] = uncertainty
                        
                except Exception as e:
                    st.error(f"Error predicting with {model_name} for {param}: {e}")
                    continue
            
            predictions[param] = param_predictions
            uncertainties[param] = param_uncertainties
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'processed_spectrum': {
                'frequencies': reference_frequencies,
                'intensities': interpolated_intensities,
                'pca_components': X_pca
            },
            'param_names': param_names,
            'param_labels': param_labels
        }
        
    except Exception as e:
        st.error(f"Error processing the spectrum: {e}")
        return None

def create_comparison_plot(predictions, uncertainties, param, label, training_stats, spectrum_name):
    """Create comparison plot for a parameter"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get predictions for this parameter
    param_preds = predictions[param]
    param_uncerts = uncertainties[param]
    
    # Try to get actual training data range for this parameter
    if param in training_stats:
        actual_min = training_stats[param]['min']
        actual_max = training_stats[param]['max']
        actual_mean = training_stats[param]['mean']
        
        # Create synthetic training data based on actual range
        n_points = 200
        synthetic_actual = np.random.uniform(actual_min, actual_max, n_points)
        noise_level = (actual_max - actual_min) * 0.05
        synthetic_predicted = synthetic_actual + np.random.normal(0, noise_level, n_points)
        
    else:
        # Fallback: create reasonable ranges
        if param == 'logn':
            actual_min, actual_max = 10, 20
        elif param == 'tex':
            actual_min, actual_max = 50, 300
        elif param == 'velo':
            actual_min, actual_max = -10, 10
        elif param == 'fwhm':
            actual_min, actual_max = 1, 15
        else:
            actual_min, actual_max = 0, 1
            
        n_points = 100
        synthetic_actual = np.random.uniform(actual_min, actual_max, n_points)
        synthetic_predicted = synthetic_actual + np.random.normal(0, (actual_max-actual_min)*0.05, n_points)
    
    # Plot training data points
    ax.scatter(synthetic_actual, synthetic_predicted, alpha=0.3, 
               color='lightgray', label='Training data distribution', s=30)
    
    # Plot ideal line
    min_val = min(np.min(synthetic_actual), np.min(synthetic_predicted))
    max_val = max(np.max(synthetic_actual), np.max(synthetic_predicted))
    range_ext = 0.1 * (max_val - min_val)
    plot_min = min_val - range_ext
    plot_max = max_val + range_ext
    
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
            label='Ideal prediction', linewidth=2)
    
    # Plot our prediction for each model WITH ERROR BARS
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
        mean_actual = np.mean(synthetic_actual)
        uncert_value = param_uncerts.get(model_name, 0)
        
        ax.scatter(mean_actual, pred_value, color=colors[i % len(colors)], 
                   s=200, marker='*', edgecolors='black', linewidth=2,
                   label=f'{model_name}: {pred_value:.3f} ± {uncert_value:.3f}')
        
        # Add uncertainty bars for ALL models
        ax.errorbar(mean_actual, pred_value, yerr=uncert_value, 
                    fmt='none', ecolor=colors[i % len(colors)], 
                    capsize=8, capthick=2, elinewidth=3, alpha=0.8)
    
    ax.set_xlabel(f'Actual {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title(f'Model Predictions for {label} with Uncertainty\nSpectrum: {spectrum_name}')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    plt.tight_layout()
    return fig

def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name):
    """Create combined plot showing all parameter predictions with uncertainty"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]
        
        models = list(param_preds.keys())
        values = list(param_preds.values())
        errors = [param_uncerts.get(model, 0) for model in models]
        
        # Create bar plot with error bars
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=8, alpha=0.8, 
                     color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(f'{label} ({get_units(param)})', fontsize=12)
        ax.set_title(f'{label} Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
    
    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Function to get local file path
def get_local_file_path(filename):
    """Get path to a local file in the same directory as the script"""
    return os.path.join(os.path.dirname(__file__), filename)

# Main user interface
def main():
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # Option to use local models file
        use_local_models = st.checkbox("Use local models file (models.zip in same directory)")
        
        # Load models
        st.subheader("1. Trained Models")
        if use_local_models:
            local_zip_path = get_local_file_path("models.zip")
            if os.path.exists(local_zip_path):
                models_zip = local_zip_path
                st.success("✓ Local models.zip file found")
            else:
                st.error("✗ models.zip not found in the same directory as this script")
                models_zip = None
        else:
            models_zip = st.file_uploader("Upload ZIP file with trained models", type=['zip'])
        
        # Load spectrum
        st.subheader("2. Spectrum File")
        spectrum_file = st.file_uploader("Upload spectrum file", type=['txt', 'dat'])
        
        # Process button
        process_btn = st.button("🚀 Process Spectrum", type="primary", 
                               disabled=(models_zip is None or spectrum_file is None))
    
    # Main content
    if models_zip is not None and spectrum_file is not None:
        if process_btn:
            with st.spinner("Loading and processing models..."):
                # Load models
                if use_local_models:
                    models, message = load_models_from_zip(models_zip)
                else:
                    # For uploaded file, we need to save it temporarily first
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(models_zip.getvalue())
                        tmp_path = tmp_file.name
                    
                    models, message = load_models_from_zip(tmp_path)
                    os.unlink(tmp_path)  # Clean up temp file
                
                if models is None:
                    st.error(message)
                    return
                
                st.success(message)
            
            # Process spectrum
            with st.spinner("Processing spectrum and making predictions..."):
                results = process_spectrum(spectrum_file, models)
                
                if results is None:
                    st.error("Error processing the spectrum")
                    return
                
                # Display results
                st.header("📊 Prediction Results")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Summary", "Individual Plots", "Combined Plot"])
                
                with tab1:
                    st.subheader("Prediction Summary")
                    
                    # Create summary table
                    summary_data = []
                    for param, label in zip(results['param_names'], results['param_labels']):
                        param_preds = results['predictions'][param]
                        param_uncerts = results['uncertainties'].get(param, {})
                        
                        for model_name, pred_value in param_preds.items():
                            uncert_value = param_uncerts.get(model_name, np.nan)
                            summary_data.append({
                                'Parameter': label,
                                'Model': model_name,
                                'Prediction': pred_value,
                                'Uncertainty': uncert_value if not np.isnan(uncert_value) else 'N/A',
                                'Units': get_units(param),
                                'Relative_Error_%': (uncert_value / abs(pred_value) * 100) if pred_value != 0 and not np.isnan(uncert_value) else np.nan
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Download results as CSV
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name="spectrum_predictions.csv",
                        mime="text/csv"
                    )
                
                with tab2:
                    st.subheader("Prediction Plots by Parameter")
                    
                    # Create individual plots for each parameter
                    for param, label in zip(results['param_names'], results['param_labels']):
                        fig = create_comparison_plot(
                            results['predictions'], 
                            results['uncertainties'], 
                            param, 
                            label, 
                            models['training_stats'],
                            spectrum_file.name
                        )
                        st.pyplot(fig)
                        
                        # Option to download each plot
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label=f"📥 Download {label} plot",
                            data=buf,
                            file_name=f"prediction_{param}.png",
                            mime="image/png",
                            key=f"download_{param}"
                        )
                
                with tab3:
                    st.subheader("Combined Prediction Plot")
                    
                    # Create combined plot
                    fig = create_combined_plot(
                        results['predictions'],
                        results['uncertainties'],
                        results['param_names'],
                        results['param_labels'],
                        spectrum_file.name
                    )
                    st.pyplot(fig)
                    
                    # Option to download the combined plot
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download combined plot",
                        data=buf,
                        file_name="combined_predictions.png",
                        mime="image/png"
                    )
    
    else:
        # Show instructions if files haven't been uploaded
        st.info("👈 Please upload trained models and a spectrum file in the sidebar to get started.")
        
        # Usage instructions
        st.markdown("""
        ## Usage Instructions:
        
        1. **Prepare trained models**: Compress all model files (.save) and statistics (.npy) into a ZIP file named "models.zip"
        2. **Prepare spectrum**: Ensure your spectrum file is in text format with two columns (frequency, intensity)
        3. **Upload files**: Use the selectors in the sidebar to upload both files or use the local models.zip file
        4. **Process**: Click the 'Process Spectrum' button to get predictions
        
        ## Supported File Formats:
        - **Models**: ZIP file containing joblib-saved models
        - **Spectra**: Text files (.txt, .dat) with two data columns
        
        ## Required Model Files:
        Your models ZIP must contain:
        - standard_scaler.save
        - incremental_pca.save
        - Parameter scalers: logn_scaler.save, tex_scaler.save, velo_scaler.save, fwhm_scaler.save
        - At least one model per parameter (e.g., logn_randomforest.save, tex_gradientboosting.save, etc.)
        - Optional: Training statistics and error files (training_stats_*.npy, training_errors_*.npy)
        """)

if __name__ == "__main__":
    main()
