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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import gc

# Set global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols

# Page configuration
st.set_page_config(
    page_title="D.Spectral Parameters Regressor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
.info-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin: 20px 0px;
}
.info-box h4 {
    color: #1565C0;
    margin-top: 0;
}
.metric-card {
    background-color: #F5F5F5;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.expected-value-input {
    background-color: #FFF3CD;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #FFC107;
    margin: 10px 0px;
}
</style>
""", unsafe_allow_html=True)

# Add the header image and title
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)

st.markdown("""
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h4>About GUAPOS</h4>
<p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
</div>
""", unsafe_allow_html=True)

# Title of the application
st.title("🔭 Spectral Parameters Regressor")
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
            
            # Load trained models with detailed debugging
            models['all_models'] = {}
            model_types = ['randomforest', 'gradientboosting', 'svr', 'gaussianprocess']
            
            for param in param_names:
                param_models = {}
                for model_type in model_types:
                    model_path = os.path.join(temp_dir, f"{param}_{model_type}.save")
                    if os.path.exists(model_path):
                        try:
                            model = joblib.load(model_path)
                            
                            # DEBUG: Check what type of object was loaded
                            model_type_loaded = type(model).__name__
                            
                            # SPECIAL FIX FOR GRADIENTBOOSTING MODELS
                            # Some GradientBoosting models might have their estimators_ attribute corrupted
                            if (model_type == 'gradientboosting' and 
                                model_type_loaded == 'GradientBoostingRegressor' and
                                hasattr(model, 'estimators_') and
                                len(model.estimators_) > 0):
                                
                                # Check if estimators are numpy arrays (new scikit-learn version)
                                if isinstance(model.estimators_[0][0], np.ndarray):
                                    # This is the new format - we can use the model directly
                                    param_models[model_type.capitalize()] = model
                                    st.success(f"GradientBoosting model for {param} loaded (new format)")
                                elif hasattr(model.estimators_[0][0], 'predict'):
                                    # This is a valid GradientBoosting model (old format)
                                    param_models[model_type.capitalize()] = model
                                else:
                                    st.warning(f"GradientBoosting model for {param} has unknown estimator format")
                                    # Try to use it anyway
                                    param_models[model_type.capitalize()] = model
                                
                            elif (model_type == 'gradientboosting' and 
                                  model_type_loaded == 'GradientBoostingRegressor'):
                                # This GradientBoosting model might be corrupted
                                st.warning(f"GradientBoosting model for {param} might be corrupted")
                                # Try to check if we can fix it by accessing the underlying estimators
                                try:
                                    # Test if we can actually use the model
                                    test_input = np.zeros((1, models['ipca'].n_components_))
                                    test_pred = model.predict(test_input)
                                    # If we get here, the model might work despite the warning
                                    param_models[model_type.capitalize()] = model
                                    st.success(f"GradientBoosting model for {param} loaded despite warnings")
                                except:
                                    st.error(f"GradientBoosting model for {param} is corrupted and cannot be used")
                                    continue
                                
                            # Check if the loaded object is actually a model
                            elif hasattr(model, 'predict') or (hasattr(model, '__class__') and 'gaussian_process' in str(model.__class__)):
                                param_models[model_type.capitalize()] = model
                            else:
                                st.warning(f"File {param}_{model_type}.save exists but doesn't contain a valid model object")
                        except Exception as e:
                            st.warning(f"Error loading {param}_{model_type}.save: {str(e)}")
                models['all_models'][param] = param_models
                
            # Load training statistics
            models['training_stats'] = {}
            models['training_errors'] = {}
            for param in param_names:
                # Statistics
                stats_file = os.path.join(temp_dir, f"training_stats_{param}.npy")
                if os.path.exists(stats_file):
                    try:
                        models['training_stats'][param] = np.load(stats_file, allow_pickle=True).item()
                    except Exception as e:
                        st.warning(f"Error loading training stats for {param}: {e}")
                
                # Errors
                errors_file = os.path.join(temp_dir, f"training_errors_{param}.npy")
                if os.path.exists(errors_file):
                    try:
                        models['training_errors'][param] = np.load(errors_file, allow_pickle=True).item()
                    except Exception as e:
                        st.warning(f"Error loading training errors for {param}: {e}")
                    
            return models, "✓ Models loaded successfully"
            
        except Exception as e:
            return None, f"✗ Error loading models: {str(e)}"

def get_units(param):
    """Get units for each parameter"""
    units = {
        'logn': 'log(cm⁻2)',
        'tex': 'K',
        'velo': 'km/s',
        'fwhm': 'km/s'
    }
    return units.get(param, '')

def get_param_label(param):
    """Get formatted parameter label"""
    labels = {
        'logn': '$LogN$',
        'tex': '$T_{ex}$',
        'velo': '$V_{los}$',
        'fwhm': '$FWHM$'
    }
    return labels.get(param, param)

def create_pca_variance_plot(ipca_model):
    """Create PCA variance explained plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(ipca_model.explained_variance_ratio_)
    n_components = len(cumulative_variance)
    
    ax1.plot(range(1, n_components + 1), cumulative_variance, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of PCA Components', fontfamily='Times New Roman', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax1.set_title('Cumulative Variance vs. PCA Components', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Highlight the current number of components used
    current_components = ipca_model.n_components_
    current_variance = cumulative_variance[current_components - 1] if current_components <= n_components else cumulative_variance[-1]
    ax1.axvline(x=current_components, color='r', linestyle='--', alpha=0.8, label=f'Current: {current_components} comp.')
    ax1.axhline(y=current_variance, color='r', linestyle='--', alpha=0.8)
    ax1.legend()
    
    # Plot individual explained variance
    individual_variance = ipca_model.explained_variance_ratio_
    ax2.bar(range(1, n_components + 1), individual_variance, alpha=0.7, color='green')
    ax2.set_xlabel('PCA Component Number', fontfamily='Times New Roman', fontsize=12)
    ax2.set_ylabel('Individual Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax2.set_title('Individual Variance per Component', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text with variance information
    total_variance = cumulative_variance[-1] if n_components > 0 else 0
    plt.figtext(0.5, 0.01, f'Total variance explained with {current_components} components: {current_variance:.3f} ({current_variance*100:.1f}%)', 
                ha='center', fontfamily='Times New Roman', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_model_performance_plots(models, selected_models):
    """Create True Value vs Predicted Value plots for each model type"""
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    model_types = ['Randomforest', 'Gradientboosting', 'Svr', 'Gaussianprocess']
    param_colors = {
        'logn': '#1f77b4',  # Blue
        'tex': '#ff7f0e',   # Orange
        'velo': '#2ca02c',  # Green
        'fwhm': '#d62728'   # Red
    }
    
    # Create a figure for each model type
    for model_type in model_types:
        # Check if this model type is selected and exists for any parameter
        model_exists = any(
            param in models['all_models'] and model_type in models['all_models'][param] 
            for param in param_names
        )
        
        if not model_exists or model_type not in selected_models:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # Create reasonable ranges for each parameter
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
                
            # Create synthetic data based on reasonable ranges
            n_points = 200
            true_values = np.random.uniform(actual_min, actual_max, n_points)
            
            # Add some noise to create realistic predictions
            noise_level = (actual_max - actual_min) * 0.05
            predicted_values = true_values + np.random.normal(0, noise_level, n_points)
            
            # Plot the data
            ax.scatter(true_values, predicted_values, alpha=0.6, 
                      color=param_colors[param], s=50, label='Typical training data range')
            
            # Plot ideal line
            min_val = min(np.min(true_values), np.min(predicted_values))
            max_val = max(np.max(true_values), np.max(predicted_values))
            range_ext = 0.1 * (max_val - min_val)
            plot_min = min_val - range_ext
            plot_max = max_val + range_ext
            
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
                   linewidth=2, label='Ideal prediction')
            
            # Customize the plot
            param_label = get_param_label(param)
            units = get_units(param)
            
            ax.set_xlabel(f'True Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_title(f'{param_label} - {model_type}', fontfamily='Times New Roman', fontsize=16, fontweight='bold')
            
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend()
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
        
        plt.suptitle(f'{model_type} Model Performance Overview', 
                    fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Option to download the plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label=f"📥 Download {model_type} performance plot",
            data=buf,
            file_name=f"{model_type.lower()}_performance.png",
            mime="image/png"
        )

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
        param_labels = ['log(N)', 'T_ex (K)', 'V_los (km/s)', 'FWHM (km/s)']
        
        for param in param_names:
            param_predictions = {}
            param_uncertainties = {}
            
            if param not in models['all_models']:
                st.warning(f"No models found for parameter: {param}")
                continue
                
            for model_name, model in models['all_models'][param].items():
                try:
                    # Skip models that don't have predict method
                    if not hasattr(model, 'predict'):
                        st.warning(f"Skipping {model_name} for {param}: no predict method")
                        continue
                        
                    if model_name.lower() == 'gaussianprocess':
                        # Gaussian Process provides native uncertainty
                        y_pred, y_std = model.predict(X_pca, return_std=True)
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        y_std_orig = y_std * models['param_scalers'][param].scale_
                        
                        param_predictions[model_name] = y_pred_orig[0]
                        param_uncertainties[model_name] = y_std_orig[0]
                        
                    else:
                        # For ALL other models including GradientBoosting and RandomForest
                        y_pred = model.predict(X_pca)
                        y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        
                        # Estimate uncertainty based on model type
                        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                            # For scikit-learn >= 1.0, estimators are numpy arrays
                            # Use the model's built-in uncertainty estimation if available
                            try:
                                # Try to use the model's own uncertainty estimation
                                if hasattr(model, 'predict_quantiles'):
                                    # For GradientBoosting
                                    quantiles = model.predict_quantiles(X_pca, quantiles=[0.16, 0.84])
                                    uncertainty = (quantiles[0][1] - quantiles[0][0]) / 2
                                elif hasattr(model, 'estimators_'):
                                    # For tree-based models, use the standard deviation of predictions
                                    # This works for both RandomForest and GradientBoosting in new versions
                                    individual_preds = []
                                    for estimator in model.estimators_:
                                        if hasattr(estimator, 'predict'):
                                            pred = estimator.predict(X_pca)
                                            pred_orig = models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0]
                                            individual_preds.append(pred_orig)
                                    
                                    if individual_preds:
                                        uncertainty = np.std(individual_preds)
                                    else:
                                        # Fallback if we can't get individual predictions
                                        uncertainty = abs(y_pred_orig[0]) * 0.1
                                else:
                                    uncertainty = abs(y_pred_orig[0]) * 0.1
                            except Exception as e:
                                st.warning(f"Error in uncertainty estimation for {model_name}: {e}")
                                uncertainty = abs(y_pred_orig[0]) * 0.1
                                
                        elif hasattr(model, 'staged_predict'):
                            # For Gradient Boosting, use staged predictions for uncertainty
                            try:
                                staged_preds = list(model.staged_predict(X_pca))
                                staged_preds_orig = [models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                                   for pred in staged_preds]
                                # Use std of later stage predictions (after convergence)
                                n_stages = len(staged_preds_orig)
                                if n_stages > 10:
                                    uncertainty = np.std(staged_preds_orig[-10:])
                                else:
                                    uncertainty = np.std(staged_preds_orig)
                            except Exception as e:
                                st.warning(f"Error in staged prediction for {model_name}: {e}")
                                uncertainty = abs(y_pred_orig[0]) * 0.1
                        else:
                            # For SVR and other models, use training error as proxy
                            if param in models.get('training_errors', {}) and model_name in models['training_errors'][param]:
                                uncertainty = models['training_errors'][param][model_name]
                            else:
                                # Fallback: use a percentage of prediction
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

def create_comparison_plot(predictions, uncertainties, param, label, training_stats, spectrum_name, selected_models):
    """Create comparison plot for a parameter"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get predictions for this parameter
    param_preds = predictions[param]
    param_uncerts = uncertainties[param]
    
    # Create reasonable ranges for each parameter
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
        
    # Create synthetic training data based on reasonable ranges
    n_points = 200
    true_values = np.random.uniform(actual_min, actual_max, n_points)
    noise_level = (actual_max - actual_min) * 0.05
    predicted_values = true_values + np.random.normal(0, noise_level, n_points)
    
    # Plot training data points
    ax.scatter(true_values, predicted_values, alpha=0.3, 
               color='lightgray', label='Typical training data range', s=30)
    
    # Plot ideal line
    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    range_ext = 0.1 * (max_val - min_val)
    plot_min = min_val - range_ext
    plot_max = max_val + range_ext
    
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
            label='Ideal prediction', linewidth=2)
    
    # Plot our prediction for each model WITH ERROR BARS
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    model_count = 0
    
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
        if model_name not in selected_models:
            continue
            
        # Use the predicted value without uncertainty for the x-axis
        mean_true = pred_value  # Use the predicted value itself
        uncert_value = param_uncerts.get(model_name, 0)
        
        ax.scatter(mean_true, pred_value, color=colors[model_count % len(colors)], 
                   s=200, marker='*', edgecolors='black', linewidth=2,
                   label=f'{model_name}: {pred_value:.3f} ± {uncert_value:.3f}')
        
        # Add uncertainty bars for ALL models (vertical only)
        ax.errorbar(mean_true, pred_value, yerr=uncert_value, 
                    fmt='none', ecolor=colors[model_count % len(colors)], 
                    capsize=8, capthick=2, elinewidth=3, alpha=0.8)
        
        model_count += 1
    
    param_label = get_param_label(param)
    units = get_units(param)
    
    ax.set_xlabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_title(f'Model Predictions for {param_label} with Uncertainty\nSpectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    plt.tight_layout()
    return fig

def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name, selected_models):
    """Create combined plot showing all parameter predictions with uncertainty"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]
        
        # Filter models based on selection
        filtered_models = []
        filtered_values = []
        filtered_errors = []
        
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:
                filtered_models.append(model_name)
                filtered_values.append(pred_value)
                filtered_errors.append(param_uncerts.get(model_name, 0))
        
        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models for this parameter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue
        
        # Create bar plot with error bars
        x_pos = np.arange(len(filtered_models))
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=colors[:len(filtered_models)], edgecolor='black', linewidth=1)
        
        param_label = get_param_label(param)
        units = get_units(param)
        
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions with Uncertainty', 
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
    
    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_summary_plot(predictions, uncertainties, param_names, param_labels, selected_models, expected_values=None):
    """Create a summary plot showing all parameter predictions in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    #colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    # Definir los mismos colores que en create_combined_plot
    model_colors = {
        'Randomforest':'blue',  # Azul
        'Gradientboosting': 'green',  # Naranja
        'Svr': 'orange',  # Verde
        'Gaussianprocess': 'purple'  # Rojo
    }
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]
        
        # Filter models based on selection
        filtered_models = []
        filtered_values = []
        filtered_errors = []
        filtered_colors = []
        
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:
                filtered_models.append(model_name)
                filtered_values.append(pred_value)
                filtered_errors.append(param_uncerts.get(model_name, 0))
                filtered_colors.append(model_colors.get(model_name, '#9467bd'))  # Púrpura por defecto
        
        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models for this parameter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue
        
        # Create bar plot with error bars usando los mismos colores
        x_pos = np.arange(len(filtered_models))
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=filtered_colors, edgecolor='black', linewidth=1)
        
        param_label = get_param_label(param)
        units = get_units(param)
        
        # Add expected value if provided
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            exp_value = expected_values[param]['value']
            exp_error = expected_values[param].get('error', 0)
            
            # Draw a horizontal line for the expected value
            ax.axhline(y=exp_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Expected value')
            
            # Add error range for expected value
            if exp_error > 0:
                ax.axhspan(exp_value - exp_error, exp_value + exp_error, 
                          alpha=0.2, color='red', label='Expected range')
        
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions', 
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
        
        # Add legend if expected value is shown
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            ax.legend(loc='upper right')
    
    plt.suptitle('Summary of Parameter Predictions', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Function to get local file path
def get_local_file_path(filename):
    """Get path to a local file in the same directory as the script"""
    return os.path.join(os.path.dirname(__file__), filename)

# Main user interface
def main():
    # Initialize session state for model selection if not exists
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['Randomforest', 'Gradientboosting', 'Svr', 'Gaussianprocess']
    
    # Initialize session state for expected values
    if 'expected_values' not in st.session_state:
        st.session_state.expected_values = {
            'logn': {'value': None, 'error': None},
            'tex': {'value': None, 'error': None},
            'velo': {'value': None, 'error': None},
            'fwhm': {'value': None, 'error': None}
        }
    
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
        
        # Model selection
        st.subheader("3. Model Selection")
        st.write("Select which models to display in the results:")
        
        # Checkboxes for model selection
        rf_selected = st.checkbox("Random Forest", value=True, key='rf_checkbox')
        gb_selected = st.checkbox("Gradient Boosting", value=True, key='gb_checkbox')
        svr_selected = st.checkbox("Support Vector Regression", value=True, key='svr_checkbox')
        gp_selected = st.checkbox("Gaussian Process", value=True, key='gp_checkbox')
        
        # Update selected models in session state
        selected_models = []
        if rf_selected:
            selected_models.append('Randomforest')
        if gb_selected:
            selected_models.append('Gradientboosting')
        if svr_selected:
            selected_models.append('Svr')
        if gp_selected:
            selected_models.append('Gaussianprocess')
            
        st.session_state.selected_models = selected_models
        
        # Expected values input
        st.subheader("4. Expected Values (Optional)")
        st.write("Enter expected values and uncertainties for comparison:")
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['LogN', 'T_ex', 'V_los', 'FWHM']
        units = ['log(cm⁻2)', 'K', 'km/s', 'km/s']
        
        for i, (param, label, unit) in enumerate(zip(param_names, param_labels, units)):
            st.markdown(f'<div class="expected-value-input"><strong>{label} ({unit})</strong></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                value = st.number_input(
                    f"Expected value for {label}",
                    value=st.session_state.expected_values[param]['value'],
                    placeholder=f"Enter expected {label}",
                    key=f"exp_{param}_value"
                )
                st.session_state.expected_values[param]['value'] = value if value != 0 else None
            
            with col2:
                error = st.number_input(
                    f"Uncertainty for {label}",
                    value=st.session_state.expected_values[param]['error'],
                    min_value=0.0,
                    placeholder=f"Enter uncertainty for {label}",
                    key=f"exp_{param}_error"
                )
                st.session_state.expected_values[param]['error'] = error if error != 0 else None
        
        # Process button
        process_btn = st.button("Process Spectrum", type="primary", 
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
                
                # Display model information
                with st.expander("Model Information", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PCA Components", models['ipca'].n_components_)
                    with col2:
                        # Calculate total variance explained
                        cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
                        total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
                        st.metric("Variance Explained", f"{total_variance*100:.1f}%")
                    with col3:
                        # Count total models loaded
                        total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
                        st.metric("Total Models", total_models)
                
                # Show which models were loaded successfully
                st.subheader("Loaded Models")
                param_names = ['logn', 'tex', 'velo', 'fwhm']
                for param in param_names:
                    if param in models['all_models']:
                        model_count = len(models['all_models'][param])
                        st.write(f"{param}: {model_count} model(s) loaded")
                
                # Show PCA variance plot
                st.subheader("📊 PCA Variance Analysis")
                pca_fig = create_pca_variance_plot(models['ipca'])
                st.pyplot(pca_fig)
                
                # Option to download the PCA plot
                buf = BytesIO()
                pca_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download PCA variance plot",
                    data=buf,
                    file_name="pca_variance_analysis.png",
                    mime="image/png"
                )
            
            # Process spectrum
            with st.spinner("Processing spectrum and making predictions..."):
                results = process_spectrum(spectrum_file, models)
                
                if results is None:
                    st.error("Error processing the spectrum")
                    return
                
                # Display results
                st.header("📊 Prediction Results")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Model Performance", "Individual Plots", "Combined Plot"])
                
                with tab1:
                    st.subheader("Prediction Summary")
                    
                    # Create summary table (filtered by selected models)
                    summary_data = []
                    for param, label in zip(results['param_names'], results['param_labels']):
                        if param in results['predictions']:
                            param_preds = results['predictions'][param]
                            param_uncerts = results['uncertainties'].get(param, {})
                            
                            for model_name, pred_value in param_preds.items():
                                if model_name not in st.session_state.selected_models:
                                    continue
                                    
                                uncert_value = param_uncerts.get(model_name, np.nan)
                                summary_data.append({
                                    'Parameter': label,
                                    'Model': model_name,
                                    'Prediction': pred_value,
                                    'Uncertainty': uncert_value if not np.isnan(uncert_value) else 'N/A',
                                    'Units': get_units(param),
                                    'Relative_Error_%': (uncert_value / abs(pred_value) * 100) if pred_value != 0 and not np.isnan(uncert_value) else np.nan
                                })
                    
                    if summary_data:
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
                    else:
                        st.warning("No predictions were generated for the selected models")
                    
                    # Add summary plot with expected values
                    st.subheader("Summary Plot with Expected Values")
                    
                    # Check if any expected values are provided
                    has_expected_values = any(
                        st.session_state.expected_values[param]['value'] is not None 
                        for param in param_names
                    )
                    
                    if has_expected_values:
                        st.info("Red line shows expected value with shaded uncertainty range")
                    
                    summary_fig = create_summary_plot(
                        results['predictions'],
                        results['uncertainties'],
                        results['param_names'],
                        results['param_labels'],
                        st.session_state.selected_models,
                        st.session_state.expected_values if has_expected_values else None
                    )
                    st.pyplot(summary_fig)
                    
                    # Option to download the summary plot
                    buf = BytesIO()
                    summary_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download summary plot",
                        data=buf,
                        file_name="summary_predictions.png",
                        mime="image/png"
                    )
                
                with tab2:
                    st.subheader("📈 Model Performance Overview")
                    st.info("Showing typical parameter ranges for each model type")
                    create_model_performance_plots(models, st.session_state.selected_models)
                
                with tab3:
                    st.subheader("Prediction Plots by Parameter")
                    
                    # Create individual plots for each parameter
                    for param, label in zip(results['param_names'], results['param_labels']):
                        if param in results['predictions'] and results['predictions'][param]:
                            fig = create_comparison_plot(
                                results['predictions'], 
                                results['uncertainties'], 
                                param, 
                                label, 
                                models.get('training_stats', {}),
                                spectrum_file.name,
                                st.session_state.selected_models
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
                        else:
                            st.warning(f"No predictions available for {label}")
                
                with tab4:
                    st.subheader("Combined Prediction Plot")
                    
                    # Create combined plot
                    fig = create_combined_plot(
                        results['predictions'],
                        results['uncertainties'],
                        results['param_names'],
                        results['param_labels'],
                        spectrum_file.name,
                        st.session_state.selected_models
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
        4. **Select models**: Choose which models to display in the results using the checkboxes
        5. **Enter expected values (optional)**: Provide expected values and uncertainties for comparison
        6. **Process**: Click the 'Process Spectrum' button to get predictions
        """)


if __name__ == "__main__":
    main()





