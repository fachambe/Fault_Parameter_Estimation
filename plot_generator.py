import os
import numpy as np
import matplotlib.pyplot as plt


def plot_rmse_vs_crlb_snr_sweep(snr_dbs, rmse_results, crlb_results, selected_keys, is_bayesian=False):
    """
    Plot RMSE vs sqrt(CRLB) across SNR for each parameter.

    Args:
        snr_dbs: Array of SNR values in dB
        rmse_results: Dict {param_name: [rmse_values_across_snr]}
        crlb_results: Dict {param_name: [sqrt_crlb_values_across_snr]}
        selected_keys: List of parameter names
        is_bayesian: If True, use Bayesian labels
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Set labels based on Bayesian vs Frequentist
    if is_bayesian:
        rmse_label = 'Bayesian RMSE'
        crlb_label = 'sqrt(BCRLB)'
        title_prefix = 'Bayesian RMSE vs sqrt(BCRLB)'
    else:
        rmse_label = 'RMSE'
        crlb_label = 'sqrt(CRLB)'
        title_prefix = 'RMSE vs sqrt(CRLB)'

    for idx, key in enumerate(selected_keys):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Plot data
        ax.plot(snr_dbs, rmse_results[key], 'bo-', label=rmse_label, markersize=6)
        ax.plot(snr_dbs, crlb_results[key], 'r--s', label=crlb_label, linewidth=2, markersize=4)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error')
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(selected_keys), len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(f'{title_prefix} - SNR Sweep', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


def plot_CI_and_pred_TF(tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz):
    """
    Plot transfer function with confidence interval.

    Args:
        tf_mean: Mean of posterior TF samples (numpy array)
        tf_lower: Lower 2.5% percentile (numpy array)
        tf_upper: Upper 97.5% percentile (numpy array)
        H_clean_db: True TF in dB (numpy array or tensor)
        freq_range_mhz: Frequency range in MHz (numpy array or tensor)
    """
    # Convert tensors to numpy if needed
    if hasattr(freq_range_mhz, 'numpy'):
        freq_range_mhz = freq_range_mhz.numpy()
    if hasattr(H_clean_db, 'numpy'):
        H_clean_db = H_clean_db.detach().numpy() if hasattr(H_clean_db, 'detach') else H_clean_db.numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(freq_range_mhz, tf_mean, 'k-', linewidth=1.5, label='Model')
    plt.plot(freq_range_mhz, H_clean_db, 'r--', linewidth=1.5, label='Truth')
    plt.fill_between(freq_range_mhz, tf_lower, tf_upper,
                     alpha=0.3, color='steelblue', label='95% CI')
    plt.xscale('log')
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel(r'$H_{1,1}$ (dB)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    #For rmse_vs_crlb_snr_sweep plot
    rmse_crlb_npz_file = "frequentist_p3_M1_seed95/rmse_vs_crlb_snr_sweep_M1_p3_seed95.npz"  # Update this path

    if os.path.exists(rmse_crlb_npz_file):
        # Load data
        data = np.load(rmse_crlb_npz_file)
        print("data", data)
        snr_dbs = data['snr_dbs']

        # Extract selected_keys from npz keys (keys starting with 'rmse_')
        selected_keys = [k.replace('rmse_', '') for k in data.files if k.startswith('rmse_')]

        # Reconstruct rmse_results and crlb_results dicts
        rmse_results = {}
        crlb_results = {}
        for key in selected_keys:
            rmse_results[key] = data[f'rmse_{key}'].tolist()
            crlb_results[key] = data[f'crlb_{key}'].tolist()

        print(f"Loaded data for {len(selected_keys)} parameters: {selected_keys}")
        print(f"SNR range: {snr_dbs}")

        # Replot
        plot_rmse_vs_crlb_snr_sweep(snr_dbs, rmse_results, crlb_results, selected_keys)
    
    # For plot_CI_and_pred_TF 
    tf_npz_file = "frequentist_p3_M1_seed95/40dBSNR_149khz-10mhz_Adam_lr0.02_tf_posterior_CI_with_fault_p3_seed95.npz"  # Update this path

    if os.path.exists(tf_npz_file):
        # Load data
        data = np.load(tf_npz_file)
        tf_mean = data['tf_mean']
        tf_lower = data['tf_lower']
        tf_upper = data['tf_upper']
        H_clean_db = data['H_clean_db']
        freq_range_mhz = data['freq_range_mhz']

        print(f"Loaded TF data: {len(freq_range_mhz)} frequency points")
        # Replot
        plot_CI_and_pred_TF(tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz)
