import numpy as np
import itk
import os
from wrappers.OsipiBase import OsipiBase
from standardized.IAR_LU_biexp import IAR_LU_biexp


def preprocess(image, bval, voxel=(60, 60, 30), direction=6, window_min_percentile=0.1, window_max_percentile=99.9, output_min=0.0, output_max=1.0):
    """
    Preprocess IVIM data for a specific voxel: extract signal, normalize, and prepare for fitting.

    :param image: NumPy array of the loaded NIfTI data (e.g., from itk.array_from_image).
    :param bval: NumPy array of b-values.
    :param voxel: Tuple (x, y, z) specifying the voxel to analyze.
    :param direction: Integer specifying the diffusion direction to extract (e.g., 6).
    :param window_min_percentile: Lower percentile for intensity clipping (not used here but included for extensibility).
    :param window_max_percentile: Upper percentile for intensity clipping (not used here).
    :param output_min: Minimum output value for normalization (default 0.0).
    :param output_max: Maximum output value for normalization (default 1.0).
    :return: Tuple (signal_1dir, unique_bval) - preprocessed signal for one direction and unique b-values.
    """
    # Extract voxel data
    x, y, z = voxel
    data_vox = np.squeeze(image[x, y, z, :])

    # Normalize data based on b=0 signal
    selsb = np.array(bval) == 0
    S0 = np.nanmean(data_vox[selsb], axis=0).astype('<f')
    if S0 == 0 or np.isnan(S0):  # Avoid division by zero or NaN
        raise ValueError(f"Invalid S0 value ({S0}) at voxel {voxel}. Cannot normalize.")
    data_vox = data_vox / S0

    # Extract signal for the specified direction (e.g., every 6th value starting at 'direction')
    signal_1dir = data_vox[direction:None:6]
    signal_1dir = np.insert(signal_1dir, 0, 1)  # Add normalized S0 (1.0) at the start

    # Get unique b-values corresponding to the signal
    unique_bval = np.unique(bval)[::6]  # Assumes b-values repeat every 6 directions
    unique_bval = np.insert(unique_bval, 0, 0)  # Add b=0 at the start

    return signal_1dir, unique_bval


def run_osipi_fit(signal_1dir, bval):
    """
    Run the IVIM fit using OsipiBase and the IAR_LU_biexp algorithm.

    :param signal_1dir: Preprocessed signal for one direction.
    :param bval: Corresponding b-values for the signal.
    :return: Dictionary with fit parameters ('f', 'Dp', 'D').
    """
    algorithm = IAR_LU_biexp()
    fit = OsipiBase.osipi_fit(algorithm, signal_1dir, bval)  # Returns dict with 'f', 'Dp', 'D'
    return fit


def analysis_pipeline(bvec_path, bval_path, data_path, voxel=(60, 60, 30), output_path=None):
    """
    Computes IVIM parameters for a specified voxel from brain MRI data.

    :param bvec_path: Path to the .bvec file.
    :param bval_path: Path to the .bval file.
    :param data_path: Path to the input NIfTI file (.nii.gz).
    :param voxel: Tuple (x, y, z) specifying the voxel to analyze (default: (60, 60, 30)).
    :param output_path: Path to the desired directory for outputs (default: None, prints to console).
    """
    # Load bvec and bval with NumPy
    bvec = np.genfromtxt(bvec_path)
    bval = np.genfromtxt(bval_path)

    # Load NIfTI with ITK
    image_itk = itk.imread(data_path)
    datas = itk.array_from_image(image_itk)  # Convert ITK image to NumPy array

    # Preprocess the data for the specified voxel
    signal_1dir, unique_bval = preprocess(datas, bval, voxel=voxel, direction=6)

    # Run the IVIM fit
    fit = run_osipi_fit(signal_1dir, unique_bval)

    # Prepare output
    results = {
        'voxel': voxel,
        'f': fit['f'],
        'D*': fit['Dp'],  # Assuming 'Dp' is D* (pseudo-diffusion)
        'D': fit['D']
    }

    # Output results
    if output_path is None:
        # Print to console if no output path is specified
        print(f"IVIM Fit Results for Voxel {voxel}:")
        print(f"f: {results['f']:.4f}")
        print(f"D*: {results['D*']:.4e} mm²/s")
        print(f"D: {results['D']:.4e} mm²/s")
    else:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save fit parameters to a text file
        output_file = os.path.join(output_path, f"ivim_fit_voxel_{voxel[0]}_{voxel[1]}_{voxel[2]}.txt")
        with open(output_file, 'w') as f:
            f.write(f"IVIM Fit Results for Voxel {voxel}:\n")
            f.write(f"f: {results['f']:.4f}\n")
            f.write(f"D*: {results['D*']:.4e} mm²/s\n")
            f.write(f"D: {results['D']:.4e} mm²/s\n")
        print(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    # Example usage
    bvec_path = './downloads/Data/brain.bvec'
    bval_path = './downloads/Data/brain.bval'
    data_path = './downloads/Data/brain.nii.gz'
    voxel = (60, 60, 30)
    output_path = './downloads/Data/ivim_results'

    results = analysis_pipeline(bvec_path, bval_path, data_path, voxel, output_path)