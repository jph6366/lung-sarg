# Usage
# python pipeline_cli.py --bvec ./downloads/Data/brain.bvec --bval ./downloads/Data/brain.bval ./downloads/Data/brain.nii.gz ./ivim_results --voxel 60 60 30

import argparse
from pipeline import analysis_pipeline


def main():
    parser = argparse.ArgumentParser(description='IVIM Analysis CLI for Brain MRI')
    parser.add_argument('data_path', type=str, help='Path to the input NIfTI file (.nii.gz)')
    parser.add_argument('output_dir', type=str, help='Directory to save output files')
    parser.add_argument('--bvec', type=str, required=True, help='Path to the .bvec file')
    parser.add_argument('--bval', type=str, required=True, help='Path to the .bval file')
    parser.add_argument(
        '--voxel',
        type=int,
        nargs=3,
        default=[60, 60, 30],
        help='Voxel coordinates (x y z) to analyze (default: 60 60 30)',
    )
    parser.add_argument(
        '--no_output',
        action='store_true',
        help='Do not write output files to the output directory (print to console instead)',
    )

    args = parser.parse_args()

    # Convert voxel list to tuple
    voxel = tuple(args.voxel)

    # Run the analysis pipeline
    analysis_pipeline(
        bvec_path=args.bvec,
        bval_path=args.bval,
        data_path=args.data_path,
        voxel=voxel,
        output_path=args.output_dir if not args.no_output else None
    )


if __name__ == '__main__':
    main()