# ivim_assets.py
import shutil
from dagster import asset, AssetIn, get_dagster_logger
from pathlib import Path
import polars as pl
import pandas as pd
import os

from ..resources import FileStorage, CollectionTables, OaiPipelineSubprocess

log = get_dagster_logger()

IVIM_COLLECTION_NAME = "ivim-mri_codecollection"

@asset
def ivim_study_samples(file_storage: FileStorage) -> pl.DataFrame:
    """
    Stages IVIM study samples from the downloads directory.
    Returns a DataFrame with study metadata.
    """
    downloads_path = file_storage.downloads_path / IVIM_COLLECTION_NAME
    os.makedirs(downloads_path, exist_ok=True)

    # Example: Hardcoded sample for simplicity; replace with actual sampling logic
    sample_data = {
        "study_id": ["brain_study_001"],
        "bvec_path": [str(downloads_path / "brain.bvec")],
        "bval_path": [str(downloads_path / "brain.bval")],
        "nifti_path": [str(downloads_path / "brain.nii.gz")],
    }
    df = pl.DataFrame(sample_data)
    log.info(f"Staged IVIM study samples: {df}")
    return df

@asset(ins={"ivim_samples": AssetIn("ivim_study_samples")})
def staged_ivim_study(ivim_samples: pl.DataFrame, file_storage: FileStorage, collection_tables: CollectionTables) -> None:
    """
    Stages IVIM study files and updates the collection tables.
    """
    for row in ivim_samples.iter_rows(named=True):
        study_id = row["study_id"]
        bvec_path = Path(row["bvec_path"])
        bval_path = Path(row["bval_path"])
        nifti_path = Path(row["nifti_path"])

        # Ensure files exist (for this example, assume they’re already downloaded)
        staged_path = file_storage.staged_path / IVIM_COLLECTION_NAME / study_id
        os.makedirs(staged_path, exist_ok=True)

        # Copy files to staged directory
        for src_path, fname in [(bvec_path, "brain.bvec"), (bval_path, "brain.bval"), (nifti_path, "brain.nii.gz")]:
            if src_path.exists():
                shutil.copy(src_path, staged_path / fname)

        # Update collection tables
        patient_df = pd.DataFrame({"patient_id": [f"patient_{study_id}"]})
        study_df = pd.DataFrame({
            "patient_id": [f"patient_{study_id}"],
            "study_instance_uid": [study_id],
            "study_date": ["2025-04-05"],  # Example date
            "study_description": ["IVIM Brain MRI"],
        })
        series_df = pd.DataFrame({
            "patient_id": [f"patient_{study_id}"],
            "study_instance_uid": [study_id],
            "series_instance_uid": [f"{study_id}_series"],
            "series_number": [1],
            "modality": ["MR"],
            "body_part_examined": ["Brain"],
            "series_description": ["IVIM Diffusion"],
        })

        collection_tables.insert_into_downloads(IVIM_COLLECTION_NAME, "patients", patient_df)
        collection_tables.insert_into_downloads(IVIM_COLLECTION_NAME, "studies", study_df)
        collection_tables.insert_into_downloads(IVIM_COLLECTION_NAME, "series", series_df)

        log.info(f"Staged IVIM study {study_id} at {staged_path}")

@asset(ins={"staged_study": AssetIn("staged_ivim_study")})
def ivim_analysis(staged_study, file_storage: FileStorage, pipeline: OaiPipelineSubprocess) -> dict:
    """
    Runs IVIM analysis on staged study files.
    """
    study_id = "brain_study_001"  # Hardcoded for simplicity; derive from staged_study if partitioned
    staged_path = file_storage.staged_path / IVIM_COLLECTION_NAME / study_id
    output_dir = file_storage.make_output_dir(
        downloads=IVIM_COLLECTION_NAME,
        dir_info={"repository_url": "example_repo", "development_status": "stable", "record_id": study_id},
        analysis_name="ivim",
        code_version="v1.0"
    )

    result = pipeline.run_pipeline(
        bvec_path=str(staged_path / "brain.bvec"),
        bval_path=str(staged_path / "brain.bval"),
        nifti_path=str(staged_path / "brain.nii.gz"),
        output_dir=str(output_dir),
        run_id=f"ivim_{study_id}"
    )
    log.info(f"IVIM analysis completed for {study_id}. Output in {output_dir}")
    return {"study_id": study_id, "output_dir": str(output_dir), "result": result}

@asset(ins={"analysis": AssetIn("ivim_analysis")})
def ivim_analysis_runs(analysis: dict, collection_tables: CollectionTables) -> None:
    """
    Stores metadata for IVIM analysis runs.
    """
    log.info(f"Storing IVIM analysis run metadata for {analysis['study_id']}")
    # Optionally, log more details to DuckDB or another store