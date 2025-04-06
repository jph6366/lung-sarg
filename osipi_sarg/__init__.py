import os
from pathlib import Path

from dagster import EnvVar, Definitions, load_assets_from_modules, load_asset_checks_from_modules, define_asset_job
# from dagster_dbt import DbtCliResource, load_assets_from_dbt_project
from dagster_duckdb_polars import DuckDBPolarsIOManager
from dagster_duckdb import DuckDBResource
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from osipi_sarg.assets.ivim import ivim_analysis, ivim_analysis_runs, ivim_study_samples, staged_ivim_study

from .assets import ivim, huggingface, idc, injested_study
from .resources import (
    DATA_DIR,
    # DBT_PROJECT_DIR,
    DATABASE_PATH,
    CollectionPublisher,
    CollectionTables,
    FileStorage,
    OaiPipelineSubprocess
)
from .sensors import staged_study_sensor, injest_and_analyze_study_job

# dbt = DbtCliResource(project_dir=DBT_PROJECT_DIR, profiles_dir=DBT_PROJECT_DIR)
duckdb_resource = DuckDBResource(database=DATABASE_PATH)
file_storage = FileStorage(
    root_dir=os.getenv("FILE_STORAGE_ROOT", str(DATA_DIR)),
    staged_dir=str(DATA_DIR / "staged"),
    ingested_dir=str(DATA_DIR / "injested"),
    downloads_dir=str(DATA_DIR / "downloads"),
)

# dbt_assets = load_assets_from_dbt_project(DBT_PROJECT_DIR, DBT_PROJECT_DIR)
dbt_assets = []
all_assets = load_assets_from_modules([ivim, huggingface, injested_study])
all_checks = load_asset_checks_from_modules([ivim, huggingface, injested_study])

# Use os.getenv to get the environment variable with a default value
root_dir = os.getenv("FILE_STORAGE_ROOT", str(DATA_DIR))


# pick how to run the OAI pipeline
pipeline_src_dir = EnvVar("PIPELINE_SRC_DIR")
env_setup_command = EnvVar("ENV_SETUP_COMMAND")

oai_pipeline_resource_env = os.getenv("OAI_PIPELINE_RESOURCE", "subprocess")
oai_pipeline_resource = OaiPipelineSubprocess(
    pipeline_src_dir=pipeline_src_dir,
    env_setup_command=env_setup_command,
)

file_storage = FileStorage(root_dir=root_dir)

# Job to stage IVIM samples
stage_ivim_samples_job = define_asset_job(
    "stage_ivim_samples",
    [ivim_study_samples],
    description="Stages IVIM study samples",
)

# Job to ingest and analyze an IVIM study
ingest_and_analyze_ivim_job = define_asset_job(
    "ingest_and_analyze_ivim",
    [ivim_study_samples, staged_ivim_study, ivim_analysis, ivim_analysis_runs],
    description="Ingests and analyzes an IVIM study",
    tags={"job": "cpu"},
)

# Job to run IVIM analysis on pre-staged studies
ivim_analysis_job = define_asset_job(
    "ivim_analysis_job",
    [ivim_analysis, ivim_analysis_runs],
    description="Runs IVIM analysis on pre-staged studies",
    tags={"job": "cpu"},
)

resources = {
    # "dbt": dbt,
    "io_manager": DuckDBPolarsIOManager(database=DATABASE_PATH, schema="main"),
    "collection_publisher": CollectionPublisher(
        hf_token=EnvVar("HUGGINGFACE_TOKEN"), file_storage=file_storage
    ),
    "duckdb": duckdb_resource,
    "collection_tables": CollectionTables(
        duckdb=duckdb_resource, file_storage=file_storage
    ),
    "oai_pipeline": oai_pipeline_resource,
    "file_storage": file_storage,
    "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
}


defs = Definitions(
    assets=[
        *dbt_assets,
        *all_assets,
        ivim_study_samples,
        staged_ivim_study,
        ivim_analysis,
        ivim_analysis_runs,
    ],    
    asset_checks=all_checks,
    resources=resources,
    jobs=[
        stage_ivim_samples_job,
        ingest_and_analyze_ivim_job,
        ivim_analysis_job,
    ],
    sensors=[
        staged_study_sensor,
    ],
)