import os
import shutil
import tempfile
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict
from pathlib import Path

import yaml
import pandas as pd
from dagster import (
    ResourceDependency,
    InitResourceContext,
    ConfigurableResource,
    get_dagster_logger,
)
from pydantic import PrivateAttr
from dagster_duckdb import DuckDBResource
from huggingface_hub import HfApi

from scripts.ivim_mri_downloads import FilePaths

log = get_dagster_logger()

DBT_PROJECT_DIR = str(Path(__file__).parent.resolve() / ".." / "dbt")
DATA_DIR = Path(__file__).parent.resolve() / ".." / "data"
PRE_STAGED_DIR = DATA_DIR / "pre-staged"
STAGED_DIR = DATA_DIR / "staged"
INJESTED_DIR = DATA_DIR / "injested"
# COLLECTIONS_DIR = DATA_DIR / "collections"
DOWNLOADS_DIR = DATA_DIR / "downloads"
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "database.duckdb"))

IVIM_DOWNLOADS_NAME = "ivim-mri_codecollection"


class DownloadInfo(TypedDict):
    repository_url: str
    development_status: str
    record_id: str


class FileStorage(ConfigurableResource):
    root_dir: str = str(DATA_DIR)
    staged_dir: str = ""
    ingested_dir: str = ""
    downloads_dir: str = ""

    def setup_for_execution(self, _: InitResourceContext) -> None:
        self._file_paths = FilePaths(
            root_dir=self.root_dir,
            staged_dir=self.staged_dir,
            ingested_dir=self.ingested_dir,
            downloads_dir=self.downloads_dir,
        )

    @property
    def staged_path(self) -> Path:
        return self._file_paths.staged_path

    @property
    def ingested_path(self) -> Path:
        return self._file_paths.ingested_path

    @property
    def downloads_path(self) -> Path:
        return self._file_paths.downloads_path

    def ensure_downloads_dir(self, downloads: str):
        return self._file_paths.ensure_downloads_dir(downloads)

    def get_study_downloads_dir(
        self,
        downloads: str,
        download_info: DownloadInfo,
        analysis_name: str,
        code_version: str = "undefined",
    ) -> Path:
        return self._file_paths.get_study_downloads_dir(downloads, download_info)

    def get_output_dir(
        self,
        downloads: str,
        download_info: DownloadInfo,
        analysis_name: str,
        code_version: str = "undefined",
    ) -> Path:
        return self._file_paths.get_output_dir(
            downloads, download_info, analysis_name, code_version
        )

    def make_output_dir(
        self,
        downloads: str,
        dir_info: DownloadInfo,
        analysis_name: str,
        code_version: str = "None",
    ) -> Path:
        return self._file_paths.make_output_dir(
            downloads, dir_info, analysis_name, code_version
        )


downloads_table_names = {"patients", "studies", "series"}


class CollectionTables(ConfigurableResource):
    duckdb: ResourceDependency[DuckDBResource]
    file_storage: ResourceDependency[FileStorage]
    downloads_names: List[str] = [IVIM_DOWNLOADS_NAME]

    def setup_for_execution(self, context: InitResourceContext) -> None:
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        self._db = self.duckdb

        with self._db.get_connection() as conn:
            for downloads_name in self.downloads_names:
                downloads_path = DOWNLOADS_DIR / downloads_name
                os.makedirs(downloads_path, exist_ok=True)

                for table in downloads_table_names:
                    table_parquet = downloads_path / f"{table}.parquet"
                    table_name = f"{downloads_name}_{table}"
                    if table_parquet.exists():
                        conn.execute(
                            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM parquet_scan('{table_parquet}')"
                        )
                    else:
                        if table == "patients":
                            conn.execute(
                                f"CREATE TABLE IF NOT EXISTS {table_name} (patient_id VARCHAR, );"
                            )
                        elif table == "studies":
                            conn.execute(
                                f"CREATE TABLE IF NOT EXISTS {table_name} (patient_id VARCHAR, study_instance_uid VARCHAR, study_date DATE, study_description VARCHAR);"
                            )
                        elif table == "series":
                            conn.execute(
                                f"CREATE TABLE IF NOT EXISTS {table_name} (patient_id VARCHAR, study_instance_uid VARCHAR, series_instance_uid VARCHAR, series_number BIGINT, modality VARCHAR, body_part_examined VARCHAR, series_description VARCHAR);"
                            )

    def teardown_after_execution(self, context: InitResourceContext) -> None:
        with self._db.get_connection() as conn:
            conn.execute("VACUUM")

    def write_downloads_parquets(self):
        with self._db.get_connection() as conn:
            for downloads_name in self.downloads_names:
                downloads_path = DOWNLOADS_DIR / downloads_name
                for table in downloads_table_names:
                    table_name = f"{downloads_name}_{table}"
                    table_parquet = downloads_path / f"{table}.parquet"
                    conn.execute(
                        f"COPY {table_name} TO '{table_parquet}' (FORMAT 'parquet')"
                    )

    def insert_into_downloads(
        self, downloads_name: str, table_name: str, df: pd.DataFrame
    ):
        if df.empty:
            return
        if downloads_name not in self.downloads_names:
            raise ValueError(f"Collection {downloads_name} not found")
        if table_name not in downloads_table_names:
            raise ValueError(f"Table {table_name} not found")

        with self._db.get_connection() as conn:
            conn.execute(f"INSERT INTO {downloads_name}_{table_name} SELECT * FROM df")


class OaiPipeline(ConfigurableResource, ABC):
    @abstractmethod
    def run_pipeline(
        self,
        image_path: str,
        output_dir: str,
        laterality: str,
        run_id: str,
        override_src_dir: Optional[str] = None,
    ):
        pass


class OaiPipelineSubprocess:  # Renamed to reflect it’s no longer OAI-specific
    pipeline_src_dir: str
    env_setup_command: str = ""

    def run_pipeline(
        self,
        bvec_path: str,
        bval_path: str,
        nifti_path: str,
        output_dir: str,
        run_id: str,
        override_src_dir: Optional[str] = None,
    ):
        src_dir = override_src_dir or self.pipeline_src_dir
        optional_env_setup = (
            f"{self.env_setup_command} && " if self.env_setup_command else ""
        )

        # Updated run_call for IVIM pipeline_cli.py
        run_call = (
            f"python ./ivim/pipeline_cli.py "
            f'--bvec "{bvec_path}" '
            f'--bval "{bval_path}" '
            f'"{nifti_path}" '
            f'"{output_dir}"'
        )

        command = f"{optional_env_setup}{run_call}"
        log.info(f"Running pipeline: {run_call}")
        log.info(f"With env setup: {command}")

        result = subprocess.run(
            command, cwd=src_dir, shell=True, capture_output=True, text=True
        )
        log.info(result.stdout)
        if result.stderr:
            log.error(result.stderr)


class CollectionPublisher(ConfigurableResource):
    hf_token: str
    tmp_dir: str = tempfile.gettempdir()

    _api: HfApi = PrivateAttr()

    def setup_for_execution(self, context: InitResourceContext) -> None:
        self._api = HfApi(token=self.hf_token)

    def publish(
        self,
        collection_name: str,
        readme: Optional[str] = None,
        generate_datapackage: bool = False,
    ):
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as temp_dir:
            collection_path = DOWNLOADS_DIR / collection_name
            log.info(
                f"Copying collection {collection_name} parquet files to {temp_dir}"
            )
            shutil.copyfile(
                collection_path / "patients.parquet",
                os.path.join(temp_dir, "patients.parquet"),
            )
            shutil.copyfile(
                collection_path / "studies.parquet",
                os.path.join(temp_dir, "studies.parquet"),
            )
            shutil.copyfile(
                collection_path / "series.parquet",
                os.path.join(temp_dir, "series.parquet"),
            )

            if readme:
                readme_path = os.path.join(temp_dir, "README.md")
                with open(readme_path, "w") as readme_file:
                    readme_file.write(readme)

            if generate_datapackage:
                datapackage = {
                    "name": collection_name,
                    "resources": [
                        {"path": "patients.parquet", "format": "parquet"},
                        {"path": "studies.parquet", "format": "parquet"},
                        {"path": "series.parquet", "format": "parquet"},
                    ],
                }
                datapackage_path = os.path.join(temp_dir, "datapackage.yaml")
                with open(datapackage_path, "w") as dp_file:
                    yaml.dump(datapackage, dp_file)

            # log.info(f"Uploading collection {collection_name} to Hugging Face")
            # # Note: the repository has to be already created
            # self._api.upload_folder(
            #     folder_path=temp_dir,
            #     repo_id=f"radiogenomics/lung_sarg_{collection_name}",
            #     repo_type="dataset",
            #     commit_message=f"Update {collection_name} collection",
            #     multi_commits=True,
            #     multi_commits_verbose=True,
            # )
