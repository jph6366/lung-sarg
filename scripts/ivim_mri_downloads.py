import shutil
from pathlib import Path

from osipi_sarg.resources import DATA_DIR, DownloadInfo


class FilePaths:
    def __init__(
        self,
        root_dir: str = str(DATA_DIR),
        staged_dir: str = "",
        ingested_dir: str = "",
        downloads_dir: str = "",
    ):
        root = Path(root_dir)
        self._staged_path = Path(staged_dir) if staged_dir else root / "staged"
        self._ingested_path = Path(ingested_dir) if ingested_dir else root / "ingested"
        self._downloads_path = (
            Path(downloads_dir) if downloads_dir else root / "collections"
        )

    @property
    def staged_path(self) -> Path:
        return self._staged_path

    @property
    def ingested_path(self) -> Path:
        return self._ingested_path

    @property
    def downloads_path(self) -> Path:
        return self._downloads_path

    def get_downloads_path(self, downloads: str) -> Path:
        return self.downloads_path / downloads

    def ensure_downloads_dir(self, downloads: str):
        downloads_dir = self.get_downloads_path(downloads)
        if not downloads_dir.exists():
            downloads_dir.mkdir(parents=True)
        return downloads_dir

    def get_study_downloads_dir(
        self,
        downloads: str,
        download_info: DownloadInfo,
    ) -> Path:
        patient, study_description, study_uid = (
            download_info["repository_url"],
            download_info["development_status"],
            download_info["record_id"],
        )
        return (
            self.get_downloads_path(downloads)
            / patient
            / f"{study_description}-{study_uid}"
        )

    def get_output_dir(
        self,
        collection: str,
        study_info: DownloadInfo,
        analysis_name: str,
        code_version: str = "None",
    ) -> Path:
        study_dir = self.get_study_downloads_dir(collection, study_info)
        output_dir = study_dir / analysis_name / code_version
        return output_dir

    def make_output_dir(
        self,
        collection: str,
        dir_info: DownloadInfo,
        analysis_name: str,
        code_version: str = "None",
    ) -> Path:
        output_dir = self.get_output_dir(
            collection, dir_info, analysis_name, code_version
        )
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        else:
            # clean out the directory
            for item in output_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        return output_dir
