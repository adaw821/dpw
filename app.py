"""
Streamlit Cloud entrypoint.

Workflow:
1. On first run (cold start) the app downloads the dataset from Google Drive
   into ./data/processed/ and (optionally) the cached TF-IDF pickles into
   ./recommendation_outputs/ml_recommender/.
2. After the data is in place, control is handed over to app_home.py, which
   is your original homepage / router.

Configuration:
- Set the Google Drive file IDs / folder ID in `GDRIVE_CONFIG` below, OR put
  them in Streamlit secrets (Settings -> Secrets) like:

      [gdrive]
      data_folder_id = "https://drive.google.com/drive/folders/1eHhsAU0lzGlctl6SNHwEFwm-wxpzTy8r?usp=sharing"
      tfidf_folder_id = ""   # optional

  Secrets override the hard-coded values.
"""

from pathlib import Path
import os
import sys
import zipfile
import shutil

import streamlit as st


# --------------------------------------------------------------------------- #
# 1. Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TFIDF_DIR = PROJECT_ROOT / "recommendation_outputs" / "ml_recommender"

REQUIRED_DATA_FILES = [
    "movies_main.parquet",
    "movie_genres.parquet",
    "movie_countries.parquet",
    "movie_companies.parquet",
    "cast.csv",
    "crew.csv",
    "keywords_long.csv",
    "ratings_clean.parquet",
    "final_ratings_raw.parquet",
    "recommendation_analysis_table.parquet",
]

OPTIONAL_TFIDF_FILES = [
    "tfidf_vectors.pkl",
    "tfidf_norms.pkl",
]


# --------------------------------------------------------------------------- #
# 2. Google Drive config (edit these or use st.secrets)
# --------------------------------------------------------------------------- #
GDRIVE_CONFIG = {
    # A Google Drive *folder* that contains all files in REQUIRED_DATA_FILES.
    # Share the folder as "Anyone with the link - Viewer".
    # Folder ID is the part after /folders/ in the URL.
    "data_folder_id": "PUT_YOUR_DATA_FOLDER_ID_HERE",

    # Optional: folder containing tfidf_vectors.pkl + tfidf_norms.pkl.
    # Leave as "" to skip - the recommender will rebuild them at runtime.
    "tfidf_folder_id": "",

    # Alternative: a single .zip file ID containing data/processed/* .
    # If set, this is preferred over data_folder_id.
    "data_zip_id": "",
}


def _get_gdrive_setting(key: str) -> str:
    """Prefer st.secrets, fall back to GDRIVE_CONFIG."""
    try:
        return st.secrets["gdrive"][key]
    except Exception:
        return GDRIVE_CONFIG.get(key, "")


# --------------------------------------------------------------------------- #
# 3. Download helpers
# --------------------------------------------------------------------------- #
def _all_present(folder: Path, files) -> bool:
    return folder.exists() and all((folder / f).exists() for f in files)


def _download_folder(folder_id: str, dest: Path) -> None:
    """Download a whole Google Drive folder via gdown."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url=url, output=str(dest), quiet=False, use_cookies=False)


def _download_zip(file_id: str, dest_dir: Path) -> None:
    """Download a single zip file from Google Drive and extract it."""
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "_download.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)

    # If the zip extracted into a nested "processed/" subfolder, flatten it.
    nested = dest_dir / "processed"
    if nested.exists() and nested.is_dir():
        for item in nested.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        nested.rmdir()


@st.cache_resource(show_spinner=False)
def ensure_assets() -> None:
    """Download data + (optional) tfidf cache once per container."""
    # ---- main dataset ----
    if not _all_present(DATA_DIR, REQUIRED_DATA_FILES):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_id = _get_gdrive_setting("data_zip_id")
        folder_id = _get_gdrive_setting("data_folder_id")

        with st.spinner("First-time setup: downloading dataset from Google Drive..."):
            if zip_id and zip_id != "PUT_YOUR_DATA_FOLDER_ID_HERE":
                _download_zip(zip_id, DATA_DIR)
            elif folder_id and folder_id != "PUT_YOUR_DATA_FOLDER_ID_HERE":
                _download_folder(folder_id, DATA_DIR)
            else:
                st.error(
                    "Google Drive ID not configured. "
                    "Set GDRIVE_CONFIG in app.py or [gdrive] in st.secrets."
                )
                st.stop()

        missing = [f for f in REQUIRED_DATA_FILES if not (DATA_DIR / f).exists()]
        if missing:
            st.error(f"Download finished but these files are missing: {missing}")
            st.stop()

    # ---- optional tfidf cache ----
    tfidf_id = _get_gdrive_setting("tfidf_folder_id")
    if tfidf_id and not _all_present(TFIDF_DIR, OPTIONAL_TFIDF_FILES):
        TFIDF_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading TF-IDF cache..."):
            try:
                _download_folder(tfidf_id, TFIDF_DIR)
            except Exception as e:
                st.warning(f"TF-IDF cache download failed ({e}); will rebuild at runtime.")


# --------------------------------------------------------------------------- #
# 4. Run
# --------------------------------------------------------------------------- #
ensure_assets()

# Make sure relative imports inside app_home.py / success_model.py / recommend_model.py
# resolve correctly when this file is the entrypoint.
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hand over to your original homepage router.
with open(PROJECT_ROOT / "app_home.py", "r", encoding="utf-8") as f:
    exec(f.read(), {"__name__": "__main__", "__file__": str(PROJECT_ROOT / "app_home.py")})
