"""Centralized project paths."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

CONFIGS_DIR = REPO_ROOT / "transpiler_configs"
DATA_DIR = REPO_ROOT / "data"
HARDWARE_RUNS_DIR = DATA_DIR / "hardware_runs"
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = REPO_ROOT / "examples"
NOTEBOOKS_DIR = REPO_ROOT / "examples"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TESTS_DIR = REPO_ROOT / "tests"
# Google's released Tesseract search settings (Rosenfeld et al., Fig. 4 cultivation N = 3), shared
# by the Tesseract decoding of their grafting circuits and of ours.
GOOGLE_TESSERACT_PROPERTIES_PATH = (
    REPO_ROOT / "google" / "zenodo" / "fig4" / "cultivation_N=3" / "decoder_properties.json"
)
