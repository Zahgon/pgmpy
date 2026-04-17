from pathlib import Path

from huggingface_hub import constants, hf_hub_download

HF_ETAG_TIMEOUT = max(constants.HF_HUB_ETAG_TIMEOUT, 30)
constants.HF_HUB_DOWNLOAD_TIMEOUT = max(constants.HF_HUB_DOWNLOAD_TIMEOUT, 30)


def read_hf_file(
    *,
    repo_id: str,
    filename: str,
    repo_type: str | None = None,
    revision: str = "main",
) -> bytes:
    """
    Downloads a file from the Hugging Face Hub and returns its cached contents.
    """
    pass
