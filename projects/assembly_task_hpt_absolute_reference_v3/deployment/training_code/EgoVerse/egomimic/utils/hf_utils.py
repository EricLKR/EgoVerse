"""HuggingFace Hub helpers.

Kept separate from the generic utils so the Hub dependency stays visible
and optional, mirroring the aws/ subpackage.
"""

import huggingface_hub


def download_from_huggingface(huggingface_repo_id: str):
    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder
