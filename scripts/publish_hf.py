"""Publish NAFNet MLX dist dirs to mlx-community + a collection."""
from pathlib import Path
from huggingface_hub import create_repo, upload_folder, create_collection, add_collection_item
ROOT = Path(__file__).resolve().parents[1]
REPOS = ["NAFNet-SIDD-width64", "NAFNet-GoPro-width64", "NAFNet-REDS-width64"]
for stem in REPOS:
    rid = f"mlx-community/{stem}"
    create_repo(rid, repo_type="model", exist_ok=True, private=False)
    upload_folder(repo_id=rid, folder_path=str(ROOT/"dist"/stem), repo_type="model",
                  commit_message="Add NAFNet MLX weights (faithful port of megvii-research/NAFNet)")
    print("published", rid, flush=True)
col = create_collection(title="NAFNet MLX", namespace="mlx-community",
    description="MLX port of NAFNet (Simple Baselines for Image Restoration): on-device deblur/denoise on Apple Silicon.",
    exists_ok=True)
for stem in REPOS:
    add_collection_item(col.slug, item_id=f"mlx-community/{stem}", item_type="model", exists_ok=True)
print("collection:", "https://huggingface.co/collections/"+col.slug, flush=True)
print("ALL_PUBLISHED", flush=True)
