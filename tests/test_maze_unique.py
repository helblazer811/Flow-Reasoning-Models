import numpy as np
from frm.config import DataConfig, load_config
from frm.datasets.loading import load_dataset

def test_maze_unique_cache_and_dispatch(tmp_path):
    directory = tmp_path / "maze_unique"
    directory.mkdir()
    solutions = np.full((3, 900), 2, dtype=np.uint8)
    solutions[:, 0] = 4
    solutions[:, -1] = 5
    masks = (solutions != 2).astype(np.uint8)
    np.savez(directory / "train.npz", solutions=solutions, clue_mask=masks)
    ds = load_dataset(DataConfig(dataset="maze_unique", root=str(tmp_path)),
                      split="train", max_samples=2, seed=3)
    assert len(ds) == 2 and ds.seq_len == 900 and ds.vocab_size == 6
    assert ds[0].shape == (2, 900)
    assert np.array_equal(ds[0][1].numpy(), masks[0])

def test_maze_unique_recipe():
    c = load_config("configs/maze_unique.yaml")
    assert c["data"].dataset == "maze_unique"
    assert c["pipeline"].stage_a_steps == 300000
    assert c["pipeline"].stage_b_steps == 100000
    assert c["loss"].sfr_noise_scale == 0
    assert c["loss"].fixed_level is None
    assert c["evaluation"].held_level == 0.3
