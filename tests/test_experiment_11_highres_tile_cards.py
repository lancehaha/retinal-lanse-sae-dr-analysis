import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment11_cards", ROOT / "scripts/run_highres_tile_knowledge_cards.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HighResolutionTileCardsTest(unittest.TestCase):
    def test_same_image_spatial_control_and_probability(self) -> None:
        labels = np.zeros((4, 5), dtype=bool)
        labels[0, 0] = True
        centres = np.asarray([[.20, .20], [.22, .20], [.80, .80], [.80, .70]], dtype=np.float32)
        images, tiles, target = MODULE.matched_rows(labels, centres, np.asarray([0, 4]), 0, 4106)
        self.assertEqual(images.tolist(), [0, 0])
        self.assertEqual(tiles.tolist(), [0, 1])
        self.assertEqual(target.tolist(), [1, 0])
        dimension = MODULE.INPUT_DIM + MODULE.FEATURE_COUNT
        self.assertEqual(MODULE.probability(np.zeros(dimension), np.zeros((1, dimension)), np.ones((1, dimension)), np.zeros(dimension), np.zeros(1)), .5)

    def test_original_coordinate_mask_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); image_path, mask_path = root / "retina.jpg", root / "mask.tif"
            Image.fromarray(np.full((896, 896, 3), 255, dtype=np.uint8), "RGB").save(image_path)
            mask = np.zeros((896, 896), dtype=np.uint8); mask[300, 300] = 255
            Image.fromarray(mask, "L").save(mask_path)
            with Image.open(image_path) as image:
                _, _, plans = MODULE.tile_plan(image.convert("RGB"))
            record = {"image": image_path, "stem": "synthetic", "masks": {name: (mask_path if name == "microaneurysms" else None) for name in (*MODULE.LESIONS, "optic_disc")}}
            labels, coverage = MODULE.tile_mask_labels(record, plans)
        self.assertEqual(labels.shape, (9, 5))
        self.assertEqual(int(labels[:, 0].sum()), 4)  # 50%-overlap maps the pixel to four tiles.
        self.assertTrue(np.all(coverage >= .8))
