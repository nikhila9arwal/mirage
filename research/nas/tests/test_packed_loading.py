"""Check cache rejection and exact tensor verification using tiny CPU data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/qwen_decoder_layer'))
import packed_weights as packed


class PackedLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        config = self.snapshot / 'config.json'
        config.write_text('{}')
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        # These are metadata fixtures, not fake model execution. Tensor bytes
        # are tested independently below; no GPU is initialized by this suite.
        names = [f'layer_{i:02d}.safetensors' for i in range(48)] + ['common.safetensors']
        for name in names:
            (self.cache / name).write_bytes(b'x')
        self.manifest = dict(format=packed.FORMAT, layers=48,
            source=dict(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()),
            files=[dict(file=name, bytes=1) for name in names])
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.cache / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_complete_metadata(self):
        self.assertEqual(packed.cache_manifest(self.cache, self.snapshot), self.manifest)

    def test_missing_or_duplicate_layer(self):
        self.manifest['files'][-2] = self.manifest['files'][0]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Incomplete/duplicate'):
            packed.cache_manifest(self.cache, self.snapshot)

    def test_wrong_checkpoint_config(self):
        (self.snapshot / 'config.json').write_text('{"different": true}')
        with self.assertRaisesRegex(ValueError, 'config differs'):
            packed.cache_manifest(self.cache, self.snapshot)

    def test_truncated_file(self):
        (self.cache / 'layer_00.safetensors').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            packed.cache_manifest(self.cache, self.snapshot)

    def weights_fixture(self):
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        layer = dict(w13=tensor, cos=torch.ones(2, dtype=torch.bfloat16))
        weights = dict(layers=[layer])
        manifest = dict(files=[dict(file='layer_00.safetensors', tensors=dict(
            w13=dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                     sha256=packed.fingerprint(tensor))))],
            rope=dict(cos=packed.fingerprint(layer['cos'])))
        return weights, manifest

    def test_exact_bytes(self):
        weights, manifest = self.weights_fixture()
        self.assertEqual(packed.verify_weights(weights, manifest), 1)

    def test_changed_weight(self):
        weights, manifest = self.weights_fixture()
        weights['layers'][0]['w13'][0, 0] = 9
        with self.assertRaisesRegex(ValueError, 'Weight-byte mismatch'):
            packed.verify_weights(weights, manifest)

    def test_wrong_shape_or_dtype(self):
        for change in (lambda t:t.reshape(4, 3), lambda t:t.float()):
            with self.subTest(change=change):
                weights, manifest = self.weights_fixture()
                weights['layers'][0]['w13'] = change(weights['layers'][0]['w13'])
                with self.assertRaisesRegex(ValueError, 'Shape/dtype mismatch'):
                    packed.verify_weights(weights, manifest)

    def test_changed_rope(self):
        weights, manifest = self.weights_fixture()
        weights['layers'][0]['cos'][0] = 0
        with self.assertRaisesRegex(ValueError, 'RoPE bytes differ'):
            packed.verify_weights(weights, manifest)


if __name__ == '__main__':
    unittest.main()
