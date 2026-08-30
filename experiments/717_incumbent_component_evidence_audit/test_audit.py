from __future__ import annotations

import unittest

from audit import canonical_sha256


class AuditContractTest(unittest.TestCase):
    def test_canonical_hash_is_order_independent(self) -> None:
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
