"""idq - driving-QA reasoning evaluation harness.

Design invariants, enforced by tests:

1. Collection and scoring never share a process. A scoring bug costs zero dollars.
2. Every cached record carries the full raw provider response, so metrics that
   were not anticipated at collection time can still be recovered later.
3. The cache key covers everything that can change an answer. If it is not in
   the key, changing it must not change the output.
4. Nothing in this package reads a credential from anywhere but os.environ.
"""

__version__ = "0.1.0"
