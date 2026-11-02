# Reviewed CUDA TCN selector version 1

This source package preserves the exact independently reviewed selector and its frozen trainer/projection dependencies. The public entry point is `scripts/select_cuda_tcn_blend.py`; it verifies these files' SHA256 values before loading them and supplies the workspace root. No ignored experiment artifact is required to import the selector or display its CLI.

The selector SHA256 is `b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f`. Historical 2021 selection lock `cuda-tcn-blend-2021-001` binds these exact bytes, with selected neural weight zero. These source files were copied byte-for-byte from the independently reviewed package `tcn_audit_reconciled_20261102T142849Z_a892a747`; its artifacts remain the original audit record. Do not edit this package when implementing a new protocol or changing an existing lock's interpretation; create a separately reviewed source version instead.

The nested layout is intentional: the immutable selector resolves source snapshots relative to its package root. Package hash validation and scoped imports prevent this minimal package from replacing unrelated workspace imports. Competition data, fitted checkpoints, downloaded predictions and selection locks remain external ignored artifacts and are required only for actual experiment verification/selection.
