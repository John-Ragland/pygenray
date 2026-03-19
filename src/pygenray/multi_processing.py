"""
Shared-memory multiprocessing stubs. Multiprocessing has been removed in favour
of JAX-based parallelism (vmap). These stubs preserve import compatibility for
any code that imports from this module directly.
"""


def _init_shared_memory(*args, **kwargs):
    raise DeprecationWarning(
        "Shared memory multiprocessing is removed. Use shoot_rays() sequentially; vmap support is coming."
    )


def _unpack_shared_memory(*args, **kwargs):
    raise DeprecationWarning(
        "Shared memory multiprocessing is removed. Use shoot_rays() sequentially; vmap support is coming."
    )


__all__ = ['_init_shared_memory', '_unpack_shared_memory']
