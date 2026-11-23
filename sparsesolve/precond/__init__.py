"""预条件器集合。"""
from .base import (Preconditioner, IdentityPrecond, make_preconditioner,
                   PRECONDITIONER_NAMES, preconditioner_display_name)
from .jacobi import JacobiPrecond
from .incomplete import IC0Precond, ILU0Precond
from .ssor import SSORPrecond

__all__ = [
    "Preconditioner", "IdentityPrecond", "JacobiPrecond", "IC0Precond",
    "ILU0Precond", "SSORPrecond", "make_preconditioner",
    "PRECONDITIONER_NAMES", "preconditioner_display_name",
]
