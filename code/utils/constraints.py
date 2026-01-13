# Portions of this file are derived from scvi-tools, part of the scverse® project.
#
# Copyright (c) 2025 Yosef Lab, Weizmann Institute of Science
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# Modifications and additional code:
# Copyright (c) 2026 Jiayang Sun
#
# Note: Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

import torch
from torch.distributions.constraints import Constraint


class _Optional(Constraint):
    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if value is None:
            return torch.ones(1, dtype=torch.bool)
        return self.constraint.check(value)

    def __repr__(self) -> str:
        return f"Optional({self.constraint})"


def optional_constraint(constraint: Constraint) -> Constraint:
    """Returns a wrapped constraint that allows optional values."""
    return _Optional(constraint)


class _OpenInterval(Constraint):
    """Constrain to a real interval ``(lower_bound, upper_bound)``"""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value: torch.Tensor):
        return (self.lower_bound < value) & (value < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        return fmt_string


# Public interface
open_interval = _OpenInterval
