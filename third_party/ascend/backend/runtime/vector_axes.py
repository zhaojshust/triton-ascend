from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


SPLIT_AXIS_FALLBACK_SIZE = 4096
DEFAULT_AXIS_FALLBACK_SIZE = 512


@dataclass
class VectorAxis:
    name: str
    length_expr: Optional[str] = None
    dynamic_source: str = "none"
    split_param: Optional[str] = None
    tiling_param: Optional[str] = None
    is_low_dim: bool = False
    is_reduction: bool = False
    pid_dim: Optional[int] = None


@dataclass
class VectorAxes:
    axes: "OrderedDict[str, VectorAxis]" = field(default_factory=OrderedDict)
    _low_dim_axis_names: List[str] = field(default_factory=list)
    _reduction_axis_names: List[str] = field(default_factory=list)

    @classmethod
    def from_hints_axes(
        cls,
        hints_axes: Optional[Mapping[str, str]] = None,
    ) -> "VectorAxes":
        result = cls()
        for axis_name, length_expr in dict(hints_axes or {}).items():
            axis = result.ensure_axis(axis_name)
            axis.length_expr = length_expr
        return result

    def ensure_axis(self, name: str) -> VectorAxis:
        axis = self.axes.get(name)
        if axis is None:
            axis = VectorAxis(name=name)
            self.axes[name] = axis
        self._apply_axis_property_flags(axis)
        return axis

    def apply_semantic_fields(
        self,
        *,
        split_params: Optional[Mapping[str, str]] = None,
        tiling_params: Optional[Mapping[str, str]] = None,
        low_dim_axes: Optional[Iterable[str]] = None,
        reduction_axes: Optional[Iterable[str]] = None,
        axis_pid_dims: Optional[Mapping[str, int]] = None,
        axis_length_exprs: Optional[Mapping[str, str]] = None,
        axis_dynamic_sources: Optional[Mapping[str, str]] = None,
    ) -> None:
        for axis_name, param in dict(split_params or {}).items():
            self.ensure_axis(axis_name).split_param = param
        for axis_name, param in dict(tiling_params or {}).items():
            self.ensure_axis(axis_name).tiling_param = param

        for axis_name in list(low_dim_axes or []):
            self._record_axis_property(self._low_dim_axis_names, axis_name)
            axis = self.axes.get(axis_name)
            if axis is not None:
                axis.is_low_dim = True

        for axis_name in list(reduction_axes or []):
            self._record_axis_property(self._reduction_axis_names, axis_name)
            axis = self.axes.get(axis_name)
            if axis is not None:
                axis.is_reduction = True

        for axis_name, pid_dim in dict(axis_pid_dims or {}).items():
            self.ensure_axis(axis_name).pid_dim = pid_dim
        for axis_name, expr in dict(axis_length_exprs or {}).items():
            self.ensure_axis(axis_name).length_expr = expr
        for axis_name, source in dict(axis_dynamic_sources or {}).items():
            self.ensure_axis(axis_name).dynamic_source = source

    def materialize_axis_sizes(
        self,
        all_args: Optional[Mapping[str, object]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, object]]]:
        all_args = dict(all_args or {})
        axis_sizes: Dict[str, int] = {}
        diagnostics: Dict[str, Dict[str, object]] = {}

        for axis in self.axes.values():
            if not self._should_materialize_axis(axis):
                continue

            axis_sizes[axis.name], diagnostics[axis.name] = self._materialize_axis_size(
                axis,
                all_args,
            )

        return axis_sizes, diagnostics

    @property
    def split_params(self) -> Dict[str, str]:
        return {
            axis.name: axis.split_param
            for axis in self.axes.values()
            if isinstance(axis.split_param, str) and axis.split_param
        }

    @property
    def tiling_params(self) -> Dict[str, str]:
        return {
            axis.name: axis.tiling_param
            for axis in self.axes.values()
            if isinstance(axis.tiling_param, str) and axis.tiling_param
        }

    @property
    def low_dim_axes(self):
        return list(self._low_dim_axis_names)

    @property
    def reduction_axes(self):
        return list(self._reduction_axis_names)

    @property
    def axis_pid_dims(self) -> Dict[str, int]:
        return {
            axis.name: axis.pid_dim
            for axis in self.axes.values()
            if isinstance(axis.pid_dim, int)
        }

    @property
    def axis_length_exprs(self) -> Dict[str, str]:
        return {
            axis.name: axis.length_expr
            for axis in self.axes.values()
            if isinstance(axis.length_expr, str) and axis.length_expr
        }

    @property
    def axis_dynamic_sources(self) -> Dict[str, str]:
        return {
            axis.name: axis.dynamic_source
            for axis in self.axes.values()
            if (
                isinstance(axis.dynamic_source, str)
                and axis.dynamic_source
                and axis.dynamic_source != "none"
            )
        }

    @staticmethod
    def _should_materialize_axis(axis: VectorAxis) -> bool:
        return bool(axis.split_param or axis.tiling_param or axis.length_expr)

    @staticmethod
    def _record_axis_property(store: List[str], axis_name: str) -> None:
        if axis_name not in store:
            store.append(axis_name)

    def _apply_axis_property_flags(self, axis: VectorAxis) -> None:
        axis.is_low_dim = axis.name in self._low_dim_axis_names
        axis.is_reduction = axis.name in self._reduction_axis_names

    @staticmethod
    def _materialize_axis_size(
        axis: VectorAxis,
        all_args: Mapping[str, object],
    ) -> Tuple[int, Dict[str, object]]:
        runtime_value = all_args.get(axis.name)
        if VectorAxes._is_positive_int(runtime_value):
            return runtime_value, {
                "axis": axis.name,
                "length_expr": axis.length_expr,
                "dynamic_source": axis.dynamic_source or "none",
                "resolved_by": "axis_name_arg",
                "resolved_value": runtime_value,
            }

        if isinstance(axis.length_expr, str) and axis.length_expr:
            runtime_value = all_args.get(axis.length_expr)
            if VectorAxes._is_positive_int(runtime_value):
                return runtime_value, {
                    "axis": axis.name,
                    "length_expr": axis.length_expr,
                    "dynamic_source": axis.dynamic_source or "none",
                    "resolved_by": "length_expr_arg",
                    "resolved_value": runtime_value,
                }

            literal_value = VectorAxes._parse_positive_int_literal(axis.length_expr)
            if literal_value is not None:
                return literal_value, {
                    "axis": axis.name,
                    "length_expr": axis.length_expr,
                    "dynamic_source": axis.dynamic_source or "none",
                    "resolved_by": "literal",
                    "resolved_value": literal_value,
                }

        if axis.split_param:
            return SPLIT_AXIS_FALLBACK_SIZE, {
                "axis": axis.name,
                "length_expr": axis.length_expr,
                "dynamic_source": axis.dynamic_source or "none",
                "resolved_by": "default_split",
                "resolved_value": SPLIT_AXIS_FALLBACK_SIZE,
            }

        return DEFAULT_AXIS_FALLBACK_SIZE, {
            "axis": axis.name,
            "length_expr": axis.length_expr,
            "dynamic_source": axis.dynamic_source or "none",
            "resolved_by": "default_tiling",
            "resolved_value": DEFAULT_AXIS_FALLBACK_SIZE,
        }

    @staticmethod
    def _parse_positive_int_literal(value: str) -> Optional[int]:
        try:
            parsed = int(value, 10)
        except (TypeError, ValueError):
            return None
        if parsed > 0:
            return parsed
        return None

    @staticmethod
    def _is_positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
