# ---
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence as Seq
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias

from core import (BuilderEmitter, CallableHof, Emitter,
                  FrontendLoweringEmitter, Jaxpr, JaxType, Primitive, TraceVal,
                  eval_emitter, none_ty)

# === embedding ===

# We keep a "current emitter" as a globally-readable context. This is purely to
# reduce clutter in user-facing code. Internally we pass around emitters
# explicitly so it's easier to follow the flow of data.
@dataclass
class CurrentEmitter:
  emitter: Emitter

@contextmanager
def set_current_emitter(emitter):
  prev = current_emitter.emitter
  current_emitter.emitter = emitter
  try:
    yield
  finally:
    current_emitter.emitter = prev

def top_level_emitter():
  return FrontendLoweringEmitter(eval_emitter)

current_emitter = CurrentEmitter(top_level_emitter())

def emit_primitive(p: Primitive, args: Seq[PyVal], funargs: Seq[Jaxpr] = ()) -> TraceVal:
  assert all(isinstance(fn, Jaxpr) for fn in funargs), funargs
  emitter = current_emitter.emitter
  args_canonical = [canonicalize_pyval(arg) for arg in args]
  arg_tys = [arg.ty for arg in args_canonical]
  if isinstance(p, CallableHof):
    result_ty = none_ty
  else:
    fun_tys = [f.ty for f in funargs]
    result_ty = p.result_type(*(tuple(fun_tys) + tuple(arg_tys)))
  return emitter.emit_primitive(p, result_ty, args_canonical, funargs)

# This turns a function that reads the implicit "current_emitter" context into
# one that takes the emitter explicitly, conforming to the `OpStream` API
@dataclass
class WithExplicitEmitter:
  f: Callable
  def __call__(self, emitter, *args):
    with set_current_emitter(emitter):
      return self.f(*args)

# `emit` requires each argument to be a `TraceVal`, which is either a `JaxVal`
# or a `Tracer`. A PyVal could be a tuples of tracers, or a Python float
# representing a rank-0 array. We canonicalize these to a `TraceVal` before
# calling `emit`.

PyVal: TypeAlias = Any
pyval_canonicalizers = {}

def register_canonicalizer(t, f):
  pyval_canonicalizers[t] = f

# may use current emitter, for example to build a FancyTuple from a python
# tuple.
def canonicalize_pyval(x: PyVal) -> TraceVal:
  if isinstance(x, TraceVal):
    return x
  elif fn := pyval_canonicalizers.get(type(x)):
    return fn(x)
  else:
    raise TypeError(f'Unrecognized JAX type: {type(x)}')

def trace_to_jaxpr(f: Callable, arg_types: Seq[JaxType]) -> Jaxpr:
  builder = BuilderEmitter(arg_types)
  with set_current_emitter(builder):
    result = canonicalize_pyval(f(*builder.args))
    return builder.build(result)
