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

from ops import TupleType, add, cond, greater, jax_float, sin, trace_to_jaxpr


def fun0():
  x = add(1.0, 2.0)
  return x

print(fun0())
print("============")
print(trace_to_jaxpr(fun0, []))
print()

def fun1():
  x = add(1.0, 2.0)
  return cond(greater(x, 1.), lambda: add(x, sin(1.0)), lambda: 1.0)

print(fun1())
print("============")
print(trace_to_jaxpr(fun1, []))
print()

def fun2(xy):
  x, y = xy
  return add(x, y)

print(fun2((1.0, 2.0)))
print("============")
print(trace_to_jaxpr(fun2, [TupleType((jax_float, jax_float))]))
