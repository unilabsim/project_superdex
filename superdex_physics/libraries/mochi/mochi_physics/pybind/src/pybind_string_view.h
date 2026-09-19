/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "pybind_helpers.h"

#include <mochi_core/utils/dynamic_string.h>

namespace nanobind::detail {

// Caster for mochi::DynamicString (a std::basic_string with a custom StlAllocator). nanobind's
// bundled std::string caster only matches the default allocator, so this custom-allocator string
// needs its own caster. pybind11's stl.h matched any allocator, which is why this was implicit
// before. Converts to/from Python str.
template <>
struct type_caster<mochi::DynamicString> {
  NB_TYPE_CASTER(mochi::DynamicString, const_name("str"))

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
    return mochi::TranslatePythonCasterExceptions(
        [&] { return FromPython(src); }, "Unexpected error converting to DynamicString");
  }

  bool FromPython(handle src) {
    if (!src.is_valid()) {
      return false;
    }
    if (PyUnicode_Check(src.ptr())) {
      Py_ssize_t size = 0;
      char const* buffer = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
      if (!buffer) {
        PyErr_Clear();
        return false;
      }
      value = mochi::DynamicString(buffer, static_cast<size_t>(size));
      return true;
    }
    if (PyBytes_Check(src.ptr())) {
      char* buffer = nullptr;
      Py_ssize_t size = 0;
      if (PyBytes_AsStringAndSize(src.ptr(), &buffer, &size) != 0) {
        PyErr_Clear();
        return false;
      }
      value = mochi::DynamicString(buffer, static_cast<size_t>(size));
      return true;
    }
    return false;
  }

  static handle from_cpp(mochi::DynamicString const& src, rv_policy, cleanup_list*) noexcept {
    return PyUnicode_FromStringAndSize(src.data(), static_cast<Py_ssize_t>(src.size()));
  }
};

// Full specialization of type_caster for std::string_view that keeps the source Python object
// alive. nanobind's default caster points directly into CPython's internal UTF-8 buffer without
// holding a reference to the Python str object.
//
// This is normally safe while the GIL is held, but becomes unsafe when the GIL is released via
// nb::call_guard<nb::gil_scoped_release>(): another thread could garbage-collect the string,
// invalidating the string_view. The caster instance lives on the dispatcher stack for the whole
// duration of the wrapped call, so storing a strong reference here prevents GC while the
// string_view is in use.
//
// This follows the same pattern as type_caster<mochi::Span<T>> in pybind_span.h.
template <>
struct type_caster<std::string_view> {
  NB_TYPE_CASTER(std::string_view, const_name("str"))

  // Conversion from Python to C++
  bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
    return mochi::TranslatePythonCasterExceptions(
        [&] { return FromPython(src); }, "Unexpected error converting to string_view");
  }

  bool FromPython(handle src) {
    if (!src.is_valid()) {
      return false;
    }
    if (PyUnicode_Check(src.ptr())) {
      Py_ssize_t size = 0;
      char const* buffer = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
      if (!buffer) {
        PyErr_Clear();
        return false;
      }
      value = std::string_view(buffer, static_cast<size_t>(size));
      _source = borrow<object>(src);
      return true;
    }
    if (PyBytes_Check(src.ptr())) {
      char const* buffer = PyBytes_AsString(src.ptr());
      if (!buffer) {
        PyErr_Clear();
        return false;
      }
      value = std::string_view(buffer, static_cast<size_t>(PyBytes_Size(src.ptr())));
      _source = borrow<object>(src);
      return true;
    }
    return false;
  }

  // Conversion from C++ to Python
  static handle from_cpp(std::string_view src, rv_policy, cleanup_list*) noexcept {
    return PyUnicode_FromStringAndSize(src.data(), static_cast<Py_ssize_t>(src.size()));
  }

 private:
  object _source; // Prevent GC of the Python str/bytes while the string_view is alive
};

} // namespace nanobind::detail
