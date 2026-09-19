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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/string_utils.h>

namespace mochi {

template <typename T>
constexpr bool IsArithmeticNdArray() {
  if constexpr (kIsNdArray<T>) {
    using E = typename T::element_type;
    return std::is_arithmetic_v<E>;
  }
  return false;
}

// A wrapper for the Span class, which Python can iterate without necessarily copying the data
template <typename T>
class PySpan {
 public:
  using NonConstT = std::remove_const_t<T>;

  PySpan() = default;
  PySpan(mochi::Span<T> span) : _span(span) {}

  // Iterator support for Python
  T* begin() const {
    return _span.data();
  }
  T* end() const {
    return _span.end();
  }

  // Size and indexing
  size_t size() const {
    return _span.size();
  }
  T& operator[](size_t index) const {
    if (index >= _span.size()) {
      throw nanobind::index_error();
    }
    return _span[index];
  }

 private:
  mochi::Span<T> _span;
};

template <typename T>
struct SpanBufferProtocol {
  using SpanT = PySpan<T>;
  using NonConstT = std::remove_const_t<T>;

  static int GetBuffer(PyObject* exporter, Py_buffer* view, int flags) noexcept {
    auto* self = nanobind::inst_ptr<SpanT>(nanobind::handle(exporter));
    if constexpr (IsArithmeticNdArray<T>()) {
      using E = typename NonConstT::element_type;
      using Scalar = std::remove_const_t<E>;
      constexpr int kNumDims = NonConstT::num_dims;
      std::array<Py_ssize_t, 1 + kNumDims> shape{};
      shape[0] = static_cast<Py_ssize_t>(self->size());
      for (int i = 0; i < kNumDims; ++i) {
        shape[1 + i] = static_cast<Py_ssize_t>(NonConstT::dims[i]);
      }
      return FillPythonBuffer(
          exporter,
          view,
          flags,
          const_cast<Scalar*>(reinterpret_cast<Scalar const*>(self->begin())),
          sizeof(Scalar),
          PythonBufferFormat<Scalar>(),
          std::is_const_v<T>,
          shape.data(),
          shape.size());
    } else {
      std::array<Py_ssize_t, 1> shape{static_cast<Py_ssize_t>(self->size())};
      return FillPythonBuffer(
          exporter,
          view,
          flags,
          const_cast<NonConstT*>(self->begin()),
          sizeof(NonConstT),
          PythonBufferFormat<NonConstT>(),
          std::is_const_v<T>,
          shape.data(),
          shape.size());
    }
  }

  inline static PyType_Slot kSlots[] = {
      {Py_bf_getbuffer, reinterpret_cast<void*>(GetBuffer)},
      {Py_bf_releasebuffer, reinterpret_cast<void*>(ReleasePythonBuffer)},
      {0, nullptr},
  };
};

} // namespace mochi

namespace nanobind::detail {

// Cast Python object to Span<T>.
template <class T>
struct type_caster<mochi::Span<T>> {
 public:
  using NonConstT = std::remove_const_t<T>;
  NB_TYPE_CASTER(mochi::Span<T>, const_name("Span[") + make_caster<T>::Name + const_name("]"))

  // Conversion from Python to C++. Must be noexcept -- never throw; return false to reject.
  bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
    return mochi::TranslatePythonCasterExceptions(
        [&] { return FromPython(src); }, "Unexpected error converting a Python iterable to Span");
  }

  bool FromPython(handle src) {
    // Direct recognition of a mochi::DynamicArray<NonConstT> instance (zero-copy, all T).
    // convert=false so we only match actual DynamicArray instances, never an implicitly-created
    // temporary.
    {
      mochi::DynamicArray<NonConstT>* dynamicArray = nullptr;
      if (try_cast<mochi::DynamicArray<NonConstT>*>(src, dynamicArray, false) && dynamicArray) {
        value = mochi::Span<T>(dynamicArray->data(), dynamicArray->size());
        _source = borrow<object>(src); // Keep the Python object alive
        return true;
      }
    }

    if constexpr (std::is_arithmetic_v<NonConstT>) {
      // bytes -> Span<char const>.
      if constexpr (std::is_same_v<NonConstT, char>) {
        if (PyBytes_Check(src.ptr())) {
          char* buffer = nullptr;
          Py_ssize_t size = 0;
          if (PyBytes_AsStringAndSize(src.ptr(), &buffer, &size) == 0) {
            value = mochi::Span<T>(buffer, static_cast<size_t>(size));
            _source = borrow<object>(src);
            return true;
          }
          PyErr_Clear();
        }
        if (PyByteArray_Check(src.ptr())) {
          value = mochi::Span<T>(
              PyByteArray_AsString(src.ptr()), static_cast<size_t>(PyByteArray_Size(src.ptr())));
          _source = borrow<object>(src);
          return true;
        }
      }
      // numpy / buffer-protocol zero-copy for a 1-D contiguous array of the right dtype.
      // The scalar type T carries constness: for a const Span this imports a read-only view; for
      // a non-const (output) Span it requires a writable array and writes through it. convert=false
      // guarantees no copy is made.
      using NdArr = nanobind::ndarray<T, nanobind::ndim<1>, nanobind::c_contig>;
      NdArr arr;
      if (try_cast<NdArr>(src, arr, false) && arr.device_type() == nanobind::device::cpu::value) {
        value = mochi::Span<T>(arr.data(), arr.shape(0));
        _source = borrow<object>(src); // Keep the numpy array (and its buffer) alive
        return true;
      }
    }

    // A non-const Span can be used as an output parameter, but we must guard against outputting
    // data to a temporary variable. Conversion to a non-const Span is only allowed if we can get
    // direct access to the destination memory (handled above), so reject here.
    if constexpr (!std::is_const_v<T>) {
      return false;
    } else {
      // Copy a Python iterable into a temporary mochi::DynamicArray held by this caster.
      Py_ssize_t len = PyObject_Length(src.ptr());
      if (len < 0) {
        PyErr_Clear();
        return false;
      }
      _storage.resize(static_cast<size_t>(len));
      PyObject* iter = PyObject_GetIter(src.ptr());
      if (!iter) {
        PyErr_Clear();
        return false;
      }
      size_t i = 0;
      bool ok = true;
      PyObject* item = nullptr;
      while ((item = PyIter_Next(iter)) != nullptr) {
        NonConstT converted{};
        if (!try_cast<NonConstT>(handle(item), converted)) {
          Py_DECREF(item);
          ok = false;
          break;
        }
        if (i < _storage.size()) {
          _storage[i] = converted;
        }
        ++i;
        Py_DECREF(item);
      }
      Py_DECREF(iter);
      if (PyErr_Occurred()) {
        PyErr_Clear();
        ok = false;
      }
      if (!ok) {
        return false;
      }
      value = mochi::Span<T>(_storage.data(), _storage.size());
      return true;
    }
  }

  // Zero-copy conversion from C++ to Python: wrap in a PySpan (which exposes iteration/__array__).
  static handle from_cpp(mochi::Span<T> src, rv_policy policy, cleanup_list* cleanup) noexcept {
    return make_caster<mochi::PySpan<T>>::from_cpp(mochi::PySpan<T>(src), policy, cleanup);
  }

 private:
  mochi::DynamicArray<NonConstT> _storage; // Keeps copied iterable data alive during the call
  object _source; // Keeps the source Python object (DynamicArray, numpy array, bytes) alive
};

} // namespace nanobind::detail

namespace mochi {

template <typename T>
auto DefSpan(nanobind::module_& m, char const* pyName) {
  namespace nb = nanobind;
  using NonConstT = std::remove_const_t<T>;
  auto c = [&]() {
    if constexpr (std::is_arithmetic_v<NonConstT> || IsArithmeticNdArray<T>()) {
      return nb::class_<PySpan<T>>(m, pyName, nb::type_slots(SpanBufferProtocol<T>::kSlots));
    } else {
      return nb::class_<PySpan<T>>(m, pyName);
    }
  }();
  nb::handle scope = c; // Non-owning handle to the class (owned by the module) for make_iterator.
  c.def(nb::init<>());
  c.def(
      "__init__",
      [](PySpan<T>* self, DynamicArray<NonConstT>& arr) {
        new (self) PySpan<T>(Span<T>(arr.data(), arr.size()));
      },
      nb::keep_alive<1, 2>(),
      nb::arg("array"));
  c.def("__len__", &PySpan<T>::size);
  // For arithmetic types, return by value (Python ints/floats are immutable anyway).
  // For const spans, also return by copy to prevent mutation through a const view.
  // For all other types, return by reference so that span[i].field = value modifies the
  // actual element — matching standard Python list semantics.
  if constexpr (std::is_arithmetic_v<std::remove_const_t<T>> || std::is_const_v<T>) {
    c.def("__getitem__", &PySpan<T>::operator[]);
  } else {
    c.def("__getitem__", &PySpan<T>::operator[], nb::rv_policy::reference_internal);
  }
  c.def("__setitem__", [](PySpan<T>& self, size_t index, T const& value) {
    if constexpr (std::is_const_v<T>) {
      throw nb::type_error("Cannot modify a const Span");
    } else {
      self[index] = value;
    }
  });
  if constexpr (std::is_arithmetic_v<std::remove_const_t<T>> || std::is_const_v<T>) {
    c.def(
        "__iter__",
        [scope](PySpan<T>& s) {
          return nb::make_iterator<nb::rv_policy::copy>(scope, "Iterator", s.begin(), s.end());
        },
        nb::keep_alive<0, 1>());
  } else {
    c.def(
        "__iter__",
        [scope](PySpan<T>& s) {
          return nb::make_iterator<nb::rv_policy::reference_internal>(
              scope, "Iterator", s.begin(), s.end());
        },
        nb::keep_alive<0, 1>());
  }
  c.def("tolist", [](PySpan<T> const& s) {
    return std::vector<std::remove_const_t<T>>(s.begin(), s.end());
  });

  // copy.copy / copy.deepcopy. A Span is a non-owning view. Copy it to a DynamicArray instead.
  c.def("__copy__", [](PySpan<T> const& self) {
    return DynamicArray<NonConstT>(self.begin(), self.end());
  });
  c.def(
      "__deepcopy__",
      [](PySpan<T> const& self, nb::dict) {
        return DynamicArray<NonConstT>(self.begin(), self.end());
      },
      nb::arg("memo"));

  if constexpr (std::is_arithmetic_v<T>) {
    c.def(
        "__array__",
        [](PySpan<T>& s, nb::object /*dtype*/, nb::object /*copy*/) -> nb::object {
          nb::object owner = nb::find(s);
          nb::ndarray<nb::numpy, T, nb::ndim<1>> arr(s.begin(), {s.size()}, owner);
          return nb::cast(std::move(arr));
        },
        nb::arg("dtype") = nb::none(),
        nb::arg("copy") = nb::none());
  }

  if constexpr (IsArithmeticNdArray<T>()) {
    c.def(
        "__array__",
        [](PySpan<T>& s, nb::object /*dtype*/, nb::object /*copy*/) -> nb::object {
          using E = typename T::element_type;
          using NonConstE = std::remove_const_t<E>;
          using EScalar = std::conditional_t<std::is_const_v<T>, NonConstE const, NonConstE>;
          int constexpr kNumDims = T::num_dims;
          std::array<size_t, 1 + kNumDims> shape{};
          shape[0] = s.size();
          for (int i = 0; i < kNumDims; ++i) {
            shape[1 + i] = static_cast<size_t>(T::dims[i]);
          }
          nb::object owner = nb::find(s);
          auto* dataPtr = reinterpret_cast<EScalar*>(s.begin());
          nb::ndarray<nb::numpy, EScalar> arr(dataPtr, 1 + kNumDims, shape.data(), owner);
          return nb::cast(std::move(arr));
        },
        nb::arg("dtype") = nb::none(),
        nb::arg("copy") = nb::none());
  }

  return c;
}

} // namespace mochi
