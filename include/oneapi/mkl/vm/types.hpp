/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/
#ifndef _ONEMKL_VM_TYPES_HPP_
#define _ONEMKL_VM_TYPES_HPP_

#include <cstdint>
#include <complex>
#include <type_traits>

namespace oneapi {
namespace mkl {
namespace vm {

namespace enums {
enum class mode : std::uint32_t {
    not_defined = 0x0,

    la = 0x1,
    ha = 0x2,
    ep = 0x3,

    global_status_report = 0x100,
};

enum class status : std::uint32_t {
    not_defined = 0x0,

    success = 0x0,

    errdom = 0x1,
    sing = 0x2,
    overflow = 0x4,
    underflow = 0x8,

    accuracy_warning = 0x80,
    fix_all = 0xff,
};

template <typename T>
struct bits_enabled { static constexpr bool enabled = false; };

template <>
struct bits_enabled<mode> { static constexpr bool enabled = true; };

template <>
struct bits_enabled<status> { static constexpr bool enabled = true; };


template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type
operator |(T lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type> (lhs) | static_cast<typename std::underlying_type<T>::type>(rhs);
    return static_cast<T>(r);
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type &
operator |=(T & lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type>(lhs) | static_cast<typename std::underlying_type<T>::type>(rhs);
    lhs = static_cast<T>(r);
    return lhs;
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type
operator &(T lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type>(lhs) & static_cast<typename std::underlying_type<T>::type>(rhs);
    return static_cast<T>(r);
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type
operator &=(T & lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type>(lhs) & static_cast<typename std::underlying_type<T>::type>(rhs);
    lhs = static_cast<T>(r);
    return lhs;
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type
operator ^(T lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type>(lhs) ^ static_cast<typename std::underlying_type<T>::type>(rhs);
    return static_cast<T>(r);
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, T>::type
operator ^=(T & lhs, T rhs) {
    auto r = static_cast<typename std::underlying_type<T>::type>(lhs) ^ static_cast<typename std::underlying_type<T>::type>(rhs);
    lhs = static_cast<T>(r);
    return lhs;
}


template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, bool>::type
operator !(T v) { return (0 == static_cast<typename std::underlying_type<T>::type>(v)); }

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, bool>::type
has_any(T v, T mask) {
    auto r = static_cast<typename std::underlying_type<T>::type>(v) & static_cast<typename std::underlying_type<T>::type>(mask);
    return (0 != r);
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, bool>::type
has_all(T v, T mask) {
    auto r = static_cast<typename std::underlying_type<T>::type>(v) & static_cast<typename std::underlying_type<T>::type>(mask);
    return (static_cast<typename std::underlying_type<T>::type>(mask) == r);
}

template <typename T>
typename std::enable_if<bits_enabled<T>::enabled, bool>::type
has_only(T v, T mask) {
    auto r = static_cast<typename std::underlying_type<T>::type>(v) ^ static_cast<typename std::underlying_type<T>::type>(mask);
    return (0 == r);
}

} // namespace enums

using enums::mode;
using enums::status;

namespace detail {
using std::int64_t;
namespace one_vm = oneapi::mkl::vm;

template <typename T>
struct error_handler {
    bool enabled_;
    bool is_usm_;

    sycl::buffer<one_vm::status, 1> buf_status_;
    one_vm::status * usm_status_;
    int64_t len_;

    one_vm::status status_to_fix_;
    T fixup_value_;
    bool copy_sign_;

    error_handler():
        enabled_ { false },
        is_usm_  { false },

        buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
        usm_status_ { nullptr },
        len_ { 0 },
        status_to_fix_ { one_vm::status::not_defined },
        fixup_value_ { T {} },
        copy_sign_ { false }
        { }

    error_handler(one_vm::status status_to_fix, T fixup_value, bool copy_sign = false):
        enabled_ { true },
        is_usm_  { false },

        buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
        usm_status_ { nullptr },
        len_ { 0 },
        status_to_fix_ { status_to_fix },
        fixup_value_ { fixup_value },
        copy_sign_ { copy_sign }
        { }

    error_handler(one_vm::status * array, std::int64_t len = 1, one_vm::status status_to_fix = one_vm::status::not_defined, T fixup_value = {}, bool copy_sign = false):
        enabled_ { true },
        is_usm_  { true },

        buf_status_ { sycl::buffer<one_vm::status, 1> { 1 } },
        usm_status_ { array },
        len_ { len },
        status_to_fix_ { status_to_fix },
        fixup_value_ { fixup_value },
        copy_sign_ { copy_sign }
        { }

    error_handler(sycl::buffer<one_vm::status, 1> & buf, std::int64_t len = 1, one_vm::status status_to_fix = one_vm::status::not_defined, T fixup_value = {}, bool copy_sign = false):
        enabled_ { true },
        is_usm_  { false },

        buf_status_ { buf },
        usm_status_ { nullptr },
        len_ { len },
        status_to_fix_ { status_to_fix },
        fixup_value_ { fixup_value },
        copy_sign_ { copy_sign }
        { }

}; // struct error_handler
} // namespace detail

using detail::error_handler;

} // namespace vm
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_VM_TYPES_HPP_