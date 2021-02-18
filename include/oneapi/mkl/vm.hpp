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

#ifndef _ONEMKL_VM_HPP_
#define _ONEMKL_VM_HPP_

#include <cstdint>
#include <complex>

#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

#include "oneapi/mkl/vm/types.hpp"

#include "oneapi/mkl/vm/predicates.hpp"
#include "oneapi/mkl/vm/detail/vm_loader.hpp"

#ifdef ENABLE_MKLCPU_BACKEND
#include "oneapi/mkl/vm/detail/mklcpu/vm_ct.hpp"
#endif
#ifdef ENABLE_MKLGPU_BACKEND
#include "oneapi/mkl/vm/detail/mklgpu/vm_ct.hpp"
#endif

namespace oneapi {
namespace mkl {
namespace vm {

// Service functions
static inline oneapi::mkl::vm::mode get_mode(cl::sycl::queue & queue) {
    // TO DO: add pre- and post conditions if needed
    // get_mode_precondition(queue);
    auto mode = detail::get_mode(get_device_id(queue), queue);
    // get_mode_postcondition(queue);
    return mode;
}
static inline oneapi::mkl::vm::mode set_mode(cl::sycl::queue & queue, oneapi::mkl::vm::mode new_mode) {
    // TO DO: add pre- and post conditions if needed
    // set_mode_precondition(queue, new_mode);
    auto mode = detail::set_mode(get_device_id(queue), queue, new_mode);
    // set_mode_postcondition(queue, new_mode);
    return mode;
}

static inline oneapi::mkl::vm::status get_status(cl::sycl::queue & queue) {
    // TO DO: add pre- and post conditions if needed
    // get_status_precondition(queue);
    auto status = detail::get_status(get_device_id(queue), queue);
    // get_status_postcondition(queue);
    return status;
}
static inline oneapi::mkl::vm::status set_status(cl::sycl::queue & queue, oneapi::mkl::vm::status new_status) {
    // TO DO: add pre- and post conditions if needed
    // set_status_precondition(queue, new_status);
    auto status = detail::set_status(get_device_id(queue), queue, new_status);
    // set_status_postcondition(queue, new_status);
    return status;
}
static inline oneapi::mkl::vm::status clear_status(cl::sycl::queue & queue) {
    // TO DO: add pre- and post conditions if needed
    // clear_status_precondition(queue);
    auto status = detail::clear_status(get_device_id(queue), queue);
    // clear_status_postcondition(queue);
    return status;
}


static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, given_mode, eh);
    // add_postcondition(q, n, a, b, y, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, given_mode, eh);
    // add_postcondition(q, n, a, b, y, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, given_mode, eh);
    // add_postcondition(q, n, a, b, y, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, given_mode, eh);
    // add_postcondition(q, n, a, b, y, given_mode, eh);
    return event;
}

static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<float> > eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, depends, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, depends, given_mode, eh);
    // add_postcondition(q, n, a, b, y, depends, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<std::complex<double> > eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, depends, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, depends, given_mode, eh);
    // add_postcondition(q, n, a, b, y, depends, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<float> eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, depends, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, depends, given_mode, eh);
    // add_postcondition(q, n, a, b, y, depends, given_mode, eh);
    return event;
}
static inline cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends = {}, oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined, oneapi::mkl::vm::error_handler<double> eh = {}) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(q, n, a, b, y, depends, given_mode, eh);
    auto event = detail::add(get_device_id(q), q, n, a, b, y, depends, given_mode, eh);
    // add_postcondition(q, n, a, b, y, depends, given_mode, eh);
    return event;
}

} // namespace vm
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_VM_HPP_
