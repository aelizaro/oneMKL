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

#ifndef _ONEMKL_VL_DETAIL_MKLCPU_CT_HPP__
#define _ONEMKL_VL_DETAIL_MKLCPU_CT_HPP__

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"

#include "oneapi/mkl/vm/detail/vm_ct_backends.hpp"
#include "oneapi/mkl/vm/detail/mklcpu/onemkl_vm_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace vm {

oneapi::mkl::vm::mode get_mode(backend_selector<backend::mklcpu> selector) {
    // TO DO: add pre- and post conditions if needed
    // get_mode_precondition(selector.get_queue());
    auto mode = oneapi::mkl::vm::mklcpu::get_mode(selector.get_queue());
    // get_mode_postcondition(selector.get_queue());
    return mode;
}
oneapi::mkl::vm::mode set_mode(backend_selector<backend::mklcpu> selector,
                               oneapi::mkl::vm::mode new_mode) {
    // TO DO: add pre- and post conditions if needed
    // set_mode_precondition(backend_seselector.get_queue(), new_mode);
    auto mode = oneapi::mkl::vm::mklcpu::set_mode(selector.get_queue(), new_mode);
    // set_mode_postcondition(backend_seselector.get_queue(), new_mode);
    return mode;
}

oneapi::mkl::vm::status get_status(backend_selector<backend::mklcpu> selector) {
    // TO DO: add pre- and post conditions if needed
    // get_status_precondition(queue);
    auto status = oneapi::mkl::vm::mklcpu::get_status(selector.get_queue());
    // get_status_postcondition(queue);
    return status;
}
oneapi::mkl::vm::status set_status(backend_selector<backend::mklcpu> selector,
                                   oneapi::mkl::vm::status new_status) {
    // TO DO: add pre- and post conditions if needed
    // set_status_precondition(backend_seselector.get_queue(), new_status);
    auto status = oneapi::mkl::vm::mklcpu::set_status(selector.get_queue(), new_status);
    // set_status_postcondition(backend_seselector.get_queue(), new_status);
    return status;
}
oneapi::mkl::vm::status clear_status(backend_selector<backend::mklcpu> selector) {
    // TO DO: add pre- and post conditions if needed
    // clear_status_precondition(queue);
    auto status = oneapi::mkl::vm::mklcpu::clear_status(selector.get_queue());
    // clear_status_postcondition(queue);
    return status;
}

cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    cl::sycl::buffer<std::complex<float>>& a,
                    cl::sycl::buffer<std::complex<float>>& b,
                    cl::sycl::buffer<std::complex<float>>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<float>> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    auto event = oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    cl::sycl::buffer<std::complex<double>>& a,
                    cl::sycl::buffer<std::complex<double>>& b,
                    cl::sycl::buffer<std::complex<double>>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<double>> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    auto event = oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    cl::sycl::buffer<float>& a, cl::sycl::buffer<float>& b,
                    cl::sycl::buffer<float>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<float> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    auto event = oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    cl::sycl::buffer<double>& a, cl::sycl::buffer<double>& b,
                    cl::sycl::buffer<double>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<double> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    auto event = oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, given_mode, eh);
    return event;
}

cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    std::complex<float>* a, std::complex<float>* b, std::complex<float>* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<float>> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    auto event =
        oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, depends, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n,
                    std::complex<double>* a, std::complex<double>* b, std::complex<double>* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<double>> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    auto event =
        oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, depends, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n, float* a, float* b,
                    float* y, cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    auto event =
        oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, depends, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    return event;
}
cl::sycl::event add(backend_selector<backend::mklcpu> selector, std::int64_t n, double* a,
                    double* b, double* y, cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
    // TO DO: add pre- and post conditions if needed
    // add_precondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    auto event =
        oneapi::mkl::vm::mklcpu::add(selector.get_queue(), n, a, b, y, depends, given_mode, eh);
    // add_postcondition(backend_seselector.get_queue(), n, a, b, y, depends, given_mode, eh);
    return event;
}

} //namespace vm
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_VL_DETAIL_MKLCPU_CT_HPP_
