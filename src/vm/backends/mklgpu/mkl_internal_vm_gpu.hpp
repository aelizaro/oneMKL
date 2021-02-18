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

#ifndef _MKL_INTERNAL_VM_GPU_HPP_
#define _MKL_INTERNAL_VM_GPU_HPP_

#include <CL/sycl.hpp>

namespace oneapi {
namespace mkl {
namespace vm {
namespace detail {

using mode_result   = std::pair<bool, oneapi::mkl::vm::mode>;
using status_result = std::pair<bool, oneapi::mkl::vm::status>;

// static functions to access global status/modes

extern mode_result get_mode(sycl::queue & queue);
extern mode_result set_mode(sycl::queue & queue, oneapi::mkl::vm::mode new_mode);
extern status_result get_status(sycl::queue & queue);
extern status_result set_status(sycl::queue & queue, oneapi::mkl::vm::status new_status);
extern status_result clear_status(sycl::queue & queue);

// Yet used detailed entri points, BUT they contains dispatching
extern sycl::event add(sycl::queue & q, int64_t n, sycl::buffer<std::complex<float> > & a, sycl::buffer<std::complex<float> > & b, sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh);
extern sycl::event add(sycl::queue & q, int64_t n, sycl::buffer<std::complex<double> > & a, sycl::buffer<std::complex<double> > & b, sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh);
extern sycl::event add(sycl::queue & q, int64_t n, sycl::buffer<float> & a, sycl::buffer<float> & b, sycl::buffer<float> & y, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<float> eh);
extern sycl::event add(sycl::queue & q, int64_t n, sycl::buffer<double> & a, sycl::buffer<double> & b, sycl::buffer<double> & y, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<double> eh);
extern sycl::event add(sycl::queue & q, int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh);
extern sycl::event add(sycl::queue & q, int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh);
extern sycl::event add(sycl::queue & q, int64_t n, float * a, float * b, float * y, sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<float> eh);
extern sycl::event add(sycl::queue & q, int64_t n, double * a, double * b, double * y, sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode lib_mode, oneapi::mkl::vm::error_handler<double> eh);

} // namespace detail
} // namespace vm
} // namespace mkl
} // namespace oneapi

#endif //_MKL_INTERNAL_VM_GPU_HPP_
