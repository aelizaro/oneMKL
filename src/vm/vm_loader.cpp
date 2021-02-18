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

#include "oneapi/mkl/vm/detail/vm_loader.hpp"

#include "function_table_initializer.hpp"
#include "vm/function_table.hpp"

namespace oneapi {
namespace mkl {
namespace vm {
namespace detail {

static oneapi::mkl::detail::table_initializer<domain::vm, vm_function_table_t> function_tables;

oneapi::mkl::vm::mode get_mode(oneapi::mkl::device libkey, cl::sycl::queue & queue) {
    return function_tables[libkey].get_mode_sycl(queue);
}
oneapi::mkl::vm::mode set_mode(oneapi::mkl::device libkey, cl::sycl::queue & queue, oneapi::mkl::vm::mode new_mode) {
    return function_tables[libkey].set_mode_sycl(queue, new_mode);
}

oneapi::mkl::vm::status get_status(oneapi::mkl::device libkey, cl::sycl::queue & queue) {
    return function_tables[libkey].get_status_sycl(queue);
}
oneapi::mkl::vm::status set_status(oneapi::mkl::device libkey, cl::sycl::queue & queue, oneapi::mkl::vm::status new_status) {
    return function_tables[libkey].set_status_sycl(queue, new_status);
}
oneapi::mkl::vm::status clear_status(oneapi::mkl::device libkey, cl::sycl::queue & queue) {
    return function_tables[libkey].clear_status_sycl(queue);
}

cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh) {
    return function_tables[libkey].cadd_sycl(q, n, a, b, y, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh) {
    return function_tables[libkey].zadd_sycl(q, n, a, b, y, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    return function_tables[libkey].sadd_sycl(q, n, a, b, y, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
    return function_tables[libkey].dadd_sycl(q, n, a, b, y, given_mode, eh);
}

cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh) {
    return function_tables[libkey].cadd_usm_sycl(q, n, a, b, y, depends, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh) {
    return function_tables[libkey].zadd_usm_sycl(q, n, a, b, y, depends, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    return function_tables[libkey].sadd_usm_sycl(q, n, a, b, y, depends, given_mode, eh);
}
cl::sycl::event add(oneapi::mkl::device libkey, cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
    return function_tables[libkey].dadd_usm_sycl(q, n, a, b, y, depends, given_mode, eh);
}

} //namespace detail
} //namespace vm
} //namespace mkl
} //namespace oneapi
