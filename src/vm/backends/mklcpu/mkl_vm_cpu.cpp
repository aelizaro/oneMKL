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

#include <CL/sycl.hpp>

#include "mkl_vml.h"

#include "cpu_common.hpp"
#include "oneapi/mkl/vm/detail/mklcpu/onemkl_vm_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace vm {
namespace mklcpu {

oneapi::mkl::vm::mode get_mode(cl::sycl::queue & queue) {
    // TO DO
    return oneapi::mkl::vm::mode::not_defined;
}
oneapi::mkl::vm::mode set_mode(cl::sycl::queue & queue, oneapi::mkl::vm::mode new_mode) {
    // TO DO
    return oneapi::mkl::vm::mode::not_defined;
}
oneapi::mkl::vm::status get_status(cl::sycl::queue & queue) {
    // TO DO
    return oneapi::mkl::vm::status::not_defined;
}
oneapi::mkl::vm::status set_status(cl::sycl::queue & queue, oneapi::mkl::vm::status new_status) {
    // TO DO
    return oneapi::mkl::vm::status::not_defined;
}
oneapi::mkl::vm::status clear_status(cl::sycl::queue & queue) {
    // TO DO
    return oneapi::mkl::vm::status::not_defined;
}

cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<float> > & a, cl::sycl::buffer<std::complex<float> > & b, cl::sycl::buffer<std::complex<float> > & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh) {
    // TO DO add error handler staff and checks if a, b, y are same
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class mkl_kernel_cadd>(cgh, [=]() {
            MKL_Complex8* a_ptr = reinterpret_cast<MKL_Complex8*>(accessor_a.get_pointer().get());
            MKL_Complex8* b_ptr = reinterpret_cast<MKL_Complex8*>(accessor_b.get_pointer().get());
            MKL_Complex8* y_ptr = reinterpret_cast<MKL_Complex8*>(accessor_y.get_pointer().get());
            ::vmcAdd(n, a_ptr, b_ptr, y_ptr, classic_mode);
        });
    });
    return event;
}
cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<std::complex<double> > & a, cl::sycl::buffer<std::complex<double> > & b, cl::sycl::buffer<std::complex<double> > & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh) {
    // TO DO add error handler staff and checks if a, b, y are same
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class mkl_kernel_zadd>(cgh, [=]() {
            MKL_Complex16* a_ptr = reinterpret_cast<MKL_Complex16*>(accessor_a.get_pointer().get());
            MKL_Complex16* b_ptr = reinterpret_cast<MKL_Complex16*>(accessor_b.get_pointer().get());
            MKL_Complex16* y_ptr = reinterpret_cast<MKL_Complex16*>(accessor_y.get_pointer().get());
            ::vmzAdd(n, a_ptr, b_ptr, y_ptr, classic_mode);
        });
    });
    return event;
}
cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<float> & a, cl::sycl::buffer<float> & b, cl::sycl::buffer<float> & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    // TO DO add error handler staff and checks if a, b, y are same
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class mkl_kernel_sadd>(cgh, [=]() {
            ::vmsAdd(n, accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_y.get_pointer(), classic_mode);
        });
    });
    return event;
}
cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, cl::sycl::buffer<double> & a, cl::sycl::buffer<double> & b, cl::sycl::buffer<double> & y, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
    // TO DO add error handler staff and checks if a, b, y are same
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class mkl_kernel_dadd>(cgh, [=]() {
            ::vmdAdd(n, accessor_a.get_pointer(), accessor_b.get_pointer(), accessor_y.get_pointer(), classic_mode);
        });
    });
    return event;
}

cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<float> * a, std::complex<float> * b, std::complex<float> * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<float> > eh) {
    // TO DO add error handler staff
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        host_task<class mkl_kernel_cadd_usm>(cgh, [=]() {
            ::vmcAdd(n, reinterpret_cast<MKL_Complex8*>(a), reinterpret_cast<MKL_Complex8*>(b), reinterpret_cast<MKL_Complex8*>(y), classic_mode);
        });
    });
    return event;
}
cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, std::complex<double> * a, std::complex<double> * b, std::complex<double> * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<std::complex<double> > eh) {
    // TO DO add error handler staff
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        host_task<class mkl_kernel_cadd_usm>(cgh, [=]() {
            ::vmzAdd(n, reinterpret_cast<MKL_Complex16*>(a), reinterpret_cast<MKL_Complex16*>(b), reinterpret_cast<MKL_Complex16*>(y), classic_mode);
        });
    });
    return event;
}
cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, float * a, float * b, float * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    // TO DO add error handler staff
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        host_task<class mkl_kernel_sadd_usm>(cgh, [=]() {
            ::vmsAdd(n, a, b, y, classic_mode);
        });
    });
    return event;
}

cl::sycl::event add(cl::sycl::queue & q, std::int64_t n, double * a, double * b, double * y, cl::sycl::vector_class<cl::sycl::event> const & depends, oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
    // TO DO add error handler staff
    auto classic_mode = get_classic_mode(given_mode);
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        host_task<class mkl_kernel_dadd_usm>(cgh, [=]() {
            ::vmdAdd(n, a, b, y, classic_mode);
        });
    });
    return event;
}

} // namespace mklcpu
} // namespace vm
} // namespace mkl
} // namespace oneapi