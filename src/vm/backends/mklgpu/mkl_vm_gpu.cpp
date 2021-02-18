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

#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/vm/detail/mklgpu/onemkl_vm_mklgpu.hpp"
#include "mkl_internal_vm_gpu.hpp"

namespace oneapi {
namespace mkl {
namespace vm {
namespace mklgpu {

oneapi::mkl::vm::mode get_mode(cl::sycl::queue& queue) {
    oneapi::mkl::vm::detail::mode_result ret = oneapi::mkl::vm::detail::get_mode(queue);
    return ret.second;
}
oneapi::mkl::vm::mode set_mode(cl::sycl::queue& queue, oneapi::mkl::vm::mode new_mode) {
    oneapi::mkl::vm::detail::mode_result ret = oneapi::mkl::vm::detail::set_mode(queue, new_mode);
    return ret.second;
}
oneapi::mkl::vm::status get_status(cl::sycl::queue& queue) {
    oneapi::mkl::vm::detail::status_result ret = oneapi::mkl::vm::detail::get_status(queue);
    return ret.second;
}
oneapi::mkl::vm::status set_status(cl::sycl::queue& queue, oneapi::mkl::vm::status new_status) {
    oneapi::mkl::vm::detail::status_result ret =
        oneapi::mkl::vm::detail::set_status(queue, new_status);
    return ret.second;
}
oneapi::mkl::vm::status clear_status(cl::sycl::queue& queue) {
    oneapi::mkl::vm::detail::status_result ret =
        oneapi::mkl::vm::detail::set_status(queue, vm::status::success);
    return ret.second;
}

cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, cl::sycl::buffer<std::complex<float>>& a,
                    cl::sycl::buffer<std::complex<float>>& b,
                    cl::sycl::buffer<std::complex<float>>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<float>> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}
cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, cl::sycl::buffer<std::complex<double>>& a,
                    cl::sycl::buffer<std::complex<double>>& b,
                    cl::sycl::buffer<std::complex<double>>& y, oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<double>> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}
cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, cl::sycl::buffer<float>& a,
                    cl::sycl::buffer<float>& b, cl::sycl::buffer<float>& y,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, given_mode, eh);
}
cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, cl::sycl::buffer<double>& a,
                    cl::sycl::buffer<double>& b, cl::sycl::buffer<double>& y,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}

cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, std::complex<float>* a,
                    std::complex<float>* b, std::complex<float>* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<float>> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, depends, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}
cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, std::complex<double>* a,
                    std::complex<double>* b, std::complex<double>* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode,
                    oneapi::mkl::vm::error_handler<std::complex<double>> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, depends, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}
cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, float* a, float* b, float* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<float> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, depends, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}

cl::sycl::event add(cl::sycl::queue& q, std::int64_t n, double* a, double* b, double* y,
                    cl::sycl::vector_class<cl::sycl::event> const& depends,
                    oneapi::mkl::vm::mode given_mode, oneapi::mkl::vm::error_handler<double> eh) {
#if !defined(_WIN64)
    return oneapi::mkl::vm::detail::add(q, n, a, b, y, depends, given_mode, eh);
#else
    throw oneapi::mkl::unimplemented("vm", "add");
    return cl::sycl::event{};
#endif
}

} // namespace mklgpu
} // namespace vm
} // namespace mkl
} // namespace oneapi