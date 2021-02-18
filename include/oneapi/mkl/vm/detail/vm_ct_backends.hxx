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

static inline oneapi::mkl::vm::mode get_mode(backend_selector<backend::BACKEND> selector);
static inline oneapi::mkl::vm::mode set_mode(backend_selector<backend::BACKEND> selector,
                                             oneapi::mkl::vm::mode new_mode);

static inline oneapi::mkl::vm::status get_status(backend_selector<backend::BACKEND> selector);
static inline oneapi::mkl::vm::status set_status(backend_selector<backend::BACKEND> selector,
                                                 oneapi::mkl::vm::status new_status);
static inline oneapi::mkl::vm::status clear_status(backend_selector<backend::BACKEND> selector);

static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n,
    cl::sycl::buffer<std::complex<float>>& a, cl::sycl::buffer<std::complex<float>>& b,
    cl::sycl::buffer<std::complex<float>>& y,
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<std::complex<float>> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n,
    cl::sycl::buffer<std::complex<double>>& a, cl::sycl::buffer<std::complex<double>>& b,
    cl::sycl::buffer<std::complex<double>>& y,
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<std::complex<double>> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, cl::sycl::buffer<float>& a,
    cl::sycl::buffer<float>& b, cl::sycl::buffer<float>& y,
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<float> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, cl::sycl::buffer<double>& a,
    cl::sycl::buffer<double>& b, cl::sycl::buffer<double>& y,
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<double> eh = {});

static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<float>* a,
    std::complex<float>* b, std::complex<float>* y,
    cl::sycl::vector_class<cl::sycl::event> const& depends = {},
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<std::complex<float>> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, std::complex<double>* a,
    std::complex<double>* b, std::complex<double>* y,
    cl::sycl::vector_class<cl::sycl::event> const& depends = {},
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<std::complex<double>> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, float* a, float* b, float* y,
    cl::sycl::vector_class<cl::sycl::event> const& depends = {},
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<float> eh = {});
static inline cl::sycl::event add(
    backend_selector<backend::BACKEND> selector, std::int64_t n, double* a, double* b, double* y,
    cl::sycl::vector_class<cl::sycl::event> const& depends = {},
    oneapi::mkl::vm::mode given_mode = oneapi::mkl::vm::mode::not_defined,
    oneapi::mkl::vm::error_handler<double> eh = {});
