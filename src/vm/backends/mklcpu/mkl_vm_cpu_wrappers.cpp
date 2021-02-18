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

#include "vm/function_table.hpp"
#include "oneapi/mkl/vm/detail/mklcpu/onemkl_vm_mklcpu.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT vm_function_table_t mkl_vm_table = { WRAPPER_VERSION,
                                                              oneapi::mkl::vm::mklcpu::get_mode,
                                                              oneapi::mkl::vm::mklcpu::set_mode,
                                                              oneapi::mkl::vm::mklcpu::get_status,
                                                              oneapi::mkl::vm::mklcpu::set_status,
                                                              oneapi::mkl::vm::mklcpu::clear_status,

                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add,

                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add,
                                                              oneapi::mkl::vm::mklcpu::add };
