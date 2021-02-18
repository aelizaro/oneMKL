/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _VM_CPU_COMMON_HPP_
#define _VM_CPU_COMMON_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/vm/types.hpp"
#include "mkl_vml.h"

namespace oneapi {
namespace mkl {
namespace vm {
namespace mklcpu {

// host_task automatically uses run_on_host_intel if it is supported by the
//  compiler. Otherwise, it falls back to single_task.
template <typename K, typename H, typename F>
static inline auto host_task_internal(H &cgh, F f, int) -> decltype(cgh.run_on_host_intel(f)) {
    return cgh.run_on_host_intel(f);
}

template <typename K, typename H, typename F>
static inline void host_task_internal(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
static inline void host_task(H &cgh, F f) {
    (void)host_task_internal<K>(cgh, f, 0);
}

static inline std::int64_t get_classic_mode(oneapi::mkl::vm::mode sycl_mode) {
    MKL_INT64 mode = VML_HA;

    switch (sycl_mode) {
        case oneapi::mkl::vm::mode::ep: mode = VML_EP; break;
        case oneapi::mkl::vm::mode::la: mode = VML_LA; break;
        case oneapi::mkl::vm::mode::ha: mode = VML_HA; break;
        case oneapi::mkl::vm::mode::not_defined:
        default: mode = VML_HA; break;
    }
    return mode;
}

} // namespace mklcpu
} // namespace vm
} // namespace mkl
} // namespace oneapi

#endif //_VM_CPU_COMMON_HPP_
