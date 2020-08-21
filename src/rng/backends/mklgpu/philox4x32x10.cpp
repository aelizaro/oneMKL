/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <iostream>
#include <CL/sycl.hpp>

#include "mkl_rng/engines.hpp"

#include "oneapi/mkl/detail/exceptions.hpp"
#include "oneapi/mkl/rng/detail/engine_impl.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace mklgpu {

class philox4x32x10_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    philox4x32x10_impl(cl::sycl::queue queue, std::uint64_t seed)
            : oneapi::mkl::rng::detail::engine_impl(queue),
              engine_(queue, seed) {}

    philox4x32x10_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : oneapi::mkl::rng::detail::engine_impl(queue),
              engine_(queue, seed) {}

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : oneapi::mkl::rng::detail::engine_impl(*other),
              engine_(other->engine_) {}

    virtual void generate(
        const oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<float, 1> r) override {
        oneapi::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const oneapi::mkl::rng::uniform<double, oneapi::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<double, 1> r) override {
        oneapi::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t, 1> r) override {
        oneapi::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        oneapi::mkl::rng::skip_ahead(engine_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        oneapi::mkl::rng::skip_ahead(engine_, num_to_skip);
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::InvalidArgumentsException(
            "leapfrog is not supported for philox4x32x10 engine");
    }

    virtual ~philox4x32x10_impl() override {}

private:
    oneapi::mkl::rng::philox4x32x10 engine_;
};

oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(sycl::queue queue, std::uint64_t seed) {
    return new philox4x32x10_impl(queue, seed);
}

oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(
    cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed) {
    return new philox4x32x10_impl(queue, seed);
}

oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(
    const oneapi::mkl::rng::detail::engine_impl& other) {
    return new philox4x32x10_impl(reinterpret_cast<const philox4x32x10_impl*>(&other));
}

} // namespace mklgpu
} // namespace rng
} // namespace mkl
} // namespace oneapi