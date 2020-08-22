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

namespace oneapi {}
using namespace oneapi;

namespace oneapi {
namespace mkl {
namespace rng {
namespace mklgpu {

class philox4x32x10_impl : public oneapi::mkl::rng::detail::engine_impl {
public:
    philox4x32x10_impl(cl::sycl::queue queue, std::uint64_t seed)
            : ::mkl::rng::detail::engine_impl(queue),
              engine_(queue, seed) {}

    philox4x32x10_impl(cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed)
            : ::mkl::rng::detail::engine_impl(queue),
              engine_(queue, seed) {}

    philox4x32x10_impl(const philox4x32x10_impl* other)
            : ::mkl::rng::detail::engine_impl(*other),
              engine_(other->engine_) {}

    // Buffers API

    virtual void generate(
        const ::mkl::rng::uniform<float, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<float, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::uniform<double, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<double, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::uniform<std::int32_t, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, sycl::buffer<std::int32_t, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::uniform<float, ::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, sycl::buffer<float, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::uniform<double, ::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, sycl::buffer<double, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::gaussian<float, ::mkl::rng::gaussian_method::box_muller2>& distr,
        std::int64_t n, sycl::buffer<float, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::gaussian<double, ::mkl::rng::gaussian_method::box_muller2>& distr,
        std::int64_t n, sycl::buffer<double, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::gaussian<float, ::mkl::rng::gaussian_method::icdf>& distr, std::int64_t n,
        sycl::buffer<float, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(
        const ::mkl::rng::gaussian<double, ::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, sycl::buffer<double, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          cl::sycl::buffer<std::uint32_t, 1> r) override {
        ::mkl::rng::generate(distr, engine_, n, r);
    }

    // USM APIs

    virtual cl::sycl::event generate(
        const ::mkl::rng::uniform<float, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::uniform<double, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::uniform<std::int32_t, ::mkl::rng::uniform_method::standard>& distr,
        std::int64_t n, std::int32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::uniform<float, ::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::uniform<double, ::mkl::rng::uniform_method::accurate>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::gaussian<float, ::mkl::rng::gaussian_method::box_muller2>& distr,
        std::int64_t n, float* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::gaussian<double, ::mkl::rng::gaussian_method::box_muller2>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::gaussian<float, ::mkl::rng::gaussian_method::icdf>& distr, std::int64_t n,
        float* r, const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const ::mkl::rng::gaussian<double, ::mkl::rng::gaussian_method::icdf>& distr,
        std::int64_t n, double* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual cl::sycl::event generate(
        const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
        const cl::sycl::vector_class<cl::sycl::event>& dependencies) override {
        return ::mkl::rng::generate(distr, engine_, n, r, dependencies);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        ::mkl::rng::skip_ahead(engine_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        ::mkl::rng::skip_ahead(engine_, num_to_skip);
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        throw oneapi::mkl::InvalidArgumentsException(
            "leapfrog is not supported for philox4x32x10 engine");
    }

    virtual ~philox4x32x10_impl() override {}

private:
    ::mkl::rng::philox4x32x10 engine_;
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