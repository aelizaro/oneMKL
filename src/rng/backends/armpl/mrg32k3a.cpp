/*******************************************************************************
* Copyright 2025 SiPearl
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

#include <iostream>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/rng/detail/engine_impl.hpp"
#include "oneapi/math/rng/detail/armpl/onemath_rng_armpl.hpp"

#include "armpl_common.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace armpl {

class mrg32k3a_impl : public oneapi::math::rng::detail::engine_impl {
public:
    mrg32k3a_impl(sycl::queue queue, std::uint32_t seed)
            : oneapi::math::rng::detail::engine_impl(queue) {
        vslNewStream(&stream_, VSL_BRNG_MRG32K3A, seed);
        state_size_ = vslGetStreamSize(stream_);
    }

    mrg32k3a_impl(sycl::queue queue, std::initializer_list<std::uint32_t> seed)
            : oneapi::math::rng::detail::engine_impl(queue) {
        vslNewStreamEx(&stream_, VSL_BRNG_MRG32K3A, 2 * seed.size(),
                       reinterpret_cast<const std::uint32_t*>(seed.begin()));
        state_size_ = vslGetStreamSize(stream_);
    }

    mrg32k3a_impl(const mrg32k3a_impl* other) : oneapi::math::rng::detail::engine_impl(*other) {
        vslCopyStream(&stream_, other->stream_);
        state_size_ = vslGetStreamSize(stream_);
    }

    // Buffers APIs

    virtual void generate(const uniform<float, uniform_method::standard>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                             acc_r.GET_MULTI_PTR, distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<double, uniform_method::standard>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                             acc_r.GET_MULTI_PTR, distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        if (distr.a() < 0)
            check_armpl_version(25, 04, 0,
                                "ArmPl : Uniform int32 generation is not functional with <0 bound");
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                             static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                             acc_r.GET_MULTI_PTR, distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<float, uniform_method::accurate>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
                             static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                             acc_r.GET_MULTI_PTR, distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const uniform<double, uniform_method::accurate>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
                             static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                             acc_r.GET_MULTI_PTR, distr.a(), distr.b());
            });
        });
    }

    virtual void generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                              static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                              acc_r.GET_MULTI_PTR, distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
                              static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                              acc_r.GET_MULTI_PTR, distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<float, gaussian_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                              static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                              acc_r.GET_MULTI_PTR, distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const gaussian<double, gaussian_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                              static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                              acc_r.GET_MULTI_PTR, distr.mean(), distr.stddev());
            });
        });
    }

    virtual void generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<float, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual void generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                          std::int64_t n, sycl::buffer<double, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual void generate(const lognormal<float, lognormal_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<float, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual void generate(const lognormal<double, lognormal_method::icdf>& distr, std::int64_t n,
                          sycl::buffer<double, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual void generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                               acc_r.GET_MULTI_PTR, distr.p());
            });
        });
    }

    virtual void generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                std::uint32_t* r_ptr = acc_r.GET_MULTI_PTR;
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF,
                               static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                               reinterpret_cast<std::int32_t*>(r_ptr), distr.p());
            });
        });
    }

    virtual void generate(const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::int32_t, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method poisson", "not yet implemented in ArmPL");
    }

    virtual void generate(const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr,
                          std::int64_t n, sycl::buffer<std::uint32_t, 1>& r) override {
        throw oneapi::math::unimplemented("rng", "method poisson", "not yet implemented in ArmPL");
    }

    virtual void generate(const bits<std::uint32_t>& distr, std::int64_t n,
                          sycl::buffer<std::uint32_t, 1>& r) override {
        sycl::buffer<char, 1> stream_buf(static_cast<char*>(stream_), state_size_);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_stream = stream_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_r = r.get_access<sycl::access::mode::read_write>(cgh);
            host_task<kernel_name<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD,
                                 static_cast<VSLStreamStatePtr>(acc_stream.GET_MULTI_PTR), n,
                                 acc_r.GET_MULTI_PTR);
            });
        });
    }

    // USM APIs

    virtual sycl::event generate(const uniform<float, uniform_method::standard>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual sycl::event generate(const uniform<double, uniform_method::standard>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual sycl::event generate(const uniform<std::int32_t, uniform_method::standard>& distr,
                                 std::int64_t n, std::int32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        if (distr.a() < 0)
            check_armpl_version(25, 04, 0,
                                "ArmPl : Uniform int32 generation is not functional with <0 bound");
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, distr.a(), distr.b());
            });
        });
    }

    virtual sycl::event generate(const uniform<float, uniform_method::accurate>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n, r, distr.a(),
                             distr.b());
            });
        });
    }

    virtual sycl::event generate(const uniform<double, uniform_method::accurate>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n, r, distr.a(),
                             distr.b());
            });
        });
    }

    virtual sycl::event generate(const gaussian<float, gaussian_method::box_muller2>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual sycl::event generate(const gaussian<double, gaussian_method::box_muller2>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual sycl::event generate(const gaussian<float, gaussian_method::icdf>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual sycl::event generate(const gaussian<double, gaussian_method::icdf>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, r, distr.mean(),
                              distr.stddev());
            });
        });
    }

    virtual sycl::event generate(const lognormal<float, lognormal_method::box_muller2>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(const lognormal<double, lognormal_method::box_muller2>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(const lognormal<float, lognormal_method::icdf>& distr,
                                 std::int64_t n, float* r,
                                 const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(const lognormal<double, lognormal_method::icdf>& distr,
                                 std::int64_t n, double* r,
                                 const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method lognormal",
                                          "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(const bernoulli<std::int32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::int32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, n, r, distr.p());
            });
        });
    }

    virtual sycl::event generate(const bernoulli<std::uint32_t, bernoulli_method::icdf>& distr,
                                 std::int64_t n, std::uint32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(cgh, [=]() {
                viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, n,
                               reinterpret_cast<int32_t*>(r), distr.p());
            });
        });
    }

    virtual sycl::event generate(
        const poisson<std::int32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::int32_t* r, const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method poisson", "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(
        const poisson<std::uint32_t, poisson_method::gaussian_icdf_based>& distr, std::int64_t n,
        std::uint32_t* r, const std::vector<sycl::event>& dependencies) override {
        throw oneapi::math::unimplemented("rng", "method poisson", "not yet implemented in ArmPL");
    }

    virtual sycl::event generate(const bits<std::uint32_t>& distr, std::int64_t n, std::uint32_t* r,
                                 const std::vector<sycl::event>& dependencies) override {
        sycl::event::wait_and_throw(dependencies);
        return queue_.submit([&](sycl::handler& cgh) {
            VSLStreamStatePtr stream = stream_;
            host_task<kernel_name_usm<mrg32k3a_impl, decltype(distr)>>(
                cgh, [=]() { viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, n, r); });
        });
    }

    virtual oneapi::math::rng::detail::engine_impl* copy_state() override {
        return new mrg32k3a_impl(this);
    }

    virtual void skip_ahead(std::uint64_t num_to_skip) override {
        vslSkipAheadStream(stream_, num_to_skip);
    }

    virtual void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) override {
        vslSkipAheadStreamEx(stream_, num_to_skip.size(), (unsigned long long*)num_to_skip.begin());
    }

    virtual void leapfrog(std::uint64_t idx, std::uint64_t stride) override {
        vslLeapfrogStream(stream_, idx, stride);
    }

    virtual ~mrg32k3a_impl() override {
        vslDeleteStream(&stream_);
    }

private:
    VSLStreamStatePtr stream_;
    std::int32_t state_size_;
};

oneapi::math::rng::detail::engine_impl* create_mrg32k3a(sycl::queue queue, std::uint32_t seed) {
    return new mrg32k3a_impl(queue, seed);
}

oneapi::math::rng::detail::engine_impl* create_mrg32k3a(sycl::queue queue,
                                                        std::initializer_list<std::uint32_t> seed) {
    return new mrg32k3a_impl(queue, seed);
}

} // namespace armpl
} // namespace rng
} // namespace math
} // namespace oneapi
