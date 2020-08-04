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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device& dev, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
         oneapi::mkl::transpose transa, oneapi::mkl::diag unit_nonunit, int m, int n, int lda, int ldb,
         fp alpha) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during TRMM:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> A(ua), B(ua);
    if (left_right == oneapi::mkl::side::right)
        rand_matrix(A, transa, n, n, lda);
    else
        rand_matrix(A, transa, m, m, lda);

    rand_matrix(B, oneapi::mkl::transpose::nontrans, m, n, ldb);

    auto B_ref = B;

    // Call Reference TRMM.
    const int m_ref = m, n_ref = n;
    const int lda_ref = lda, ldb_ref = ldb;

    using fp_ref = typename ref_type_info<fp>::type;

    ::trmm(convert_to_cblas_side(left_right), convert_to_cblas_uplo(upper_lower),
           convert_to_cblas_trans(transa), convert_to_cblas_diag(unit_nonunit), &m_ref, &n_ref,
           (fp_ref*)&alpha, (fp_ref*)A.data(), &lda_ref, (fp_ref*)B_ref.data(), &ldb_ref);

    // Call DPC++ TRMM.

    try {
#ifdef CALL_RT_API
        done = oneapi::mkl::blas::trmm(main_queue, left_right, upper_lower, transa, unit_nonunit, m, n,
                                  alpha, A.data(), lda, B.data(), ldb, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::trmm,
                    (main_queue, left_right, upper_lower, transa, unit_nonunit, m, n, alpha,
                     A.data(), lda, B.data(), ldb, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during TRMM:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of TRMM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(B, B_ref, m, n, ldb, 10 * std::max(m, n), std::cout);

    return (int)good;
}

class TrmmUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(TrmmUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, 72, 27, 101, 102,
                                  alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, 72, 27, 101, 102,
                                  alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                  102, alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                  102, alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                  102, alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                  102, alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101, 102,
                                  alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                  oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101, 102,
                                  alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                  oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101, 102,
                                  alpha));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                  oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101, 102,
                                  alpha));
}
TEST_P(TrmmUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                   oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::lower,
                                   oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                   oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), oneapi::mkl::side::right, oneapi::mkl::uplo::upper,
                                   oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72, 27, 101,
                                   102, alpha));
}
TEST_P(TrmmUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::unit, 72,
                                                27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                oneapi::mkl::diag::unit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit,
                                                72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                                oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit,
                                                72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72,
                                                27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                                oneapi::mkl::transpose::trans, oneapi::mkl::diag::nonunit, 72,
                                                27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::lower,
                                                oneapi::mkl::transpose::conjtrans, oneapi::mkl::diag::nonunit,
                                                72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
                                                oneapi::mkl::transpose::conjtrans, oneapi::mkl::diag::nonunit,
                                                72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), oneapi::mkl::side::right,
                                                oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                                oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
}
TEST_P(TrmmUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::unit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::unit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::lower, oneapi::mkl::transpose::conjtrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::left,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), oneapi::mkl::side::right,
                                                 oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans,
                                                 oneapi::mkl::diag::nonunit, 72, 27, 101, 102, alpha));
}

INSTANTIATE_TEST_SUITE_P(TrmmUsmTestSuite, TrmmUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace