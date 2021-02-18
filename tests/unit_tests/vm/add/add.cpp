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
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "oneapi/mkl/vm.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "test_helper.hpp"

#include "gtest/gtest.h"

#include "vm_common.hpp"

using namespace cl::sycl;
using std::vector;

extern std::vector<device*> devices;

namespace {

static constexpr int VLEN = 4;

static ulp_table_type ulp_table = {
    { MAX_HA_ULP_S, 4.5 }, { MAX_LA_ULP_S, 5.0 }, { MAX_EP_ULP_S, 5.0E3 },
    { MAX_HA_ULP_D, 2.0 }, { MAX_LA_ULP_D, 5.0 }, { MAX_EP_ULP_D, 7.0E7 },
    { MAX_HA_ULP_C, 2.0 }, { MAX_LA_ULP_C, 5.0 }, { MAX_EP_ULP_C, 5.0E3 },
    { MAX_HA_ULP_Z, 2.0 }, { MAX_LA_ULP_Z, 5.0 }, { MAX_EP_ULP_Z, 7.0E7 },
};

//!
//! @brief Accuracy test
//!
template <typename A, typename R>
int vAddAccuracyLiteTest(device* dev) {
    static constexpr int ACCURACY_LEN = VLEN;
    int argtype =

        ARG2_RES1;

    // *************************************************************
    // Data table declaraion
    // *************************************************************
    data_3_t data{
        .i = 0,

        .data_f32 =
            std::vector<data_3_f32_t>

        {

            { { UINT32_C(0x40D9B85C) },
              { UINT32_C(0xC007309A) },
              { UINT32_C(
                  0x4096200F) } }, //  0: vsAdd ( 6.80375481     , -2.1123414      ) = ( 4.6914134       );
            { { UINT32_C(0x40B52EFA) },
              { UINT32_C(0x40BF006A) },
              { UINT32_C(
                  0x413A17B2) } }, //  1: vsAdd ( 5.66198444     , 5.96880054      ) = ( 11.630785       );
            { { UINT32_C(0x4103BA28) },
              { UINT32_C(0xC0C1912F) },
              { UINT32_C(
                  0x400BC642) } }, //  2: vsAdd ( 8.2329483      , -6.04897261     ) = ( 2.1839757       );
            { { UINT32_C(0xC052EA36) },
              { UINT32_C(0x40ABAABC) },
              { UINT32_C(
                  0x40046B42) } }, //  3: vsAdd ( -3.2955451     , 5.3645916       ) = ( 2.0690465       );
        }

        ,
        .data_f64 =
            std::vector<data_3_f64_t>

        {

            { { UINT64_C(0x401B370B60E66E18) },
              { UINT64_C(0xC000E6134801CC26) },
              { UINT64_C(
                  0x4012C401BCE58805) } }, //  0: vdAdd ( 6.80375434309419092      , -2.11234146361813924      ) = ( 4.69141287947605168       );
            { { UINT64_C(0x4016A5DF421D4BBE) },
              { UINT64_C(0x4017E00D485FC01A) },
              { UINT64_C(
                  0x402742F6453E85EC) } }, //  1: vdAdd ( 5.66198447517211711      , 5.96880066952146571       ) = ( 11.6307851446935828       );
            { { UINT64_C(0x40207744D998EE8A) },
              { UINT64_C(0xC0183225E080644C) },
              { UINT64_C(
                  0x400178C7A562F190) } }, //  2: vdAdd ( 8.23294715873568705      , -6.04897261413232101      ) = ( 2.18397454460336604       );
            { { UINT64_C(0xC00A5D46A314BA8E) },
              { UINT64_C(0x4015755793FAEAB0) },
              { UINT64_C(
                  0x40008D6884E11AD2) } }, //  3: vdAdd ( -3.2955448857022196      , 5.36459189623808186       ) = ( 2.06904701053586226       );
        }

        ,

        .data_c32 =
            std::vector<data_3_c32_t>

        {

            { { UINT32_C(0xC007309A), UINT32_C(0x40D9B85C) },
              { UINT32_C(0x40BF006A), UINT32_C(0x40B52EFA) },
              { UINT32_C(0x4076D03A),
                UINT32_C(
                    0x414773AB) } }, //  0: vcAdd ( -2.1123414      + i * 6.80375481     , 5.96880054      + i * 5.66198444      ) = ( 3.85645914      + i * 12.4657393      );
            { { UINT32_C(0xC0C1912F), UINT32_C(0x4103BA28) },
              { UINT32_C(0x40ABAABC), UINT32_C(0xC052EA36) },
              { UINT32_C(0xBF2F3398),
                UINT32_C(
                    0x409DFF35) } }, //  1: vcAdd ( -6.04897261     + i * 8.2329483      , 5.3645916       + i * -3.2955451      ) = ( -0.684381008    + i * 4.9374032       );
            { { UINT32_C(0x3F8A29C0), UINT32_C(0xC08E3964) },
              { UINT32_C(0x4024F46C), UINT32_C(0xBEE77440) },
              { UINT32_C(0x406A094C),
                UINT32_C(
                    0xC09CB0A8) } }, //  2: vcAdd ( 1.07939911      + i * -4.44450569    , 2.57741833      + i * -0.452058792    ) = ( 3.65681744      + i * -4.89656448     );
            { { UINT32_C(0x3E8939C0), UINT32_C(0xC02D136C) },
              { UINT32_C(0x41052EB4), UINT32_C(0x4110B6A8) },
              { UINT32_C(0x41097882),
                UINT32_C(
                    0x40CAE39A) } }, //  3: vcAdd ( 0.268018723     + i * -2.70431042    , 8.32390213      + i * 9.04459381      ) = ( 8.59192085      + i * 6.34028339      );
        }

        ,
        .data_c64 =
            std::vector<data_3_c64_t>

        {

            { { UINT64_C(0xC000E6134801CC26), UINT64_C(0x401B370B60E66E18) },
              { UINT64_C(0x4017E00D485FC01A), UINT64_C(0x4016A5DF421D4BBE) },
              { UINT64_C(0x400EDA0748BDB40E),
                UINT64_C(
                    0x4028EE755181DCEB) } }, //  0: vzAdd ( -2.11234146361813924      + i * 6.80375434309419092      , 5.96880066952146571       + i * 5.66198447517211711       ) = ( 3.85645920590332647       + i * 12.465738818266308        );
            { { UINT64_C(0xC0183225E080644C), UINT64_C(0x40207744D998EE8A) },
              { UINT64_C(0x4015755793FAEAB0), UINT64_C(0xC00A5D46A314BA8E) },
              { UINT64_C(0xBFE5E672642BCCE0),
                UINT64_C(
                    0x4013BFE661A77FCD) } }, //  1: vzAdd ( -6.04897261413232101      + i * 8.23294715873568705      , 5.36459189623808186       + i * -3.2955448857022196       ) = ( -0.684380717894239154     + i * 4.93740227303346746       );
            { { UINT64_C(0x3FF1453801E28A70), UINT64_C(0xC011C72C86338E59) },
              { UINT64_C(0x40049E8D96893D1C), UINT64_C(0xBFDCEE88B739DD20) },
              { UINT64_C(0x400D4129977A8254),
                UINT64_C(
                    0xC013961511A72C2B) } }, //  2: vzAdd ( 1.07939911590861115       + i * -4.44450578393624429     , 2.57741849523848821       + i * -0.452058962756796134     ) = ( 3.65681761114709936       + i * -4.89656474669304043      );
            { { UINT64_C(0x3FD12735D3224E60), UINT64_C(0xC005A26D910B44DC) },
              { UINT64_C(0x4020A5D666294BAC), UINT64_C(0x402216D5173C2DAA) },
              { UINT64_C(0x40212F1014C25E1F),
                UINT64_C(
                    0x40195C7365F2B8E6) } }, //  3: vzAdd ( 0.268018203912310682      + i * -2.70431054416313366     , 8.32390136007401082       + i * 9.04459450349425609       ) = ( 8.59191956398632151       + i * 6.34028395933112243       );
        }

    };

    // *************************************************************
    // Variable declaraions
    // *************************************************************
    // Input arguments
    A arg1;
    std::vector<A> varg1;

    A arg2;

    std::vector<A> varg2;

    // Output results
    R ref1;
    std::vector<R> vref1;
    std::vector<R> vres1;

    // Number of errors
    int errs = 0;
    // Number of printed errors
    int printed_errs = 0;

    // *************************************************************
    // Vector input data initialization
    // *************************************************************
    for (int i = 0; i < ACCURACY_LEN; ++i) {
        // Getting values from reference data table

        data.get_values(arg1, arg2, ref1);

        // Pushing values into vectors
        varg1.push_back(arg1);
        vres1.push_back(777);
        vref1.push_back(ref1);

        varg2.push_back(arg2);
    } // for (int i = 0; i < ACCURACY_LEN; ++i)

    // Catch asynchronous exceptions
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
            }
        } // for (std::exception_ptr const& e : exceptions)
    };

    // Create execution queue with asynchronous error handling
    queue main_queue(*dev, exception_handler);

    // Get device name
    std::string dev_name = main_queue.get_device().get_info<info::device::name>();

    // *************************************************************
    // Loop by all 3 accuracy modes of VM: HA, LA, EP:
    // set computation mode, run VM and check results
    // *************************************************************
    for (int acc = 0; acc < ACCURACY_NUM; ++acc) {
        // Clear result vectors
        std::fill(vres1.begin(), vres1.end(), 777);

        // Create sycl buffers
        buffer<A, 1> in1(varg1.begin(), varg1.end());

        buffer<A, 1> in2(varg2.begin(), varg2.end());

        buffer<R, 1> out1(vres1.begin(), vres1.end());

        // Run VM function
        try {
#ifdef CALL_RT_API
            oneapi::mkl::vm::add(main_queue, varg1.size(), in1, in2, out1, accuracy_mode[acc]);
#else
            TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::vm::add, varg1.size(), in1, in2, out1,
                               accuracy_mode[acc]);
#endif

            // Get results from sycl buffers
            auto host_vres1 = out1.template get_access<access::mode::read>();

            for (int i = 0; i < vres1.size(); ++i) {
                vres1[i] = host_vres1[i];
            }

            main_queue.wait_and_throw();
        }
        catch (const oneapi::mkl::unimplemented& e) {
            return test_skipped;
        }
        catch (exception& e) {
            std::cerr << "SYCL exception during Accuracy Test\n"
                      << e.what() << std::endl
                      << "OpenCl status: " << e.get_cl_code() << std::endl;
            return test_failed;
        }

        // *************************************************************
        // Compute ulp between computed and expected (reference)
        // values and check
        // *************************************************************
        for (int i = 0; i < ACCURACY_LEN; ++i) {
            // Check simple indexing function
            errs +=
                check_result<A, R>("", "add", ",simple", i, argtype, acc, varg1[i], varg1[i],
                                   vres1[i], vres1[i], vref1[i], vref1[i], ulp_table, false, false);

        } // for (int i = 0; i < ACCURACY_LEN; ++i)
    } // for (int acc = 0; acc < ACCURACY_NUM; ++acc)

    std::cout << "\tResult: " << ((errs == 0) ? "PASS" : "FAIL") << std::endl;

    return (errs == 0) ? test_passed : test_failed;
}

// Wrapper to vAddAccuracyLiteTest<float, float>
int vAddSinglePrecisionAccuracyLiteTest(device* dev) {
    return vAddAccuracyLiteTest<float, float>(dev);
}

// Wrapper to vAddAccuracyLiteTest<double, double>
int vAddDoublePrecisionAccuracyLiteTest(device* dev) {
    return vAddAccuracyLiteTest<double, double>(dev);
}

// Wrapper to vAddAccuracyLiteTest<std::complex<float>, std::complex<float>>
int vAddComplexSinglePrecisionAccuracyLiteTest(device* dev) {
    return vAddAccuracyLiteTest<std::complex<float>, std::complex<float>>(dev);
}

// Wrapper to vAddAccuracyLiteTest<std::complex<double>, std::complex<double>>
int vAddComplexDoublePrecisionAccuracyLiteTest(device* dev) {
    return vAddAccuracyLiteTest<std::complex<double>, std::complex<double>>(dev);
}

class AddTests : public ::testing::TestWithParam<cl::sycl::device*> {};

TEST_P(AddTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(::vAddSinglePrecisionAccuracyLiteTest(GetParam()));
}

TEST_P(AddTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(::vAddDoublePrecisionAccuracyLiteTest(GetParam()));
}

TEST_P(AddTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(::vAddComplexSinglePrecisionAccuracyLiteTest(GetParam()));
}

TEST_P(AddTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(::vAddComplexDoublePrecisionAccuracyLiteTest(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(AddTestsuite, AddTests, ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
