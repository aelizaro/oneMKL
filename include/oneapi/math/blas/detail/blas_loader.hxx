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

// Buffer APIs

ONEMATH_EXPORT void herk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, float alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, float beta,
                         sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void herk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, double alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, double beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         float alpha, sycl::buffer<float, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         double alpha, sycl::buffer<double, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         float alpha, sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         double alpha, sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, sycl::buffer<float, 1>& a,
                         std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, sycl::buffer<float, 1>& a,
                         sycl::buffer<float, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& a,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& a,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void spr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                        sycl::buffer<float, 1>& a);
ONEMATH_EXPORT void spr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, double alpha, sycl::buffer<double, 1>& x, std::int64_t incx,
                        sycl::buffer<double, 1>& a);

ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<float, 1>& b, std::int64_t ldb,
                               std::int64_t stride_b, float beta, sycl::buffer<float, 1>& c,
                               std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<double, 1>& b, std::int64_t ldb,
                               std::int64_t stride_b, double beta, sycl::buffer<double, 1>& c,
                               std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                               std::int64_t stride_b, std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               sycl::half alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, sycl::half beta,
                               sycl::buffer<sycl::half, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               float alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, float beta,
                               sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                               std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               float alpha, sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, float beta,
                               sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                               std::int64_t batch_size);
ONEMATH_EXPORT void gemm_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                               transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                               float alpha, sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, float beta,
                               sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);

ONEMATH_EXPORT void syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, float alpha,
                         sycl::buffer<float, 1>& a, std::int64_t lda, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, double alpha,
                         sycl::buffer<double, 1>& a, std::int64_t lda, double beta,
                         sycl::buffer<double, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                         std::int64_t ldc);
ONEMATH_EXPORT void syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                         std::int64_t lda, std::complex<double> beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void syrk_batch(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               transpose trans, std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                               float beta, sycl::buffer<float, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void syrk_batch(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               transpose trans, std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stride_a,
                               double beta, sycl::buffer<double, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void syrk_batch(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               transpose trans, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                               std::int64_t lda, std::int64_t stride_a, std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void syrk_batch(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               transpose trans, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                               std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                               std::int64_t stride_c, std::int64_t batch_size);

ONEMATH_EXPORT void her2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);
ONEMATH_EXPORT void her2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

ONEMATH_EXPORT void hbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void hbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::int64_t k, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy);

ONEMATH_EXPORT void rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, float c,
                        float s);
ONEMATH_EXPORT void rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, double c,
                        double s);
ONEMATH_EXPORT void rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                        std::int64_t incy, float c, float s);
ONEMATH_EXPORT void rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                        std::int64_t incy, double c, double s);

ONEMATH_EXPORT void axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         double alpha, sycl::buffer<double, 1>& x, std::int64_t incx,
                         sycl::buffer<double, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                         std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                         std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy);

ONEMATH_EXPORT void axpy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                               std::int64_t stridex, sycl::buffer<float, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);
ONEMATH_EXPORT void axpy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               double alpha, sycl::buffer<double, 1>& x, std::int64_t incx,
                               std::int64_t stridex, sycl::buffer<double, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);
ONEMATH_EXPORT void axpy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                               std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);
ONEMATH_EXPORT void axpy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                               std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          float alpha, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                          sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          double alpha, sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                          sycl::buffer<double, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                          std::int64_t incx, std::complex<float> beta,
                          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                          std::int64_t incx, std::complex<double> beta,
                          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void gerc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);
ONEMATH_EXPORT void gerc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

ONEMATH_EXPORT void syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k, float alpha,
                          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
                          std::int64_t ldb, float beta, sycl::buffer<float, 1>& c,
                          std::int64_t ldc);
ONEMATH_EXPORT void syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k, double alpha,
                          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
                          std::int64_t ldb, double beta, sycl::buffer<double, 1>& c,
                          std::int64_t ldc);
ONEMATH_EXPORT void syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k,
                          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                          std::int64_t ldb, std::complex<float> beta,
                          sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k,
                          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                          std::int64_t ldb, std::complex<double> beta,
                          sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
                         std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
                         std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx,
                         double beta, sycl::buffer<double, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy);

ONEMATH_EXPORT void gemv_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                               std::int64_t m, std::int64_t n, float alpha,
                               sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                               sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                               float beta, sycl::buffer<float, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void gemv_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                               std::int64_t m, std::int64_t n, double alpha,
                               sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                               sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                               double beta, sycl::buffer<double, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void gemv_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                               std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& x,
                               std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void gemv_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                               std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& x,
                               std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                               std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void dgmm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               std::int64_t m, std::int64_t n, sycl::buffer<float, 1>& a,
                               std::int64_t lda, std::int64_t stridea, sycl::buffer<float, 1>& x,
                               std::int64_t incx, std::int64_t stridex, sycl::buffer<float, 1>& c,
                               std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size);

ONEMATH_EXPORT void dgmm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               std::int64_t m, std::int64_t n, sycl::buffer<double, 1>& a,
                               std::int64_t lda, std::int64_t stridea, sycl::buffer<double, 1>& x,
                               std::int64_t incx, std::int64_t stridex, sycl::buffer<double, 1>& c,
                               std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size);

ONEMATH_EXPORT void dgmm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                               std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& x,
                               std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                               std::int64_t stridec, std::int64_t batch_size);

ONEMATH_EXPORT void dgmm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               std::int64_t m, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                               std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& x,
                               std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                               std::int64_t stridec, std::int64_t batch_size);

ONEMATH_EXPORT void her(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda);
ONEMATH_EXPORT void her(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, double alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda);

ONEMATH_EXPORT void hpr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& a);
ONEMATH_EXPORT void hpr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, double alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& a);

ONEMATH_EXPORT void gemm_bias(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                              transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<int8_t, 1>& a,
                              std::int64_t lda, int8_t ao, sycl::buffer<uint8_t, 1>& b,
                              std::int64_t ldb, uint8_t bo, float beta, sycl::buffer<int32_t, 1>& c,
                              std::int64_t ldc, sycl::buffer<int32_t, 1>& co);
ONEMATH_EXPORT void gemm_bias(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                              transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<int8_t, 1>& a,
                              std::int64_t lda, int8_t ao, sycl::buffer<int8_t, 1>& b,
                              std::int64_t ldb, int8_t bo, float beta, sycl::buffer<int32_t, 1>& c,
                              std::int64_t ldc, sycl::buffer<int32_t, 1>& co);
ONEMATH_EXPORT void gemm_bias(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                              transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a,
                              std::int64_t lda, uint8_t ao, sycl::buffer<int8_t, 1>& b,
                              std::int64_t ldb, int8_t bo, float beta, sycl::buffer<int32_t, 1>& c,
                              std::int64_t ldc, sycl::buffer<int32_t, 1>& co);
ONEMATH_EXPORT void gemm_bias(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                              transpose transb, offset offsetc, std::int64_t m, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a,
                              std::int64_t lda, uint8_t ao, sycl::buffer<uint8_t, 1>& b,
                              std::int64_t ldb, uint8_t bo, float beta, sycl::buffer<int32_t, 1>& c,
                              std::int64_t ldc, sycl::buffer<int32_t, 1>& co);

ONEMATH_EXPORT void iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<float, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<double, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);

ONEMATH_EXPORT void hpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void hpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy);

ONEMATH_EXPORT void spmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
                         sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void spmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
                         sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                         sycl::buffer<double, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void rotmg(oneapi::math::device libkey, sycl::queue& queue,
                          sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
                          sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param);
ONEMATH_EXPORT void rotmg(oneapi::math::device libkey, sycl::queue& queue,
                          sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
                          sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param);

ONEMATH_EXPORT void swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void geru(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda);
ONEMATH_EXPORT void geru(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda);

ONEMATH_EXPORT void nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& result);
ONEMATH_EXPORT void nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<double, 1>& result);
ONEMATH_EXPORT void nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<float, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& result);
ONEMATH_EXPORT void nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<double, 1>& x, std::int64_t incx,
                         sycl::buffer<double, 1>& result);

ONEMATH_EXPORT void gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                          float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                          sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                          sycl::buffer<float, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                          double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                          sycl::buffer<double, 1>& b, std::int64_t ldb, double beta,
                          sycl::buffer<double, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                          std::int64_t ldb, std::complex<float> beta,
                          sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                          std::int64_t ldb, std::complex<double> beta,
                          sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& b, std::int64_t ldb, double beta,
                         sycl::buffer<double, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                         std::int64_t ldb, std::complex<float> beta,
                         sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                         std::int64_t ldb, std::complex<double> beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::half alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                         sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, sycl::half beta,
                         sycl::buffer<sycl::half, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         float alpha, sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                         sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                         transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                         float alpha, sycl::buffer<bfloat16, 1>& a, std::int64_t lda,
                         sycl::buffer<bfloat16, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void syr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& a,
                         std::int64_t lda);
ONEMATH_EXPORT void syr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
                         std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
                         sycl::buffer<double, 1>& a, std::int64_t lda);

ONEMATH_EXPORT void ger(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                        sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& a,
                        std::int64_t lda);
ONEMATH_EXPORT void ger(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, double alpha, sycl::buffer<double, 1>& x, std::int64_t incx,
                        sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& a,
                        std::int64_t lda);

ONEMATH_EXPORT void trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

ONEMATH_EXPORT void dotu(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& result);
ONEMATH_EXPORT void dotu(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& result);

ONEMATH_EXPORT void hemm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                         std::int64_t ldb, std::complex<float> beta,
                         sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void hemm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                         std::int64_t ldb, std::complex<double> beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void hpr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& a);
ONEMATH_EXPORT void hpr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& a);

ONEMATH_EXPORT void gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                         float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                         double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                         sycl::buffer<double, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                         std::int64_t incx, std::complex<float> beta,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                         std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                         std::int64_t incx, std::complex<double> beta,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                         sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
                         std::int64_t ldb, float beta, sycl::buffer<float, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                         sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
                         std::int64_t ldb, double beta, sycl::buffer<double, 1>& c,
                         std::int64_t ldc);
ONEMATH_EXPORT void symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n,
                         std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                         std::int64_t ldb, std::complex<float> beta,
                         sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, std::int64_t m, std::int64_t n,
                         std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                         std::int64_t ldb, std::complex<double> beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

ONEMATH_EXPORT void dotc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<float>, 1>& result);
ONEMATH_EXPORT void dotc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                         sycl::buffer<std::complex<double>, 1>& result);

ONEMATH_EXPORT void syr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                        sycl::buffer<float, 1>& a, std::int64_t lda);
ONEMATH_EXPORT void syr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                        std::int64_t n, double alpha, sycl::buffer<double, 1>& x, std::int64_t incx,
                        sycl::buffer<double, 1>& a, std::int64_t lda);

ONEMATH_EXPORT void trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                         uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

ONEMATH_EXPORT void symv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void symv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                         sycl::buffer<double, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, sycl::buffer<float, 1>& a,
                         sycl::buffer<float, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& a,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& a,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, sycl::buffer<float, 1>& a,
                         std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void copy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<float, 1>& y, std::int64_t incy, std::int64_t stridey,
                               std::int64_t batch_size);

ONEMATH_EXPORT void copy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                               sycl::buffer<double, 1>& y, std::int64_t incy, std::int64_t stridey,
                               std::int64_t batch_size);

ONEMATH_EXPORT void copy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                               std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                               std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void copy_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                               std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                               std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

ONEMATH_EXPORT void hemv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy);
ONEMATH_EXPORT void hemv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy);

ONEMATH_EXPORT void iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<float, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<double, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);
ONEMATH_EXPORT void iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                          sycl::buffer<std::int64_t, 1>& result,
                          oneapi::math::index_base base = oneapi::math::index_base::zero);

ONEMATH_EXPORT void sbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                         std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                         sycl::buffer<float, 1>& y, std::int64_t incy);
ONEMATH_EXPORT void sbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                         std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx,
                         double beta, sycl::buffer<double, 1>& y, std::int64_t incy);

ONEMATH_EXPORT void asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& result);
ONEMATH_EXPORT void asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<double, 1>& result);
ONEMATH_EXPORT void asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<float, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& result);
ONEMATH_EXPORT void asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<double, 1>& x, std::int64_t incx,
                         sycl::buffer<double, 1>& result);

ONEMATH_EXPORT void tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
                         std::int64_t incx);
ONEMATH_EXPORT void tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx);
ONEMATH_EXPORT void tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx);

ONEMATH_EXPORT void spr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, float alpha, sycl::buffer<float, 1>& x, std::int64_t incx,
                         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& a);
ONEMATH_EXPORT void spr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                         std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
                         std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
                         sycl::buffer<double, 1>& a);

ONEMATH_EXPORT void trsm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                               std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<float, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
ONEMATH_EXPORT void trsm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                               std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<double, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
ONEMATH_EXPORT void trsm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                               std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);
ONEMATH_EXPORT void trsm_batch(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                               uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                               std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

ONEMATH_EXPORT void rotm(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                         std::int64_t incy, sycl::buffer<float, 1>& param);
ONEMATH_EXPORT void rotm(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                         std::int64_t incy, sycl::buffer<double, 1>& param);

ONEMATH_EXPORT void dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                        std::int64_t incy, sycl::buffer<float, 1>& result);
ONEMATH_EXPORT void dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                        std::int64_t incy, sycl::buffer<double, 1>& result);
ONEMATH_EXPORT void dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                        std::int64_t incy, sycl::buffer<double, 1>& result);

ONEMATH_EXPORT void sdsdot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                           float sb, sycl::buffer<float, 1>& x, std::int64_t incx,
                           sycl::buffer<float, 1>& y, std::int64_t incy,
                           sycl::buffer<float, 1>& result);

ONEMATH_EXPORT void her2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k,
                          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                          std::int64_t ldb, float beta, sycl::buffer<std::complex<float>, 1>& c,
                          std::int64_t ldc);
ONEMATH_EXPORT void her2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                          transpose trans, std::int64_t n, std::int64_t k,
                          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                          std::int64_t ldb, double beta, sycl::buffer<std::complex<double>, 1>& c,
                          std::int64_t ldc);

ONEMATH_EXPORT void rotg(oneapi::math::device libkey, sycl::queue& queue, sycl::buffer<float, 1>& a,
                         sycl::buffer<float, 1>& b, sycl::buffer<float, 1>& c,
                         sycl::buffer<float, 1>& s);
ONEMATH_EXPORT void rotg(oneapi::math::device libkey, sycl::queue& queue,
                         sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
                         sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s);
ONEMATH_EXPORT void rotg(oneapi::math::device libkey, sycl::queue& queue,
                         sycl::buffer<std::complex<float>, 1>& a,
                         sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
                         sycl::buffer<std::complex<float>, 1>& s);
ONEMATH_EXPORT void rotg(oneapi::math::device libkey, sycl::queue& queue,
                         sycl::buffer<std::complex<double>, 1>& a,
                         sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
                         sycl::buffer<std::complex<double>, 1>& s);

ONEMATH_EXPORT void omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, float alpha,
                                   sycl::buffer<float, 1>& a, std::int64_t lda,
                                   std::int64_t stride_a, sycl::buffer<float, 1>& b,
                                   std::int64_t ldb, std::int64_t stride_b,
                                   std::int64_t batch_size);
ONEMATH_EXPORT void omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, double alpha,
                                   sycl::buffer<double, 1>& a, std::int64_t lda,
                                   std::int64_t stride_a, sycl::buffer<double, 1>& b,
                                   std::int64_t ldb, std::int64_t stride_b,
                                   std::int64_t batch_size);
ONEMATH_EXPORT void omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                   sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                                   std::int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b,
                                   std::int64_t ldb, std::int64_t stride_b,
                                   std::int64_t batch_size);
ONEMATH_EXPORT void omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                   sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                                   std::int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                                   std::int64_t ldb, std::int64_t stride_b,
                                   std::int64_t batch_size);

ONEMATH_EXPORT void imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, float alpha,
                                   sycl::buffer<float, 1>& ab, std::int64_t lda, std::int64_t ldb,
                                   std::int64_t stride, std::int64_t batch_size);
ONEMATH_EXPORT void imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, double alpha,
                                   sycl::buffer<double, 1>& ab, std::int64_t lda, std::int64_t ldb,
                                   std::int64_t stride, std::int64_t batch_size);
ONEMATH_EXPORT void imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                   sycl::buffer<std::complex<float>, 1>& ab, std::int64_t lda,
                                   std::int64_t ldb, std::int64_t stride, std::int64_t batch_size);
ONEMATH_EXPORT void imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                   sycl::buffer<std::complex<double>, 1>& ab, std::int64_t lda,
                                   std::int64_t ldb, std::int64_t stride, std::int64_t batch_size);

ONEMATH_EXPORT void omatadd_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                  transpose transb, std::int64_t m, std::int64_t n, float alpha,
                                  sycl::buffer<float, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, float beta, sycl::buffer<float, 1>& b,
                                  std::int64_t ldb, std::int64_t stride_b,
                                  sycl::buffer<float, 1>& c, std::int64_t ldc,
                                  std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void omatadd_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                  transpose transb, std::int64_t m, std::int64_t n, double alpha,
                                  sycl::buffer<double, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, double beta, sycl::buffer<double, 1>& b,
                                  std::int64_t ldb, std::int64_t stride_b,
                                  sycl::buffer<double, 1>& c, std::int64_t ldc,
                                  std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void omatadd_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                  transpose transb, std::int64_t m, std::int64_t n,
                                  std::complex<float> alpha,
                                  sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, std::complex<float> beta,
                                  sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                                  std::int64_t stride_b, sycl::buffer<std::complex<float>, 1>& c,
                                  std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);
ONEMATH_EXPORT void omatadd_batch(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                  transpose transb, std::int64_t m, std::int64_t n,
                                  std::complex<double> alpha,
                                  sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, std::complex<double> beta,
                                  sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                                  std::int64_t stride_b, sycl::buffer<std::complex<double>, 1>& c,
                                  std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

ONEMATH_EXPORT void omatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
                             std::int64_t lda, sycl::buffer<float, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void omatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, double alpha,
                             sycl::buffer<double, 1>& a, std::int64_t lda,
                             sycl::buffer<double, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void omatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, std::complex<float> alpha,
                             sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                             sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb);
ONEMATH_EXPORT void omatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, std::complex<double> alpha,
                             sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                             sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb);

ONEMATH_EXPORT void omatcopy2(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                              std::int64_t m, std::int64_t n, float alpha,
                              sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t strideb);
ONEMATH_EXPORT void omatcopy2(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                              std::int64_t m, std::int64_t n, double alpha,
                              sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t strideb);
ONEMATH_EXPORT void omatcopy2(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& b,
                              std::int64_t ldb, std::int64_t strideb);
ONEMATH_EXPORT void omatcopy2(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& b,
                              std::int64_t ldb, std::int64_t strideb);

ONEMATH_EXPORT void imatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, float alpha,
                             sycl::buffer<float, 1>& ab, std::int64_t lda, std::int64_t ldb);
ONEMATH_EXPORT void imatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, double alpha,
                             sycl::buffer<double, 1>& ab, std::int64_t lda, std::int64_t ldb);
ONEMATH_EXPORT void imatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, std::complex<float> alpha,
                             sycl::buffer<std::complex<float>, 1>& ab, std::int64_t lda,
                             std::int64_t ldb);
ONEMATH_EXPORT void imatcopy(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                             std::int64_t m, std::int64_t n, std::complex<double> alpha,
                             sycl::buffer<std::complex<double>, 1>& ab, std::int64_t lda,
                             std::int64_t ldb);

ONEMATH_EXPORT void omatadd(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                            transpose transb, std::int64_t m, std::int64_t n, float alpha,
                            sycl::buffer<float, 1>& a, std::int64_t lda, float beta,
                            sycl::buffer<float, 1>& b, std::int64_t ldb, sycl::buffer<float, 1>& c,
                            std::int64_t ldc);
ONEMATH_EXPORT void omatadd(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                            transpose transb, std::int64_t m, std::int64_t n, double alpha,
                            sycl::buffer<double, 1>& a, std::int64_t lda, double beta,
                            sycl::buffer<double, 1>& b, std::int64_t ldb,
                            sycl::buffer<double, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void omatadd(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                            transpose transb, std::int64_t m, std::int64_t n,
                            std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                            std::int64_t lda, std::complex<float> beta,
                            sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                            sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc);
ONEMATH_EXPORT void omatadd(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                            transpose transb, std::int64_t m, std::int64_t n,
                            std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                            std::int64_t lda, std::complex<double> beta,
                            sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                            sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc);

// USM APIs

ONEMATH_EXPORT sycl::event herk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k, float alpha,
                                const std::complex<float>* a, std::int64_t lda, float beta,
                                std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event herk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k, double alpha,
                                const std::complex<double>* a, std::int64_t lda, double beta,
                                std::complex<double>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                float alpha, float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                double alpha, double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<float> alpha, std::complex<float>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<double> alpha, std::complex<double>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                float alpha, std::complex<float>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event scal(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                double alpha, std::complex<double>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const float* a,
                                std::int64_t lda, float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const double* a,
                                std::int64_t lda, double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const float* a,
                                float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const double* a,
                                double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<float>* a, std::complex<float>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<double>* a, std::complex<double>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event spr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, float alpha, const float* x, std::int64_t incx,
                               float* a, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event spr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, double alpha, const double* x, std::int64_t incx,
                               double* a, const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, float* alpha,
                                      const float** a, std::int64_t* lda, const float** b,
                                      std::int64_t* ldb, float* beta, float** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, double* alpha,
                                      const double** a, std::int64_t* lda, const double** b,
                                      std::int64_t* ldb, double* beta, double** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, std::complex<float>* alpha,
                                      const std::complex<float>** a, std::int64_t* lda,
                                      const std::complex<float>** b, std::int64_t* ldb,
                                      std::complex<float>* beta, std::complex<float>** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, std::complex<double>* alpha,
                                      const std::complex<double>** a, std::int64_t* lda,
                                      const std::complex<double>** b, std::int64_t* ldb,
                                      std::complex<double>* beta, std::complex<double>** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, sycl::half* alpha,
                                      const sycl::half** a, std::int64_t* lda, const sycl::half** b,
                                      std::int64_t* ldb, sycl::half* beta, sycl::half** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, float* alpha,
                                      const sycl::half** a, std::int64_t* lda, const sycl::half** b,
                                      std::int64_t* ldb, float* beta, float** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, float* alpha,
                                      const std::int8_t** a, std::int64_t* lda,
                                      const std::int8_t** b, std::int64_t* ldb, float* beta,
                                      float** c, std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* transa, transpose* transb, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, float* alpha,
                                      const std::int8_t** a, std::int64_t* lda,
                                      const std::int8_t** b, std::int64_t* ldb, float* beta,
                                      std::int32_t** c, std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, float alpha, const float* a,
                                      std::int64_t lda, std::int64_t stride_a, const float* b,
                                      std::int64_t ldb, std::int64_t stride_b, float beta, float* c,
                                      std::int64_t ldc, std::int64_t stride_c,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, double alpha, const double* a,
                                      std::int64_t lda, std::int64_t stride_a, const double* b,
                                      std::int64_t ldb, std::int64_t stride_b, double beta,
                                      double* c, std::int64_t ldc, std::int64_t stride_c,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
    const std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
    const std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
    std::complex<float>* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
    const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
    const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double>* c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, sycl::half alpha,
                                      const sycl::half* a, std::int64_t lda, std::int64_t stride_a,
                                      const sycl::half* b, std::int64_t ldb, std::int64_t stride_b,
                                      sycl::half beta, sycl::half* c, std::int64_t ldc,
                                      std::int64_t stride_c, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, float alpha,
                                      const sycl::half* a, std::int64_t lda, std::int64_t stride_a,
                                      const sycl::half* b, std::int64_t ldb, std::int64_t stride_b,
                                      float beta, float* c, std::int64_t ldc, std::int64_t stride_c,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, float alpha,
                                      const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                                      const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                                      float beta, float* c, std::int64_t ldc, std::int64_t stride_c,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose transa, transpose transb, std::int64_t m,
                                      std::int64_t n, std::int64_t k, float alpha,
                                      const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                                      const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                                      float beta, std::int32_t* c, std::int64_t ldc,
                                      std::int64_t stride_c, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k, float alpha,
                                const float* a, std::int64_t lda, float beta, float* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k, double alpha,
                                const double* a, std::int64_t lda, double beta, double* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, std::complex<float> beta, std::complex<float>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, std::complex<double> beta,
                                std::complex<double>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo* upper_lower, transpose* trans, std::int64_t* n,
                                      std::int64_t* k, float* alpha, const float** a,
                                      std::int64_t* lda, float* beta, float** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo* upper_lower, transpose* trans, std::int64_t* n,
                                      std::int64_t* k, double* alpha, const double** a,
                                      std::int64_t* lda, double* beta, double** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo* upper_lower, transpose* trans, std::int64_t* n,
                                      std::int64_t* k, std::complex<float>* alpha,
                                      const std::complex<float>** a, std::int64_t* lda,
                                      std::complex<float>* beta, std::complex<float>** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo* upper_lower, transpose* trans, std::int64_t* n,
                                      std::int64_t* k, std::complex<double>* alpha,
                                      const std::complex<double>** a, std::int64_t* lda,
                                      std::complex<double>* beta, std::complex<double>** c,
                                      std::int64_t* ldc, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo upper_lower, transpose trans, std::int64_t n,
                                      std::int64_t k, float alpha, const float* a, std::int64_t lda,
                                      std::int64_t stride_a, float beta, float* c, std::int64_t ldc,
                                      std::int64_t stride_c, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo upper_lower, transpose trans, std::int64_t n,
                                      std::int64_t k, double alpha, const double* a,
                                      std::int64_t lda, std::int64_t stride_a, double beta,
                                      double* c, std::int64_t ldc, std::int64_t stride_c,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo upper_lower, transpose trans, std::int64_t n,
                                      std::int64_t k, std::complex<float> alpha,
                                      const std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float> beta,
                                      std::complex<float>* c, std::int64_t ldc,
                                      std::int64_t stride_c, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syrk_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      uplo upper_lower, transpose trans, std::int64_t n,
                                      std::int64_t k, std::complex<double> alpha,
                                      const std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double> beta,
                                      std::complex<double>* c, std::int64_t ldc,
                                      std::int64_t stride_c, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event her2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event her2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                const std::complex<float>* x, std::int64_t incx,
                                std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                const std::complex<double>* x, std::int64_t incx,
                                std::complex<double> beta, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               std::complex<float>* x, std::int64_t incx, std::complex<float>* y,
                               std::int64_t incy, float c, float s,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               std::complex<double>* x, std::int64_t incx, std::complex<double>* y,
                               std::int64_t incy, double c, double s,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               float* x, std::int64_t incx, float* y, std::int64_t incy, float c,
                               float s, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               double* x, std::int64_t incx, double* y, std::int64_t incy, double c,
                               double s, const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                float alpha, const float* x, std::int64_t incx, float* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                double alpha, const double* x, std::int64_t incx, double* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<float> alpha, const std::complex<float>* x,
                                std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<double> alpha, const std::complex<double>* x,
                                std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, float* alpha, const float** x,
                                      std::int64_t* incx, float** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, double* alpha, const double** x,
                                      std::int64_t* incx, double** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, std::complex<float>* alpha,
                                      const std::complex<float>** x, std::int64_t* incx,
                                      std::complex<float>** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, std::complex<double>* alpha,
                                      const std::complex<double>** x, std::int64_t* incx,
                                      std::complex<double>** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, float alpha, const float* x,
                                      std::int64_t incx, std::int64_t stridex, float* y,
                                      std::int64_t incy, std::int64_t stridey,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, double alpha, const double* x,
                                      std::int64_t incx, std::int64_t stridex, double* y,
                                      std::int64_t incy, std::int64_t stridey,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float>* x, std::int64_t incx,
                                      std::int64_t stridex, std::complex<float>* y,
                                      std::int64_t incy, std::int64_t stridey,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double>* x, std::int64_t incx,
                                      std::int64_t stridex, std::complex<double>* y,
                                      std::int64_t incy, std::int64_t stridey,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 float alpha, const float* x, std::int64_t incx, const float beta,
                                 float* y, std::int64_t incy,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 double alpha, const double* x, std::int64_t incx,
                                 const double beta, double* y, std::int64_t incy,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 std::complex<float> alpha, const std::complex<float>* x,
                                 std::int64_t incx, const std::complex<float> beta,
                                 std::complex<float>* y, std::int64_t incy,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event axpby(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 std::complex<double> alpha, const std::complex<double>* x,
                                 std::int64_t incx, const std::complex<double> beta,
                                 std::complex<double>* y, std::int64_t incy,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gerc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gerc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k, float alpha,
                                 const float* a, std::int64_t lda, const float* b, std::int64_t ldb,
                                 float beta, float* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k, double alpha,
                                 const double* a, std::int64_t lda, const double* b,
                                 std::int64_t ldb, double beta, double* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<float> alpha, const std::complex<float>* a,
                                 std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                 std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syr2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<double> alpha, const std::complex<double>* a,
                                 std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                 std::complex<double> beta, std::complex<double>* c,
                                 std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, float alpha, const float* a,
                                std::int64_t lda, const float* x, std::int64_t incx, float beta,
                                float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, double alpha, const double* a,
                                std::int64_t lda, const double* x, std::int64_t incx, double beta,
                                double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                const std::complex<float>* x, std::int64_t incx,
                                std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                const std::complex<double>* x, std::int64_t incx,
                                std::complex<double> beta, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose trans, std::int64_t m, std::int64_t n, float alpha,
                                      const float* a, std::int64_t lda, std::int64_t stridea,
                                      const float* x, std::int64_t incx, std::int64_t stridex,
                                      float beta, float* y, std::int64_t incy, std::int64_t stridey,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose trans, std::int64_t m, std::int64_t n, double alpha,
                                      const double* a, std::int64_t lda, std::int64_t stridea,
                                      const double* x, std::int64_t incx, std::int64_t stridex,
                                      double beta, double* y, std::int64_t incy,
                                      std::int64_t stridey, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose trans, std::int64_t m,
    std::int64_t n, std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
    std::int64_t stridea, const std::complex<float>* x, std::int64_t incx, std::int64_t stridex,
    std::complex<float> beta, std::complex<float>* y, std::int64_t incy, std::int64_t stridey,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose trans, std::int64_t m,
    std::int64_t n, std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
    std::int64_t stridea, const std::complex<double>* x, std::int64_t incx, std::int64_t stridex,
    std::complex<double> beta, std::complex<double>* y, std::int64_t incy, std::int64_t stridey,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* trans, std::int64_t* m, std::int64_t* n,
                                      float* alpha, const float** a, std::int64_t* lda,
                                      const float** x, std::int64_t* incx, float* beta, float** y,
                                      std::int64_t* incy, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* trans, std::int64_t* m, std::int64_t* n,
                                      double* alpha, const double** a, std::int64_t* lda,
                                      const double** x, std::int64_t* incx, double* beta,
                                      double** y, std::int64_t* incy, std::int64_t group_count,
                                      std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* trans, std::int64_t* m, std::int64_t* n,
                                      std::complex<float>* alpha, const std::complex<float>** a,
                                      std::int64_t* lda, const std::complex<float>** x,
                                      std::int64_t* incx, std::complex<float>* beta,
                                      std::complex<float>** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemv_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      transpose* trans, std::int64_t* m, std::int64_t* n,
                                      std::complex<double>* alpha, const std::complex<double>** a,
                                      std::int64_t* lda, const std::complex<double>** x,
                                      std::int64_t* incx, std::complex<double>* beta,
                                      std::complex<double>** y, std::int64_t* incy,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, std::int64_t m, std::int64_t n,
                                      const float* a, std::int64_t lda, std::int64_t stridea,
                                      const float* x, std::int64_t incx, std::int64_t stridex,
                                      float* c, std::int64_t ldc, std::int64_t stridec,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, std::int64_t m, std::int64_t n,
                                      const double* a, std::int64_t lda, std::int64_t stridea,
                                      const double* x, std::int64_t incx, std::int64_t stridex,
                                      double* c, std::int64_t ldc, std::int64_t stridec,
                                      std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, std::int64_t m, std::int64_t n,
                                      const std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stridea, const std::complex<float>* x,
                                      std::int64_t incx, std::int64_t stridex,
                                      std::complex<float>* c, std::int64_t ldc,
                                      std::int64_t stridec, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, std::int64_t m, std::int64_t n,
                                      const std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stridea, const std::complex<double>* x,
                                      std::int64_t incx, std::int64_t stridex,
                                      std::complex<double>* c, std::int64_t ldc,
                                      std::int64_t stridec, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, std::int64_t* m, std::int64_t* n,
                                      const float** a, std::int64_t* lda, const float** x,
                                      std::int64_t* incx, float** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, std::int64_t* m, std::int64_t* n,
                                      const double** a, std::int64_t* lda, const double** x,
                                      std::int64_t* incx, double** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, std::int64_t* m, std::int64_t* n,
                                      const std::complex<float>** a, std::int64_t* lda,
                                      const std::complex<float>** x, std::int64_t* incx,
                                      std::complex<float>** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dgmm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, std::int64_t* m, std::int64_t* n,
                                      const std::complex<double>** a, std::int64_t* lda,
                                      const std::complex<double>** x, std::int64_t* incx,
                                      std::complex<double>** c, std::int64_t* ldc,
                                      std::int64_t group_count, std::int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event her(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, float alpha, const std::complex<float>* x,
                               std::int64_t incx, std::complex<float>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event her(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, double alpha, const std::complex<double>* x,
                               std::int64_t incx, std::complex<double>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hpr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, float alpha, const std::complex<float>* x,
                               std::int64_t incx, std::complex<float>* a,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hpr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, double alpha, const std::complex<double>* x,
                               std::int64_t incx, std::complex<double>* a,
                               const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const float* x, std::int64_t incx, std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const double* x, std::int64_t incx, std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const std::complex<float>* x, std::int64_t incx,
                                 std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamin(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const std::complex<double>* x, std::int64_t incx,
                                 std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* a, const std::complex<float>* x,
                                std::int64_t incx, std::complex<float> beta, std::complex<float>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hpmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* a, const std::complex<double>* x,
                                std::int64_t incx, std::complex<double> beta,
                                std::complex<double>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event spmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, float alpha, const float* a, const float* x,
                                std::int64_t incx, float beta, float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event spmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, double alpha, const double* a, const double* x,
                                std::int64_t incx, double beta, double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event rotmg(oneapi::math::device libkey, sycl::queue& queue, float* d1,
                                 float* d2, float* x1, float y1, float* param,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rotmg(oneapi::math::device libkey, sycl::queue& queue, double* d1,
                                 double* d2, double* x1, double y1, double* param,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                float* x, std::int64_t incx, float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                double* x, std::int64_t incx, double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<float>* x, std::int64_t incx, std::complex<float>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event swap(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                std::complex<double>* x, std::int64_t incx, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event geru(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event geru(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<float>* x, std::int64_t incx, float* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<double>* x, std::int64_t incx, double* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const float* x, std::int64_t incx, float* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event nrm2(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const double* x, std::int64_t incx, double* result,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                                 float alpha, const float* a, std::int64_t lda, const float* b,
                                 std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                                 double alpha, const double* a, std::int64_t lda, const double* b,
                                 std::int64_t ldb, double beta, double* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                                 std::complex<float> alpha, const std::complex<float>* a,
                                 std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                 std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemmt(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                                 std::complex<double> alpha, const std::complex<double>* a,
                                 std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                 std::complex<double> beta, std::complex<double>* c,
                                 std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                float alpha, const float* a, std::int64_t lda, const float* b,
                                std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                double alpha, const double* a, std::int64_t lda, const double* b,
                                std::int64_t ldb, double beta, double* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                sycl::half alpha, const sycl::half* a, std::int64_t lda,
                                const sycl::half* b, std::int64_t ldb, sycl::half beta,
                                sycl::half* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                float alpha, const sycl::half* a, std::int64_t lda,
                                const sycl::half* b, std::int64_t ldb, float beta, float* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm(oneapi::math::device libkey, sycl::queue& queue, transpose transa,
                                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                                float alpha, const bfloat16* a, std::int64_t lda, const bfloat16* b,
                                std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gemm_bias(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose transa, transpose transb, offset offsetc, int64_t m,
                                     int64_t n, int64_t k, float alpha, const std::int8_t* a,
                                     int64_t lda, std::int8_t ao, const std::uint8_t* b,
                                     int64_t ldb, std::uint8_t bo, float beta, std::int32_t* c,
                                     int64_t ldc, const std::int32_t* co,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_bias(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose transa, transpose transb, offset offsetc, int64_t m,
                                     int64_t n, int64_t k, float alpha, const std::int8_t* a,
                                     int64_t lda, std::int8_t ao, const std::int8_t* b, int64_t ldb,
                                     std::int8_t bo, float beta, std::int32_t* c, int64_t ldc,
                                     const std::int32_t* co,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_bias(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose transa, transpose transb, offset offsetc, int64_t m,
                                     int64_t n, int64_t k, float alpha, const std::uint8_t* a,
                                     int64_t lda, std::uint8_t ao, const std::int8_t* b,
                                     int64_t ldb, std::int8_t bo, float beta, std::int32_t* c,
                                     int64_t ldc, const std::int32_t* co,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gemm_bias(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose transa, transpose transb, offset offsetc, int64_t m,
                                     int64_t n, int64_t k, float alpha, const std::uint8_t* a,
                                     int64_t lda, std::uint8_t ao, const std::uint8_t* b,
                                     int64_t ldb, std::uint8_t bo, float beta, std::int32_t* c,
                                     int64_t ldc, const std::int32_t* co,
                                     const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, float alpha, const float* x, std::int64_t incx,
                                const float* y, std::int64_t incy, float* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, double alpha, const double* x, std::int64_t incx,
                                const double* y, std::int64_t incy, double* a, std::int64_t lda,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event ger(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, float alpha, const float* x, std::int64_t incx,
                               const float* y, std::int64_t incy, float* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event ger(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, double alpha, const double* x, std::int64_t incx,
                               const double* y, std::int64_t incy, double* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                float* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                double* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, uplo upper_lower, transpose trans,
                                      diag unit_diag, int64_t m, int64_t n, float alpha,
                                      const float* a, int64_t lda, int64_t stride_a, float* b,
                                      int64_t ldb, int64_t stride_b, int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, uplo upper_lower, transpose trans,
                                      diag unit_diag, int64_t m, int64_t n, double alpha,
                                      const double* a, int64_t lda, int64_t stride_a, double* b,
                                      int64_t ldb, int64_t stride_b, int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, uplo upper_lower, transpose trans,
                                      diag unit_diag, int64_t m, int64_t n,
                                      std::complex<float> alpha, const std::complex<float>* a,
                                      int64_t lda, int64_t stride_a, std::complex<float>* b,
                                      int64_t ldb, int64_t stride_b, int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side left_right, uplo upper_lower, transpose trans,
                                      diag unit_diag, int64_t m, int64_t n,
                                      std::complex<double> alpha, const std::complex<double>* a,
                                      int64_t lda, int64_t stride_a, std::complex<double>* b,
                                      int64_t ldb, int64_t stride_b, int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, uplo* upper_lower, transpose* trans,
                                      diag* unit_diag, int64_t* m, int64_t* n, float* alpha,
                                      const float** a, int64_t* lda, float** b, int64_t* ldb,
                                      int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, uplo* upper_lower, transpose* trans,
                                      diag* unit_diag, int64_t* m, int64_t* n, double* alpha,
                                      const double** a, int64_t* lda, double** b, int64_t* ldb,
                                      int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, uplo* upper_lower, transpose* trans,
                                      diag* unit_diag, int64_t* m, int64_t* n,
                                      std::complex<float>* alpha, const std::complex<float>** a,
                                      int64_t* lda, std::complex<float>** b, int64_t* ldb,
                                      int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsm_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      side* left_right, uplo* upper_lower, transpose* trans,
                                      diag* unit_diag, int64_t* m, int64_t* n,
                                      std::complex<double>* alpha, const std::complex<double>** a,
                                      int64_t* lda, std::complex<double>** b, int64_t* ldb,
                                      int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event dotu(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dotu(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* result,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hemm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hemm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hpr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* a,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hpr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* a,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                float alpha, const float* a, std::int64_t lda, const float* x,
                                std::int64_t incx, float beta, float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                double alpha, const double* a, std::int64_t lda, const double* x,
                                std::int64_t incx, double beta, double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                                std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event gbmv(oneapi::math::device libkey, sycl::queue& queue, transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                                std::complex<double> beta, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const float* a, std::int64_t lda, float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const double* a, std::int64_t lda, double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                                const float* a, std::int64_t lda, const float* b, std::int64_t ldb,
                                float beta, float* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                                const double* a, std::int64_t lda, const double* b,
                                std::int64_t ldb, double beta, double* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event symm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event dotc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float>* y, std::int64_t incy,
                                std::complex<float>* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dotc(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double>* y, std::int64_t incy,
                                std::complex<double>* result,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event syr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, float alpha, const float* x, std::int64_t incx,
                               float* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event syr(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                               std::int64_t n, double alpha, const double* x, std::int64_t incx,
                               double* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                float* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                double* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trmm(oneapi::math::device libkey, sycl::queue& queue, side left_right,
                                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* b, std::int64_t ldb,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event symv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                const float* x, std::int64_t incx, float beta, float* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event symv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                const double* x, std::int64_t incx, double beta, double* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const float* a,
                                float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const double* a,
                                double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<float>* a, std::complex<float>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tpsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<double>* a, std::complex<double>* x,
                                std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const float* a,
                                std::int64_t lda, float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, const double* a,
                                std::int64_t lda, double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event trsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const float* x, std::int64_t incx, float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const double* x, std::int64_t incx, double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<float>* x, std::int64_t incx,
                                std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<double>* x, std::int64_t incx,
                                std::complex<double>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, const float** x, std::int64_t* incx,
                                      float** y, std::int64_t* incy, int64_t group_count,
                                      int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, const double** x, std::int64_t* incx,
                                      double** y, std::int64_t* incy, int64_t group_count,
                                      int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, const std::complex<float>** x,
                                      std::int64_t* incx, std::complex<float>** y,
                                      std::int64_t* incy, int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t* n, const std::complex<double>** x,
                                      std::int64_t* incx, std::complex<double>** y,
                                      std::int64_t* incy, int64_t group_count, int64_t* group_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, const float* x, std::int64_t incx,
                                      std::int64_t stridex, float* y, std::int64_t incy,
                                      std::int64_t stridey, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, const double* x, std::int64_t incx,
                                      std::int64_t stridex, double* y, std::int64_t incy,
                                      std::int64_t stridey, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, const std::complex<float>* x,
                                      std::int64_t incx, std::int64_t stridex,
                                      std::complex<float>* y, std::int64_t incy,
                                      std::int64_t stridey, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event copy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                      std::int64_t n, const std::complex<double>* x,
                                      std::int64_t incx, std::int64_t stridex,
                                      std::complex<double>* y, std::int64_t incy,
                                      std::int64_t stridey, std::int64_t batch_size,
                                      const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event hemv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                const std::complex<float>* x, std::int64_t incx,
                                std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event hemv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                const std::complex<double>* x, std::int64_t incx,
                                std::complex<double> beta, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const float* x, std::int64_t incx, std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const double* x, std::int64_t incx, std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const std::complex<float>* x, std::int64_t incx,
                                 std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event iamax(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                 const std::complex<double>* x, std::int64_t incx,
                                 std::int64_t* result,
                                 oneapi::math::index_base base = oneapi::math::index_base::zero,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event sbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::int64_t k, float alpha, const float* a,
                                std::int64_t lda, const float* x, std::int64_t incx, float beta,
                                float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event sbmv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, std::int64_t k, double alpha, const double* a,
                                std::int64_t lda, const double* x, std::int64_t incx, double beta,
                                double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<float>* x, std::int64_t incx, float* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const std::complex<double>* x, std::int64_t incx, double* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const float* x, std::int64_t incx, float* result,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event asum(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                const double* x, std::int64_t incx, double* result,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const float* a, std::int64_t lda, float* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const double* a, std::int64_t lda, double* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event tbsv(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                                const std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* x, std::int64_t incx,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event spr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, float alpha, const float* x, std::int64_t incx,
                                const float* y, std::int64_t incy, float* a,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event spr2(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                std::int64_t n, double alpha, const double* x, std::int64_t incx,
                                const double* y, std::int64_t incy, double* a,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event rotm(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                float* x, std::int64_t incx, float* y, std::int64_t incy,
                                float* param, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rotm(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                double* x, std::int64_t incx, double* y, std::int64_t incy,
                                double* param, const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               const float* x, std::int64_t incx, const float* y, std::int64_t incy,
                               float* result, const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               const double* x, std::int64_t incx, const double* y,
                               std::int64_t incy, double* result,
                               const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event dot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                               const float* x, std::int64_t incx, const float* y, std::int64_t incy,
                               double* result, const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event sdsdot(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                                  float sb, const float* x, std::int64_t incx, const float* y,
                                  std::int64_t incy, float* result,
                                  const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event her2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<float> alpha, const std::complex<float>* a,
                                 std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                 float beta, std::complex<float>* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event her2k(oneapi::math::device libkey, sycl::queue& queue, uplo upper_lower,
                                 transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<double> alpha, const std::complex<double>* a,
                                 std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                 double beta, std::complex<double>* c, std::int64_t ldc,
                                 const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event rotg(oneapi::math::device libkey, sycl::queue& queue, float* a, float* b,
                                float* c, float* s,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rotg(oneapi::math::device libkey, sycl::queue& queue, double* a,
                                double* b, double* c, double* s,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rotg(oneapi::math::device libkey, sycl::queue& queue,
                                std::complex<float>* a, std::complex<float>* b, float* c,
                                std::complex<float>* s,
                                const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event rotg(oneapi::math::device libkey, sycl::queue& queue,
                                std::complex<double>* a, std::complex<double>* b, double* c,
                                std::complex<double>* s,
                                const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          float alpha, const float* a, std::int64_t lda,
                                          std::int64_t stride_a, float* b, std::int64_t ldb,
                                          std::int64_t stride_b, std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          double alpha, const double* a, std::int64_t lda,
                                          std::int64_t stride_a, double* b, std::int64_t ldb,
                                          std::int64_t stride_b, std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          std::complex<float> alpha, const std::complex<float>* a,
                                          std::int64_t lda, std::int64_t stride_a,
                                          std::complex<float>* b, std::int64_t ldb,
                                          std::int64_t stride_b, std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          std::complex<double> alpha, const std::complex<double>* a,
                                          std::int64_t lda, std::int64_t stride_a,
                                          std::complex<double>* b, std::int64_t ldb,
                                          std::int64_t stride_b, std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          float alpha, float* ab, std::int64_t lda,
                                          std::int64_t ldb, std::int64_t stride,
                                          std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          double alpha, double* ab, std::int64_t lda,
                                          std::int64_t ldb, std::int64_t stride,
                                          std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          std::complex<float> alpha, std::complex<float>* ab,
                                          std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                          std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose trans, std::int64_t m, std::int64_t n,
                                          std::complex<double> alpha, std::complex<double>* ab,
                                          std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                          std::int64_t batch_size,
                                          const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatadd_batch(oneapi::math::device libkey, sycl::queue& queue,
                                         transpose transa, transpose transb, std::int64_t m,
                                         std::int64_t n, float alpha, const float* a,
                                         std::int64_t lda, std::int64_t stride_a, float beta,
                                         const float* b, std::int64_t ldb, std::int64_t stride_b,
                                         float* c, std::int64_t ldc, std::int64_t stride_c,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd_batch(oneapi::math::device libkey, sycl::queue& queue,
                                         transpose transa, transpose transb, std::int64_t m,
                                         std::int64_t n, double alpha, const double* a,
                                         std::int64_t lda, std::int64_t stride_a, double beta,
                                         const double* b, std::int64_t ldb, std::int64_t stride_b,
                                         double* c, std::int64_t ldc, std::int64_t stride_c,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float>* a,
    std::int64_t lda, std::int64_t stride_a, std::complex<float> beta, const std::complex<float>* b,
    std::int64_t ldb, std::int64_t stride_b, std::complex<float>* c, std::int64_t ldc,
    std::int64_t stride_c, std::int64_t batch_size,
    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd_batch(
    oneapi::math::device libkey, sycl::queue& queue, transpose transa, transpose transb,
    std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double>* a,
    std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
    const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b, std::complex<double>* c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n, float alpha,
                                    const float* a, std::int64_t lda, float* b, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n, double alpha,
                                    const double* a, std::int64_t lda, double* b, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha, const std::complex<float>* a,
                                    std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha, const std::complex<double>* a,
                                    std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatcopy2(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose trans, std::int64_t m, std::int64_t n, float alpha,
                                     const float* a, std::int64_t lda, std::int64_t stridea,
                                     float* b, std::int64_t ldb, std::int64_t strideb,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy2(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose trans, std::int64_t m, std::int64_t n, double alpha,
                                     const double* a, std::int64_t lda, std::int64_t stridea,
                                     double* b, std::int64_t ldb, std::int64_t strideb,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy2(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose trans, std::int64_t m, std::int64_t n,
                                     std::complex<float> alpha, const std::complex<float>* a,
                                     std::int64_t lda, std::int64_t stridea, std::complex<float>* b,
                                     std::int64_t ldb, std::int64_t strideb,
                                     const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy2(oneapi::math::device libkey, sycl::queue& queue,
                                     transpose trans, std::int64_t m, std::int64_t n,
                                     std::complex<double> alpha, const std::complex<double>* a,
                                     std::int64_t lda, std::int64_t stridea,
                                     std::complex<double>* b, std::int64_t ldb,
                                     std::int64_t strideb,
                                     const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event imatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n, float alpha,
                                    float* ab, std::int64_t lda, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n, double alpha,
                                    double* ab, std::int64_t lda, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha, std::complex<float>* ab,
                                    std::int64_t lda, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy(oneapi::math::device libkey, sycl::queue& queue,
                                    transpose trans, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha, std::complex<double>* ab,
                                    std::int64_t lda, std::int64_t ldb,
                                    const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatadd(oneapi::math::device libkey, sycl::queue& queue,
                                   transpose transa, transpose transb, std::int64_t m,
                                   std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                   float beta, const float* b, std::int64_t ldb, float* c,
                                   std::int64_t ldc,
                                   const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd(oneapi::math::device libkey, sycl::queue& queue,
                                   transpose transa, transpose transb, std::int64_t m,
                                   std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                   double beta, const double* b, std::int64_t ldb, double* c,
                                   std::int64_t ldc,
                                   const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd(oneapi::math::device libkey, sycl::queue& queue,
                                   transpose transa, transpose transb, std::int64_t m,
                                   std::int64_t n, std::complex<float> alpha,
                                   const std::complex<float>* a, std::int64_t lda,
                                   std::complex<float> beta, const std::complex<float>* b,
                                   std::int64_t ldb, std::complex<float>* c, std::int64_t ldc,
                                   const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatadd(oneapi::math::device libkey, sycl::queue& queue,
                                   transpose transa, transpose transb, std::int64_t m,
                                   std::int64_t n, std::complex<double> alpha,
                                   const std::complex<double>* a, std::int64_t lda,
                                   std::complex<double> beta, const std::complex<double>* b,
                                   std::int64_t ldb, std::complex<double>* c, std::int64_t ldc,
                                   const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          float* alpha, const float** a, std::int64_t* lda,
                                          float** b, std::int64_t* ldb, std::int64_t group_count,
                                          std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          double* alpha, const double** a, std::int64_t* lda,
                                          double** b, std::int64_t* ldb, std::int64_t group_count,
                                          std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          std::complex<float>* alpha, const std::complex<float>** a,
                                          std::int64_t* lda, std::complex<float>** b,
                                          std::int64_t* ldb, std::int64_t group_count,
                                          std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event omatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          std::complex<double>* alpha,
                                          const std::complex<double>** a, std::int64_t* lda,
                                          std::complex<double>** b, std::int64_t* ldb,
                                          std::int64_t group_count, std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});

ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          float* alpha, float** ab, std::int64_t* lda,
                                          std::int64_t* ldb, std::int64_t group_count,
                                          std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          double* alpha, double** ab, std::int64_t* lda,
                                          std::int64_t* ldb, std::int64_t group_count,
                                          std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          std::complex<float>* alpha, std::complex<float>** ab,
                                          std::int64_t* lda, std::int64_t* ldb,
                                          std::int64_t group_count, std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
ONEMATH_EXPORT sycl::event imatcopy_batch(oneapi::math::device libkey, sycl::queue& queue,
                                          transpose* trans, std::int64_t* m, std::int64_t* n,
                                          std::complex<double>* alpha, std::complex<double>** ab,
                                          std::int64_t* lda, std::int64_t* ldb,
                                          std::int64_t group_count, std::int64_t* groupsize,
                                          const std::vector<sycl::event>& dependencies = {});
