// Buffer APIs

void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<std::complex<float>>& tauq, sycl::buffer<std::complex<float>>& taup,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& d, sycl::buffer<double>& e,
           sycl::buffer<double>& tauq, sycl::buffer<double>& taup, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tauq, sycl::buffer<float>& taup, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tauq,
           sycl::buffer<std::complex<double>>& taup, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<double>& b, std::int64_t ldb, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& b,
           std::int64_t ldb, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
           std::int64_t m, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<double>& s, sycl::buffer<double>& u, std::int64_t ldu,
           sycl::buffer<double>& vt, std::int64_t ldvt, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
           std::int64_t m, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& s, sycl::buffer<float>& u, std::int64_t ldu,
           sycl::buffer<float>& vt, std::int64_t ldvt, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
           std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<float>& s, sycl::buffer<std::complex<float>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
           std::int64_t m, std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<double>& s, sycl::buffer<std::complex<double>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<double>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& w,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz, oneapi::math::uplo uplo,
           std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb, sycl::buffer<float>& w,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz, oneapi::math::uplo uplo,
           std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& d,
           sycl::buffer<float>& e, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& w,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& w,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz, oneapi::math::uplo uplo,
           std::int64_t n, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
           std::int64_t ldb, sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz, oneapi::math::uplo uplo,
           std::int64_t n, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
           std::int64_t ldb, sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& d, sycl::buffer<double>& e,
           sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size);

void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size);

void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
           oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size);

void ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size);

void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size);

void ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size);

void unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size);

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<float>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size);

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<double>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size);

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size);

void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size);

void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size);

void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size);

void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv, sycl::buffer<float>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size);

void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size);

void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);

void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);

void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size);

void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size);

void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size);

void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size);

void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size);

void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size);

// USM APIs

sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, float* d, float* e, std::complex<float>* tauq,
                  std::complex<float>* taup, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  double* d, double* e, double* tauq, double* taup, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  float* d, float* e, float* tauq, float* taup, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, double* d, double* e, std::complex<double>* tauq,
                  std::complex<double>* taup, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, std::int64_t* ipiv, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  std::int64_t* ipiv, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  std::int64_t* ipiv, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::int64_t* ipiv, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getri(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                  std::int64_t* ipiv, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getri(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                  std::int64_t* ipiv, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getri(sycl::queue& queue, std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                  std::int64_t nrhs, std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* b, std::int64_t ldb, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                  std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t* ipiv, double* b,
                  std::int64_t ldb, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                  std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t* ipiv, float* b,
                  std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event getrs(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                  std::int64_t nrhs, std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* b, std::int64_t ldb, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, double* a, std::int64_t lda, double* s, double* u,
                  std::int64_t ldu, double* vt, std::int64_t ldvt, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, float* a, std::int64_t lda, float* s, float* u,
                  std::int64_t ldu, float* vt, std::int64_t ldvt, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  float* s, std::complex<float>* u, std::int64_t ldu, std::complex<float>* vt,
                  std::int64_t ldvt, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gesvd(sycl::queue& queue, oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                  std::int64_t m, std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  double* s, std::complex<double>* u, std::int64_t ldu, std::complex<double>* vt,
                  std::int64_t ldvt, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, float* w,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event heevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, double* w,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* b, std::int64_t ldb, float* w,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* b, std::int64_t ldb, double* w,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, float* d, float* e,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event hetrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, double* d, double* e,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event hetrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event orgbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, double* a,
                  std::int64_t lda, double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, float* a,
                  std::int64_t lda, float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event orgtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, float* a,
                  std::int64_t lda, float* tau, float* c, std::int64_t ldc, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ormtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, double* a,
                  std::int64_t lda, double* tau, double* c, std::int64_t ldc, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* c, std::int64_t ldc, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ormrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* c, std::int64_t ldc, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* c, std::int64_t ldc, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ormqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* c, std::int64_t ldc, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potri(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  float* a, std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  double* a, std::int64_t lda, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                  std::int64_t ldb, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event potrs(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* w, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event syevd(sycl::queue& queue, oneapi::math::job jobz, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* w, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, double* a, std::int64_t lda, double* b,
                  std::int64_t ldb, double* w, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, float* a, std::int64_t lda, float* b,
                  std::int64_t ldb, float* w, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* d, double* e, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event sytrd(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* d, float* e, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event sytrf(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                  std::int64_t ldb, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, double* a,
                  std::int64_t lda, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs, float* a,
                  std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trtrs(sycl::queue& queue, oneapi::math::uplo uplo, oneapi::math::transpose trans,
                  oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ungbr(sycl::queue& queue, oneapi::math::generate vec, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event ungtr(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* c,
                  std::int64_t ldc, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event unmrq(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* c,
                  std::int64_t ldc, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* c,
                  std::int64_t ldc, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event unmqr(sycl::queue& queue, oneapi::math::side side, oneapi::math::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* c,
                  std::int64_t ldc, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* c, std::int64_t ldc, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event unmtr(sycl::queue& queue, oneapi::math::side side, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* c, std::int64_t ldc, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<float>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, float** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, double** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<float>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<double>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, float* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, float** a, std::int64_t* lda, std::int64_t** ipiv,
                        float** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, double** a, std::int64_t* lda, std::int64_t** ipiv,
                        double** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<float>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<double>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        float* a, std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        double* a, std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        float** a, std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        double** a, std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, double** a,
                        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        float* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        double* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::int64_t* nrhs, float** a, std::int64_t* lda, float** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::int64_t* nrhs, double** a, std::int64_t* lda, double** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<float>** a, std::int64_t* lda,
                        std::complex<float>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<double>** a, std::int64_t* lda,
                        std::complex<double>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<float>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<double>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies = {});

// SCRATCHPAD APIs

template <typename T>
std::int64_t gebrd_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t gerqf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t geqrf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t gesvd_scratchpad_size(sycl::queue& queue, oneapi::math::jobsvd jobu,
                                   oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldu, std::int64_t ldvt);

template <typename T>
std::int64_t getrf_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t getri_scratchpad_size(sycl::queue& queue, std::int64_t n, std::int64_t lda);

template <typename T>
std::int64_t getrs_scratchpad_size(sycl::queue& queue, oneapi::math::transpose trans,
                                   std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                   std::int64_t ldb);

template <typename T>
std::int64_t heevd_scratchpad_size(sycl::queue& queue, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda);

template <typename T>
std::int64_t hegvd_scratchpad_size(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldb);

template <typename T>
std::int64_t hetrd_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t hetrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t orgbr_scratchpad_size(sycl::queue& queue, oneapi::math::generate vect, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda);

template <typename T>
std::int64_t orgtr_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t orgqr_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda);

template <typename T>
std::int64_t ormrq_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc);

template <typename T>
std::int64_t ormqr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc);

template <typename T>
std::int64_t ormtr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldc);

template <typename T>
std::int64_t potrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t potrs_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, std::int64_t lda, std::int64_t ldb);

template <typename T>
std::int64_t potri_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t sytrf_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t syevd_scratchpad_size(sycl::queue& queue, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda);

template <typename T>
std::int64_t sygvd_scratchpad_size(sycl::queue& queue, std::int64_t itype, oneapi::math::job jobz,
                                   oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldb);

template <typename T>
std::int64_t sytrd_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t trtrs_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                   oneapi::math::transpose trans, oneapi::math::diag diag,
                                   std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                   std::int64_t ldb);

template <typename T>
std::int64_t ungbr_scratchpad_size(sycl::queue& queue, oneapi::math::generate vect, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda);

template <typename T>
std::int64_t ungqr_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda);

template <typename T>
std::int64_t ungtr_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                                   std::int64_t lda);

template <typename T>
std::int64_t unmrq_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc);

template <typename T>
std::int64_t unmqr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda, std::int64_t ldc);

template <typename T>
std::int64_t unmtr_scratchpad_size(sycl::queue& queue, oneapi::math::side side,
                                   oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldc);

template <typename T>
std::int64_t getrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_ipiv, std::int64_t batch_size);

template <typename T>
std::int64_t getri_batch_scratchpad_size(sycl::queue& queue, std::int64_t n, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_ipiv,
                                         std::int64_t batch_size);

template <typename T>
std::int64_t getrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::transpose trans,
                                         std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_ipiv,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size);

template <typename T>
std::int64_t geqrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size);

template <typename T>
std::int64_t potrf_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                         std::int64_t n, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t batch_size);

template <typename T>
std::int64_t potrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo uplo,
                                         std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size);

template <typename T>
std::int64_t orgqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size);

template <typename T>
std::int64_t ungqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size);

template <typename T>
std::int64_t getrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes);

template <typename T>
std::int64_t getri_batch_scratchpad_size(sycl::queue& queue, std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes);

template <typename T>
std::int64_t getrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::transpose* trans,
                                         std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes);

template <typename T>
std::int64_t geqrf_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes);

template <typename T>
std::int64_t orgqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes);

template <typename T>
std::int64_t potrf_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo* uplo,
                                         std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes);

template <typename T>
std::int64_t potrs_batch_scratchpad_size(sycl::queue& queue, oneapi::math::uplo* uplo,
                                         std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes);

template <typename T>
std::int64_t ungqr_batch_scratchpad_size(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes);
