#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  const int iMax = std::min(A.nbRows, iRowBlkA + szBlock);
  const int jMax = std::min(B.nbCols, iColBlkB + szBlock);
  const int kMax = std::min(A.nbCols, iColBlkA + szBlock);

  const char* env = std::getenv("LOOP_ORDER");
  const std::string order = env ? std::string(env) : "jki";

  if (order == "ijk") {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = iRowBlkA; i < iMax; ++i)
      for (int j = iColBlkB; j < jMax; ++j)
        for (int k = iColBlkA; k < kMax; ++k)
          C(i, j) += A(i, k) * B(k, j);
  } else if (order == "jik") {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = iColBlkB; j < jMax; ++j)
      for (int i = iRowBlkA; i < iMax; ++i)
        for (int k = iColBlkA; k < kMax; ++k)
          C(i, j) += A(i, k) * B(k, j);
  } else if (order == "ikj") {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = iRowBlkA; i < iMax; ++i)
      for (int k = iColBlkA; k < kMax; ++k)
        for (int j = iColBlkB; j < jMax; ++j)
          C(i, j) += A(i, k) * B(k, j);
  } else if (order == "kij") {
    for (int k = iColBlkA; k < kMax; ++k)
      for (int i = iRowBlkA; i < iMax; ++i)
        for (int j = iColBlkB; j < jMax; ++j)
          C(i, j) += A(i, k) * B(k, j);
  } else if (order == "kji") {
    for (int k = iColBlkA; k < kMax; ++k)
      for (int j = iColBlkB; j < jMax; ++j)
        for (int i = iRowBlkA; i < iMax; ++i)
          C(i, j) += A(i, k) * B(k, j);
  } else {  // "jki" default
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = iColBlkB; j < jMax; ++j)
      for (int k = iColBlkA; k < kMax; ++k)
        for (int i = iRowBlkA; i < iMax; ++i)
          C(i, j) += A(i, k) * B(k, j);
  }
}
int getBlockSize() {
  const char* env = std::getenv("BLOCK_SIZE");
  if (env) {
    int v = std::atoi(env);
    if (v > 0) return v;
  }
  return 32;
}
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  const int szBlock = getBlockSize();
  for (int i = 0; i < A.nbRows; i += szBlock)
    for (int j = 0; j < B.nbCols; j += szBlock)
      for (int k = 0; k < A.nbCols; k += szBlock)
        prodSubBlocks(i, j, k, szBlock, A, B, C);
  return C;
}
