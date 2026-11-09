#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


import numpy as np
import torch

__spec__ = {"rfft1_d": "Rfft1DTestSpec", "aclRfft1D": "Rfft1DTestSpec"}

MAX_DFT_LENGTH = 4096
MAX_FFT_LENGTH = 262144


def _short_dft(n, norm):
    # Match Rfft1DDftGen: aligned lengths intentionally get an extra block.
    rows = n + 16 - n % 16
    cols = n + 8 - n % 8
    frequencies = n // 2 + 1

    # Same association as the C++ generator: ((-2*pi * i) * j) / n.
    i = torch.arange(n, dtype=torch.float64, device="cpu")
    j = torch.arange(frequencies, dtype=torch.float64, device="cpu")

    angle = (-2.0 * torch.pi) * i[:, None] * j[None, :] / n

    scale = {1: 1.0, 2: 1.0 / n, 3: 1.0 / np.sqrt(n)}[norm]

    values = torch.empty((n, frequencies, 2), dtype=torch.float64, device="cpu")

    values[..., 0] = torch.cos(angle) * scale
    values[..., 1] = torch.sin(angle) * scale

    matrix = torch.zeros((rows, cols), dtype=torch.float32, device="cpu")

    if 0 < n % 16 <= 8:
        matrix.view(-1)[: values.numel()] = values.reshape(-1).to(torch.float32)

        # Return NumPy ndarray for compatibility with the test framework.
        return matrix.reshape(-1).numpy()

    matrix[:n, : 2 * frequencies] = values.reshape(n, -1).to(torch.float32)

    # NZ layout conversion:
    # (rows, cols)
    # -> (rows, cols // 8, 8)
    # -> (cols // 8, rows, 8)
    # -> contiguous flat buffer
    matrix = matrix.reshape(rows, cols // 8, 8).transpose(0, 1).contiguous().reshape(-1)

    # Return NumPy ndarray.
    return matrix.numpy()


def _fft_recursion(values, length):
    # Port of FftRec: recursive radix-2 DIT with e^(-2*pi*i*k/N) twiddles and
    # no normalization, matching the adapter's staging of the Bluestein beta
    # spectrum.
    if length <= 1:
        return values.copy()
    even = _fft_recursion(values[0::2], length // 2)
    odd = _fft_recursion(values[1::2], length // 2)
    twiddle = np.exp((-2j * np.pi / length) * np.arange(length // 2))
    product = twiddle * odd
    return np.concatenate((even + product, even - product))


def _calculate_factors(length):
    # Port of CalculateFactors: greedily split into at most three factors
    # from [2, 64] (keeping the first three), then fall back to Bluestein
    # over lengthPad = 2 * 2^ceil(log2(length)) when the split cannot rebuild
    # the length or the length is not a multiple of 64.
    def decompose(value):
        found = []
        for factor in range(64, 1, -1):
            while value % factor == 0:
                value //= factor
                found.append(factor)
        found += [1] * (3 - len(found))
        return found[:3]

    factors = [128, 64, 64] if length == 524288 else decompose(length)
    is_bluestein = length % 64 != 0 or factors[0] * factors[1] * factors[2] != length
    length_pad = length
    if is_bluestein:
        length_pad = 2 * (1 << (length - 1).bit_length())
        factors = [128, 64, 64] if length_pad == 524288 else decompose(length_pad)
    return factors, is_bluestein, length_pad


def _alpha_beta(n):
    # Port of GenerateAlphaBeta: the Bluestein staging buffers — the
    # conjugate chirp alpha, conj(beta) as duplicated-real and split-imag
    # buffers with the reversed tail, and the unnormalized FFT of the
    # wrapped beta sequence (DoubleRI keeps each spectrum half interleaved).
    length_pad = 2 * (1 << (n - 1).bit_length())
    # Chirp b_i = exp(i*pi*i^2/n); i^2 is reduced mod 2n (its period in the
    # exponent) so the angle stays small for large n.
    index = np.arange(n)
    chirp = np.exp(1j * np.pi * ((index * index) % (2 * n)) / n)

    alpha = np.empty(2 * n)  # alpha_i = conj(chirp_i)
    alpha[0::2] = chirp.real
    alpha[1::2] = -chirp.imag

    conj_real = np.zeros(2 * length_pad)
    conj_imag = np.zeros(2 * length_pad)
    conj_real[0 : 2 * n : 2] = chirp.real
    conj_real[1 : 2 * n : 2] = chirp.real
    conj_imag[0 : 2 * n : 2] = chirp.imag
    conj_imag[1 : 2 * n : 2] = -chirp.imag

    # The reversed tail reuses chirp[pow2 - i] for i in [pow2 - n + 1, pow2);
    # the gap in between stays zero.
    tail = np.arange(length_pad - n + 1, length_pad)
    wrapped = chirp[length_pad - tail]
    conj_real[2 * tail] = wrapped.real
    conj_real[2 * tail + 1] = wrapped.real
    conj_imag[2 * tail] = wrapped.imag
    conj_imag[2 * tail + 1] = -wrapped.imag

    beta = np.zeros(length_pad, dtype=np.complex128)
    beta[:n] = chirp
    beta[tail] = wrapped
    spectrum = _fft_recursion(beta, length_pad)

    spectrum_real = np.empty(2 * length_pad)  # (real, real) pass
    spectrum_real[0::2] = spectrum.real
    spectrum_real[1::2] = spectrum.real
    spectrum_imag = np.empty(2 * length_pad)  # (-imag, imag) pass
    spectrum_imag[0::2] = -spectrum.imag
    spectrum_imag[1::2] = spectrum.imag

    return np.concatenate((alpha, conj_real, conj_imag, spectrum_real, spectrum_imag))


def _terminator_dft(n, norm):
    """Port of GenerateTerminatorMatrix for n > 4096 (flat fp32 buffer).

    Layout consumed by the kernel (rfft1_d_colley_tukey.h / rfft1_d_bluestein.h):
    [dftReal | dftImag | twiddleReal | twiddleImag] and, for Bluestein, a
    mirrored back half (twiddleReal written twice) plus the alpha/beta tail.
    """
    factors, is_bluestein, _ = _calculate_factors(n)

    norm_param = 1.0
    if norm != 1:
        norm_param = 1.0 / n if norm == 2 else 1.0 / np.sqrt(n)
    # Bluestein inverse-FFT scale: 2 * 2^ceil(log2(n)).
    inv_scale = 2.0 * (1 << (n - 1).bit_length())

    dft_real, dft_imag = [], []
    dft_real_back, dft_imag_back = [], []
    twiddle_real, twiddle_imag, twiddle_imag_back = [], [], []

    prev_factors = 1
    for cur_index, cur_factor in enumerate(factors):
        rows_num = cur_factor * (2 if cur_index == 0 and is_bluestein else 1)
        cols_num = 2 * cur_factor if cur_index == 0 else cur_factor

        if cur_factor != 1:
            rows = np.arange(cur_factor)[:, None]
            cols = np.arange(cur_factor)[None, :]
            theta = -2.0 * np.pi * rows * cols / cur_factor

            if cur_index == 0:
                # Stage 0 keeps the radix DFT matrix with interleaved
                # (real, imag) columns; Bluestein appends a mirrored half
                # used by the inverse pass. norm is folded in here (and only
                # here), the back half carries the inverse-FFT scale.
                forward = np.empty((cur_factor, 2 * cur_factor))
                forward[:, 0::2] = np.cos(theta)
                forward[:, 1::2] = np.sin(theta)
                backward = np.empty((cur_factor, 2 * cur_factor))
                backward[:, 0::2] = np.cos(theta)
                backward[:, 1::2] = -np.sin(theta)
                real_cur = np.zeros(rows_num * cols_num)
                real_back = np.zeros(rows_num * cols_num)
                real_cur[: forward.size] = forward.ravel()
                real_back[: backward.size] = backward.ravel()
                if is_bluestein:
                    mirror = np.empty((cur_factor, 2 * cur_factor))
                    mirror[:, 0::2] = -np.sin(theta)
                    mirror[:, 1::2] = np.cos(theta)
                    mirror_back = np.empty((cur_factor, 2 * cur_factor))
                    mirror_back[:, 0::2] = np.sin(theta)
                    mirror_back[:, 1::2] = np.cos(theta)
                    real_cur[cur_factor * cols_num :] = mirror.ravel()
                    real_back[cur_factor * cols_num :] = mirror_back.ravel()
                dft_real.append(real_cur * norm_param)
                dft_real_back.append(real_back / inv_scale)
            else:
                dft_real.append(np.cos(theta).ravel())
                dft_imag.append(np.sin(theta).ravel())
                dft_real_back.append(np.cos(theta).ravel())
                dft_imag_back.append(-np.sin(theta).ravel())

                # Twiddles: two slots per (row, prev factor) holding the
                # (cos, cos), (-sin, sin) and (sin, -sin) triplets.
                tw_cols = np.arange(prev_factors)[None, :]
                tw_theta = -2.0 * np.pi * rows * tw_cols / (cur_factor * prev_factors)
                twiddle_real.append(np.repeat(np.cos(tw_theta), 2, axis=1).ravel())
                twiddle_imag.append(
                    np.stack((-np.sin(tw_theta), np.sin(tw_theta)), axis=-1).ravel()
                )
                twiddle_imag_back.append(
                    np.stack((np.sin(tw_theta), -np.sin(tw_theta)), axis=-1).ravel()
                )
        prev_factors *= cur_factor

    parts = [
        np.concatenate(dft_real),
        np.concatenate(dft_imag),
        np.concatenate(twiddle_real),
        np.concatenate(twiddle_imag),
    ]
    if is_bluestein:
        parts += [
            np.concatenate(dft_real_back),
            np.concatenate(dft_imag_back),
            np.concatenate(twiddle_real),
            np.concatenate(twiddle_imag_back),
            _alpha_beta(n),
        ]
    return np.concatenate(parts).astype(np.float32)


def _is_aclnn_call(rest):
    # aclnn passes C-declaration order (self, n, dim, norm, out):
    # the second positional argument is integer n.
    #
    # Kernel mode passes (x, dft):
    # the second positional argument is the coefficient buffer.
    return (
        bool(rest)
        and isinstance(rest[0], (int, np.integer))
        and not isinstance(rest[0], (bool, np.bool_))
    )


class Rfft1DTestSpec:
    """Kernel: generate the dft input; ACLNN: plain random input; golden: float64 FFT."""

    def customize_inputs(x, *rest, **kwargs):
        if _is_aclnn_call(rest):
            # ACLNN generates the DFT matrix internally.
            # Keep framework-generated inputs unchanged.
            return None

        # Kernel mode: (x, dft)
        (dft,) = rest

        n = int(kwargs["n"])
        norm = int(kwargs["norm"])

        if x.shape[-1] != n:
            raise ValueError("Kernel Rfft1D requires x.shape[-1] == n")

        if norm not in (1, 2, 3):
            raise ValueError("norm must be 1 (backward), 2 (forward), or 3 (ortho)")

        if not 1 <= n <= MAX_FFT_LENGTH:
            raise ValueError(
                f"Kernel Rfft1D requires 1 <= n <= {MAX_FFT_LENGTH}, got {n}"
            )

        # n <= 4096 feeds the plain DFT matrix (FastDFT path); larger n feeds
        # the Cooley-Tukey / Bluestein terminator matrices.
        coefficients = (
            _short_dft(n, norm) if n <= MAX_DFT_LENGTH else _terminator_dft(n, norm)
        )

        # dft is a NumPy ndarray generated by the test framework.
        if coefficients.size != dft.size:
            raise ValueError(
                f"dft buffer has {dft.size} elements; expected {coefficients.size}"
            )

        # Keep the framework input type as NumPy ndarray.
        return (x, coefficients.reshape(dft.shape))

    def golden(x, *rest, **kwargs):
        if _is_aclnn_call(rest):
            # ACLNN mode: (x, n, dim, norm, out)
            n = int(rest[0])
            dim = int(rest[1])
            norm = int(rest[2])

        else:
            # Kernel mode: (x, dft), n/norm are keyword attributes.
            (dft,) = rest

            n = int(kwargs["n"])
            norm = int(kwargs["norm"])
            dim = -1

        normalization = {1: "backward", 2: "forward", 3: "ortho"}[norm]

        # Convert NumPy input to PyTorch float64 Tensor.
        x_tensor = torch.as_tensor(x, dtype=torch.float64, device="cpu")

        # Calculate independent FFT reference using PyTorch.
        spectrum = torch.fft.rfft(x_tensor, n=n, dim=dim, norm=normalization)

        # Stack real and imaginary components.
        # Output shape: (..., n // 2 + 1, 2)
        output = torch.stack((spectrum.real, spectrum.imag), dim=-1).to(torch.float32)

        # Convert PyTorch Tensor to NumPy ndarray.
        return [output.numpy()]

    tolerance = {"float32": {"standard": "mix_tolerance"}}
