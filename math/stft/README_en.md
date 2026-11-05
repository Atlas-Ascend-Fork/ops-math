# Stft

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: computes the Fourier transform of the input within a sliding window.
- Computation formula:

  - When normalized=false:

    $$
    X[w,m]=\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft})
    $$

  - When normalized=true:

    $$
    X[w,m]=\frac{1}{\sqrt{nFft}}(\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft}))
    $$

  Here:

  - $w$ is the frequency point of the FFT.
  - $m$ is the index of the sliding window.
  - $self$ is a 1D or 2D Tensor. If $self$ is 1D, it is a time-series sampling sequence. If $self$ is 2D, it is multiple time-series sampling sequences.
  - $hopLength$ is the sliding window size.
  - $window$ is a 1D Tensor and the window function of the STFT (for example, hann_window), whose length is $winLength$.
  - $exp(-j*\frac{2{\pi}wk}{nFft})$ is the twiddle factor.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to `self` in the formula. It must be a 1D/2D Tensor with the shape [L]/[B, L], where L is the length of the time-series sampling sequence and B is the number of time-series sampling sequences. Inputs Inf, -Inf, and NaN are not supported.</td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>window</td>
      <td>Optional input</td>
      <td>It must be a 1D Tensor, corresponding to `window` in the formula. The shape is [winLength], where winLength is the length of the STFT window function, and the data type is the same as that of `x`.</td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hop_length</td>
      <td>Optional attribute</td>
      <td><ul><li>Spacing between sliding windows (greater than or equal to 0), corresponding to `hopLength` in the formula.</li><li>The default value is n_fft/4.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>win_length</td>
      <td>Optional attribute</td>
      <td><ul><li>Size of window (greater than or equal to 0), corresponding to `winLength` in the formula.</li><li>The default value is n_fft.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normalized</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether to normalize the Fourier transform result.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onesided</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether to return the full result or half of the result.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_complex</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether the return value is a complex Tensor or a Tensor with the real and imaginary parts separated.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>n_fft</td>
      <td>Attribute</td>
      <td>Number of FFT points (greater than 0), corresponding to `nFft` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result of the Fourier transform of `x` within `window`. It must be a 2D/3D/4D Tensor, corresponding to `X[w,m]` in the formula. If return_complex=True, y is a complex Tensor with the shape [N, T] or [B, N, T]. If return_complex=False, y is a real Tensor with the shape [N, T, 2] or [B, N, T, 2]. Here, N=n_fft(onesided=False) or (n_fft // 2 + 1)(onesided=True); T is the number of sliding windows, T = (L - n_fft) // hop_length + 1.</td>
      <td>FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- n_fft <= L.
- win_length <= n_fft.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_stft](examples/test_aclnn_stft.cpp) | Invokes the Stft operator through the [aclStft](docs/aclStft_en.md) API. |
| Graph mode | -  | Invokes the Stft operator by building a graph through the [operator IR](op_graph/stft_proto.h).         |
