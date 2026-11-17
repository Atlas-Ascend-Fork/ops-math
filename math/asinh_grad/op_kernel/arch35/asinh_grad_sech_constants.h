/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Exact certified constants from reviewed constants.json
// SHA21d1706cf20d6449f133a47d4bb429199d85ff6d3e84bd172df1236cf225b90b.
#ifndef ASINH_GRAD_SECH_CONSTANTS_H
#define ASINH_GRAD_SECH_CONSTANTS_H
namespace NsAsinhGradCertifiedSech {
static constexpr float SECH_C0 = 0x1.6280000000000p-7f;   // 0x3c314000
static constexpr float SECH_C1 = 0x1.90bfbe0000000p-17f;  // 0x37485fdf
static constexpr float SECH_C2 = 0x1.1cf79a0000000p-42f;  // 0x2a8e7bcd
static constexpr float SECH_BETA = 0x1.7154760000000p+6f; // 0x42b8aa3b
static constexpr float SECH_C3 = 0x1.5555560000000p-3f;   // 0x3e2aaaab
static constexpr float SECH_C4 = 0x1.5555560000000p-5f;   // 0x3d2aaaab
static constexpr float SECH_C5 = 0x1.1111120000000p-7f;   // 0x3c088889
static constexpr float SECH_RADIUS = 0x1p-41f;
__aicore__ inline void InitializeSechTable(AscendC::LocalTensor<uint32_t>& table)
{
    table.SetValue(0, 0x3f800000U);
    table.SetValue(1, 0x3f8164d2U);
    table.SetValue(2, 0x3f82cd87U);
    table.SetValue(3, 0x3f843a29U);
    table.SetValue(4, 0x3f85aac3U);
    table.SetValue(5, 0x3f871f62U);
    table.SetValue(6, 0x3f88980fU);
    table.SetValue(7, 0x3f8a14d5U);
    table.SetValue(8, 0x3f8b95c2U);
    table.SetValue(9, 0x3f8d1adfU);
    table.SetValue(10, 0x3f8ea43aU);
    table.SetValue(11, 0x3f9031dcU);
    table.SetValue(12, 0x3f91c3d3U);
    table.SetValue(13, 0x3f935a2bU);
    table.SetValue(14, 0x3f94f4f0U);
    table.SetValue(15, 0x3f96942dU);
    table.SetValue(16, 0x3f9837f0U);
    table.SetValue(17, 0x3f99e046U);
    table.SetValue(18, 0x3f9b8d3aU);
    table.SetValue(19, 0x3f9d3edaU);
    table.SetValue(20, 0x3f9ef532U);
    table.SetValue(21, 0x3fa0b051U);
    table.SetValue(22, 0x3fa27043U);
    table.SetValue(23, 0x3fa43516U);
    table.SetValue(24, 0x3fa5fed7U);
    table.SetValue(25, 0x3fa7cd94U);
    table.SetValue(26, 0x3fa9a15bU);
    table.SetValue(27, 0x3fab7a3aU);
    table.SetValue(28, 0x3fad583fU);
    table.SetValue(29, 0x3faf3b79U);
    table.SetValue(30, 0x3fb123f6U);
    table.SetValue(31, 0x3fb311c4U);
    table.SetValue(32, 0x3fb504f3U);
    table.SetValue(33, 0x3fb6fd92U);
    table.SetValue(34, 0x3fb8fbafU);
    table.SetValue(35, 0x3fbaff5bU);
    table.SetValue(36, 0x3fbd08a4U);
    table.SetValue(37, 0x3fbf179aU);
    table.SetValue(38, 0x3fc12c4dU);
    table.SetValue(39, 0x3fc346cdU);
    table.SetValue(40, 0x3fc5672aU);
    table.SetValue(41, 0x3fc78d75U);
    table.SetValue(42, 0x3fc9b9beU);
    table.SetValue(43, 0x3fcbec15U);
    table.SetValue(44, 0x3fce248cU);
    table.SetValue(45, 0x3fd06334U);
    table.SetValue(46, 0x3fd2a81eU);
    table.SetValue(47, 0x3fd4f35bU);
    table.SetValue(48, 0x3fd744fdU);
    table.SetValue(49, 0x3fd99d16U);
    table.SetValue(50, 0x3fdbfbb8U);
    table.SetValue(51, 0x3fde60f5U);
    table.SetValue(52, 0x3fe0ccdfU);
    table.SetValue(53, 0x3fe33f89U);
    table.SetValue(54, 0x3fe5b907U);
    table.SetValue(55, 0x3fe8396aU);
    table.SetValue(56, 0x3feac0c7U);
    table.SetValue(57, 0x3fed4f30U);
    table.SetValue(58, 0x3fefe4baU);
    table.SetValue(59, 0x3ff28177U);
    table.SetValue(60, 0x3ff5257dU);
    table.SetValue(61, 0x3ff7d0dfU);
    table.SetValue(62, 0x3ffa83b3U);
    table.SetValue(63, 0x3ffd3e0cU);
    table.SetValue(64, 0x00000000U);
    table.SetValue(65, 0xb1c43fd0U);
    table.SetValue(66, 0xb34ea7a9U);
    table.SetValue(67, 0xb2f14c87U);
    table.SetValue(68, 0x334f9891U);
    table.SetValue(69, 0xb352c2e6U);
    table.SetValue(70, 0xb37eda4bU);
    table.SetValue(71, 0x336a92deU);
    table.SetValue(72, 0xb260aba1U);
    table.SetValue(73, 0x3336fcb7U);
    table.SetValue(74, 0xb3697465U);
    table.SetValue(75, 0x330628cdU);
    table.SetValue(76, 0x33675624U);
    table.SetValue(77, 0x32bc4f9cU);
    table.SetValue(78, 0xb32e0212U);
    table.SetValue(79, 0x32dc8061U);
    table.SetValue(80, 0x33231b71U);
    table.SetValue(81, 0xb359be90U);
    table.SetValue(82, 0xb30c5563U);
    table.SetValue(83, 0xb331a601U);
    table.SetValue(84, 0x33412342U);
    table.SetValue(85, 0x31fb9715U);
    table.SetValue(86, 0x30c3125aU);
    table.SetValue(87, 0xb323ec33U);
    table.SetValue(88, 0xb32c9d5eU);
    table.SetValue(89, 0xb3162d36U);
    table.SetValue(90, 0xb3162b08U);
    table.SetValue(91, 0xb314ad82U);
    table.SetValue(92, 0xb22deaf6U);
    table.SetValue(93, 0xb3252debU);
    table.SetValue(94, 0xb37c5aa8U);
    table.SetValue(95, 0x32154889U);
    table.SetValue(96, 0x32cfe77aU);
    table.SetValue(97, 0xb266b974U);
    table.SetValue(98, 0x330ec5f7U);
    table.SetValue(99, 0xb31bd983U);
    table.SetValue(100, 0xb3414fe8U);
    table.SetValue(101, 0xb3130b1aU);
    table.SetValue(102, 0xb2d6663eU);
    table.SetValue(103, 0xb2976da2U);
    table.SetValue(104, 0x320aa837U);
    table.SetValue(105, 0xb2dd5119U);
    table.SetValue(106, 0xb37323a2U);
    table.SetValue(107, 0xb006c6c2U);
    table.SetValue(108, 0x3228fc24U);
    table.SetValue(109, 0xb2944353U);
    table.SetValue(110, 0xb35c1daaU);
    table.SetValue(111, 0xb3286024U);
    table.SetValue(112, 0xb2d4a58aU);
    table.SetValue(113, 0xb2f61d41U);
    table.SetValue(114, 0xb3504a1cU);
    table.SetValue(115, 0xb37b43e3U);
    table.SetValue(116, 0xb21eab59U);
    table.SetValue(117, 0x33657d15U);
    table.SetValue(118, 0xb2441be6U);
    table.SetValue(119, 0x33207898U);
    table.SetValue(120, 0xb24116deU);
    table.SetValue(121, 0x3276cca1U);
    table.SetValue(122, 0xb348464aU);
    table.SetValue(123, 0x32f167ffU);
    table.SetValue(124, 0x32292436U);
    table.SetValue(125, 0x336615a2U);
    table.SetValue(126, 0xb2923758U);
    table.SetValue(127, 0x31cf486cU);
}
} // namespace NsAsinhGradCertifiedSech
#endif
