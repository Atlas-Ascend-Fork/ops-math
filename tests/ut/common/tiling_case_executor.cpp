/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_case_executor.h"
#include <cstring>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)
static void GetPlatFormInfos(const char* compileInfoStr, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                             map<string, string>& intrinsics);

namespace {
// 清零当前栈指针以下的一段区间，抹掉此前调用及其被调函数在栈上留下的残留，
// 避免 tiling 的栈帧复用这些残留，导致 tilingData 中结构体 padding 不可预期。
// 必须放在独立 noinline 函数里用 alloca：alloca 缓冲位于本函数栈帧之下，
// 本函数返回后这段空间被释放，正好是紧接其后 tiling 调用会复用的区间。
constexpr size_t TILING_STACK_SCRUB_BYTES = 32 * 1024;

__attribute__((noinline)) void ScrubReleasedStack()
{
    auto* buf = static_cast<uint8_t*>(__builtin_alloca(TILING_STACK_SCRUB_BYTES));
    std::memset(buf, 0, TILING_STACK_SCRUB_BYTES);
    // 内存屏障：避免 memset 被当作对即将失效的栈内存的无效写而优化掉
    asm volatile("" : : "r"(buf) : "memory");
}
} // namespace

/**
 * @brief 构造 tiling 执行上下文（张量/属性/平台信息/TilingData/Workspace 等）
 *
 * 构造完成后由 RunTiling() 执行 tiling 函数。返回的 bundle 持有 context 依赖的全部资源，
 * 生命周期需覆盖 RunTiling 调用与后续断言。
 */
TilingContextBundle BuildTilingContext(const gert::TilingContextPara& tilingContextPara, bool withWorkspace)
{
    TilingContextBundle bundle;
    auto contextFaker = gert::TilingContextFaker();

    // 1. input/output information
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();
    if (tilingContextPara.inputInstanceNum_.size() != 0 || tilingContextPara.outputInstanceNum_.size() != 0) {
        contextFaker.IrInstanceNum(tilingContextPara.inputInstanceNum_, tilingContextPara.outputInstanceNum_);
    } else {
        contextFaker.NodeIoNum(inputNum, outputNum);
    }
    std::vector<gert::Tensor*> inputTensors = {};
    std::vector<gert::Tensor*> outputTensors = {};
    for (size_t index = 0; index < inputNum; index++) {
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
            tilingContextPara.inputTensorDesc_[index].shape_,
            gert::StorageFormat(tilingContextPara.inputTensorDesc_[index].format_,
                                tilingContextPara.inputTensorDesc_[index].format_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, tilingContextPara.inputTensorDesc_[index].dtype_,
            tilingContextPara.inputTensorDesc_[index].isConst_ ? tilingContextPara.inputTensorDesc_[index].constValue_ :
                                                                 nullptr);
        inputTensors.push_back(curTensor.get());
        bundle.inputTensors.push_back(std::move(curTensor));
    }
    for (size_t index = 0; index < outputNum; index++) {
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
            tilingContextPara.outputTensorDesc_[index].shape_,
            gert::StorageFormat(tilingContextPara.outputTensorDesc_[index].format_,
                                tilingContextPara.outputTensorDesc_[index].format_, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, tilingContextPara.outputTensorDesc_[index].dtype_,
            tilingContextPara.outputTensorDesc_[index].isConst_ ?
                tilingContextPara.outputTensorDesc_[index].constValue_ :
                nullptr);
        outputTensors.push_back(curTensor.get());
        bundle.outputTensors.push_back(std::move(curTensor));
    }
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);
    for (auto& attrInfo : tilingContextPara.attrs_) {
        switch (attrInfo.attr_.type_) {
            case Ops::Math::AnyValue::ValueType::VT_BOOL: {
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_INT: {
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_FLOAT: {
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_STRING: {
                contextFaker.Attr(
                    attrInfo.attrName_,
                    ge::AscendString(reinterpret_cast<std::string*>(attrInfo.attr_.valuePtr_.get())->c_str()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_LIST_BOOL: {
                contextFaker.Attr(attrInfo.attrName_,
                                  *reinterpret_cast<std::vector<bool>*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_LIST_INT: {
                contextFaker.Attr(attrInfo.attrName_,
                                  *reinterpret_cast<std::vector<int64_t>*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_LIST_LIST_INT: {
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>>*>(
                                                          attrInfo.attr_.valuePtr_.get()));
                break;
            }
            case Ops::Math::AnyValue::ValueType::VT_LIST_FLOAT: {
                contextFaker.Attr(attrInfo.attrName_,
                                  *reinterpret_cast<std::vector<float>*>(attrInfo.attr_.valuePtr_.get()));
                break;
            }
            default:
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_
                          << "is not supported!" << std::endl;
        }
    }

    // 2. base information
    bundle.platformInfo = std::make_unique<fe::PlatFormInfos>();
    bundle.platformInfo->Init();
    bundle.tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);
    bundle.workspace = gert::ContinuousVector::Create<size_t>(4096);
    auto& fakerWithBaseInfo = contextFaker.SetOpType(tilingContextPara.opName_.c_str())
                                  .CompileInfo(tilingContextPara.compileInfo_)
                                  .PlatformInfo(reinterpret_cast<char*>(bundle.platformInfo.get()))
                                  .TilingData(bundle.tilingData.get());
    if (withWorkspace) {
        fakerWithBaseInfo.Workspace(reinterpret_cast<gert::ContinuousVector*>(bundle.workspace.get()));
    }
    auto contextHolder = fakerWithBaseInfo.Build();
    string compileInfoStringPrefix =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )";
    string compileInfoStringMiddle =
        R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )";
    map<string, string> socToUpper = {{"ascend910b", "Ascend910B"},       {"ascend910_93", "Ascend910_93"},
                                      {"ascend950", "Ascend950"},         {"ascend310p", "Ascend310P"},
                                      {"ascend910", "Ascend910"},         {"ascend310b", "Ascend310B"},
                                      {"ascend610lite", "Ascend610Lite"}, {"ascend031", "Ascend031"},
                                      {"ascend035", "Ascend035"},         {"kirinx90", "KrinX90"},
                                      {"kirin9030", "Kirin9030"},         {"mc62", "MC62"},
                                      {"ascend5162", "Ascend5162"}};
    std::string buildSocVersion = STR(BUILD_SOC_VERSION);
    if (!buildSocVersion.empty()) {
        buildSocVersion = socToUpper[buildSocVersion];
    }
    string compileInfoStringSuffix = R"(, "socVersion":)"
                                     R"(")" +
                                     buildSocVersion + R"("} })";
    string compileInfoString = compileInfoStringPrefix + std::to_string(tilingContextPara.ubSize_) +
                               compileInfoStringMiddle + std::to_string(tilingContextPara.coreNum_) +
                               compileInfoStringSuffix;
    map<string, string> socToArch = {{"Ascend310P", "2002"},
                                     {"Ascend910B", "2201"},
                                     {"Ascend910_93", "2201"},
                                     {"Ascend950", "3510"},
                                     {"Ascend910", "1001"}};
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"NpuArch", socToArch[buildSocVersion]}, {"Short_SoC_version", buildSocVersion}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    bundle.context = contextHolder.GetContext();
    bundle.context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    bundle.context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    bundle.context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    bundle.context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    bundle.context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    bundle.holder = std::move(contextHolder);
    return bundle;
}

/**
 * @brief 执行 tiling 函数（查找 op tiling 实现并调用）
 */
ge::graphStatus RunTiling(TilingContextBundle& bundle, const gert::TilingContextPara& tilingContextPara)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr) {
        throw std::invalid_argument("not found spaceRegistry");
    }
    auto functionStruct = spaceRegistry->GetOpImpl(tilingContextPara.opName_.c_str());
    if (functionStruct == nullptr) {
        throw std::invalid_argument("not found " + tilingContextPara.opName_);
    }
    auto tilingFunc = functionStruct->tiling;
    // 执行 tiling 前先清零它将要使用的栈区间，抹掉此前调用（如 BuildTilingContext
    // 及其被调函数）留在栈上的残值，避免 tilingData 中结构体 padding 不可预期。
    ScrubReleasedStack();
    return tilingFunc(bundle.context);
}

template <typename T>
static string to_string(void* buf, size_t size)
{
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static void GetPlatFormInfos(const char* compileInfoStr, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                             map<string, string>& intrinsics)
{
    string default_hardward_info = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,
                          "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "CORE_NUM": 32}})";
    nlohmann::json compileInfoJson = nlohmann::json::parse(compileInfoStr);
    if (compileInfoJson.type() != nlohmann::json::value_t::object) {
        compileInfoJson = nlohmann::json::parse(default_hardward_info.c_str());
    }

    map<string, string> socInfoKeys = {{"ai_core_cnt", "CORE_NUM"},
                                       {"l2_size", "L2_SIZE"},
                                       {"cube_core_cnt", "cube_core_cnt"},
                                       {"vector_core_cnt", "vector_core_cnt"},
                                       {"core_type_list", "core_type_list"}};
    socInfos["core_type_list"] = "AICore";

    for (auto& t : socInfoKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            auto& objJson = compileInfoJson["hardware_info"][t.second];
            if (objJson.is_number_integer()) {
                socInfos[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            } else if (objJson.is_string()) {
                socInfos[t.first] = objJson;
            }
        }
    }
    map<string, string> aicoreSpecKeys = {{"ub_size", "UB_SIZE"},
                                          {"l0_a_size", "L0A_SIZE"},
                                          {"l0_b_size", "L0B_SIZE"},
                                          {"l0_c_size", "L0C_SIZE"},
                                          {"l1_size", "L1_SIZE"},
                                          {"bt_size", "BT_SIZE"},
                                          {"load3d_constraints", "load3d_constraints"}};
    aicoreSpec["cube_freq"] = "cube_freq";
    for (auto& t : aicoreSpecKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(t.second)) {
            if (t.second == "load3d_constraints") {
                aicoreSpec[t.first] = compileInfoJson["hardware_info"][t.second].get<string>();
            } else {
                aicoreSpec[t.first] = to_string(compileInfoJson["hardware_info"][t.second].get<uint32_t>());
            }
        }
    }

    std::string intrinsicsKeys[] = {"Intrinsic_data_move_l12ub",  "Intrinsic_data_move_l0c2ub",
                                    "Intrinsic_fix_pipe_l0c2out", "Intrinsic_data_move_out2l1_nd2nz",
                                    "Intrinsic_matmul_ub_to_ub",  "Intrinsic_conv_ub_to_ub",
                                    "Intrinsic_data_move_l12bt"};
    for (string key : intrinsicsKeys) {
        if (compileInfoJson.contains("hardware_info") && compileInfoJson["hardware_info"].contains(key) &&
            compileInfoJson["hardware_info"][key].get<bool>()) {
            intrinsics[key] = "float16";
            if (key.find("Intrinsic_data_move_l12bt") != string::npos) {
                intrinsics[key] = "bf16";
            }
        }
    }
}

void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult,
                     uint64_t expectTilingKey, const string& expectTilingData,
                     const std::vector<size_t>& expectWorkspaces)
{
    auto bundle = BuildTilingContext(tilingContextPara);
    auto tilingRet = RunTiling(bundle, tilingContextPara);
    auto* tilingContext = bundle.context;

    // check tiling func
    EXPECT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        ASSERT_EQ(workspaceCount, expectWorkspaces.size());
        auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
        }
    }

    // check tiling key
    auto tilingKeyResult = tilingContext->GetTilingKey();
    ASSERT_EQ(tilingKeyResult, expectTilingKey);

    // check tiling data
    if (expectTilingData == EMPTY_EXPECT_TILING_DATA) {
        return;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult,
                     uint64_t expectTilingKey, const std::vector<size_t>& expectWorkspaces)
{
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}

bool ExecuteTiling(const gert::TilingContextPara& tilingContextPara, TilingInfo& tilingInfo)
{
    auto bundle = BuildTilingContext(tilingContextPara);
    auto tilingRet = RunTiling(bundle, tilingContextPara);
    auto* tilingContext = bundle.context;

    if (tilingRet != ge::GRAPH_SUCCESS) {
        return false;
    }

    tilingInfo.tilingKey = tilingContext->GetTilingKey();
    tilingInfo.blockNum = tilingContext->GetBlockDim();
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        auto workSpaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            tilingInfo.workspaceSizes.push_back(workSpaceSizes[i]);
        }
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    tilingInfo.tilingData = std::make_unique<uint8_t[]>(rawTilingData->GetDataSize());
    tilingInfo.tilingDataSize = rawTilingData->GetDataSize();
    std::memcpy(tilingInfo.tilingData.get(), rawTilingData->GetData(), rawTilingData->GetDataSize());

    return true;
}

/* EleBaseTilingData struct format
    {
        int64_t dim0;
        int32_t coreNum;
        int32_t ubFormer;
        int64_t blockFormer;
        int64_t blockNum;
        int64_t ubLoopOfFormerBlock;
        int64_t ubLoopOfTailBlock;
        int64_t ubTailOfFormerBlock;
        int64_t ubTailOfTailBlock;
        int64_t elemNum;
        uint64_t scheMode;
    };
*/
static string eleToString(void* buf)
{
    string result;
    const int64_t* dataInt64 = reinterpret_cast<const int64_t*>(buf);
    result += std::to_string(dataInt64[0]);
    result += " ";

    const int32_t* dataInt32 = reinterpret_cast<const int32_t*>(buf + 8);
    result += std::to_string(dataInt32[0]);
    result += " ";
    result += std::to_string(dataInt32[1]);
    result += " ";

    dataInt64 = reinterpret_cast<const int64_t*>(buf + 16);
    for (size_t i = 0; i < 7; i++) {
        result += std::to_string(dataInt64[i]);
        result += " ";
    }

    const uint64_t* dataUint64 = reinterpret_cast<const uint64_t*>(buf + 72);
    result += std::to_string(dataUint64[0]);
    result += " ";

    return result;
}

void ExecuteTestCaseForEle(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult,
                           bool needCheckTilingKey, uint64_t expectTilingKey, bool needCheckTilingData,
                           const string& expectTilingData, const std::vector<size_t>& expectWorkspaces)
{
    auto bundle = BuildTilingContext(tilingContextPara);
    auto tilingRet = RunTiling(bundle, tilingContextPara);
    auto* tilingContext = bundle.context;

    // check tiling func
    EXPECT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
        }
    }

    // check tiling key
    if (needCheckTilingKey) {
        auto tilingKeyResult = tilingContext->GetTilingKey();
        ASSERT_EQ(tilingKeyResult, expectTilingKey);
    }

    // check tiling data
    if (needCheckTilingData) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        auto tilingDataResult = eleToString(rawTilingData->GetData());
        EXPECT_EQ(tilingDataResult, expectTilingData);
    }
}
