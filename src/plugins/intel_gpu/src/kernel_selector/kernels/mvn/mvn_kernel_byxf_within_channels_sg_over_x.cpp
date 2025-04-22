// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_byxf_within_channels_sg_over_x.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr char ACC_TYPE_NAME[] = "float";
static constexpr size_t ACC_TYPE_SIZE = 4;
static constexpr size_t SLM_ELEMS_PER_WI_DATA_ELEM = 2;

ParamsKey MVNKernelByxfWithinChannelsSubgroupOverX::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNNormalizeVariance();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

size_t MVNKernelByxfWithinChannelsSubgroupOverX::GetDatasetSize(const mvn_params& params) const {
    const auto& input = params.inputs[0];
    return input.X().v * input.Y().v;
}

size_t MVNKernelByxfWithinChannelsSubgroupOverX::GetSubgroupSize(const mvn_params& params) const {
    const auto& input = params.inputs[0];
    size_t subgroup_size = 0;
    for (auto i = 0; i < params.engineInfo.supportedSimdSizes.size(); i++) {
        auto& tested_subgroup_size = params.engineInfo.supportedSimdSizes[i];
        if ((tested_subgroup_size > subgroup_size) && (input.X().v % tested_subgroup_size == 0))
            subgroup_size = tested_subgroup_size;
    }
    return subgroup_size;
}

size_t MVNKernelByxfWithinChannelsSubgroupOverX::GetNumSubgroupsPerWorkgroup(const mvn_params& params) const {
    const auto& input = params.inputs[0];
    const auto& dataset_size = GetDatasetSize(params);
    const auto& subgroup_size = GetSubgroupSize(params);
    size_t num_subgroups_per_workgroup = 1;

    while ((2 * num_subgroups_per_workgroup * subgroup_size <= params.engineInfo.maxWorkGroupSize) && 
        (dataset_size % (2 * num_subgroups_per_workgroup * subgroup_size) == 0))
        num_subgroups_per_workgroup *= 2;

    return num_subgroups_per_workgroup;
}

MVNKernelByxfWithinChannelsSubgroupOverX::tile_config MVNKernelByxfWithinChannelsSubgroupOverX::GetTileConfig(
    const mvn_params& params) const {
    const auto& input = params.inputs[0];

    const auto& dataset_size = GetDatasetSize(params);
    const auto& subgroup_size = GetSubgroupSize(params);
    const auto& num_sg_per_wg = GetNumSubgroupsPerWorkgroup(params);

    tile_config tile_cfg;

    tile_cfg.tile_size = dataset_size / (num_sg_per_wg * subgroup_size);
    tile_cfg.tile_size_x = std::min(tile_cfg.tile_size, input.X().v / subgroup_size);
    tile_cfg.tile_size_y = tile_cfg.tile_size / tile_cfg.tile_size_x;

    return tile_cfg;
}

JitConstants MVNKernelByxfWithinChannelsSubgroupOverX::GetJitConstants(const mvn_params& params,
                                                const MVNKernelBase::DispatchData dispatchData) const {
    const auto& input_dt = params.inputs[0].GetDType();

    const auto& input = params.inputs[0];

    const auto& dataset_size = GetDatasetSize(params);
    const auto& subgroup_size = GetSubgroupSize(params);
    const auto& num_sg_per_wg = GetNumSubgroupsPerWorkgroup(params);
    const auto& tile_cfg = GetTileConfig(params);

    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstants({
        MakeJitConstant("DATA_SET_SIZE", dataset_size),
        MakeJitConstant("SG_SIZE", subgroup_size),
        MakeJitConstant("NUM_SG_PER_WG", num_sg_per_wg),
        MakeJitConstant("WG_SIZE", num_sg_per_wg * subgroup_size),
        MakeJitConstant("TILE_SIZE", tile_cfg.tile_size),
        MakeJitConstant("TILE_SIZE_X", tile_cfg.tile_size_x),
        MakeJitConstant("TILE_SIZE_Y", tile_cfg.tile_size_y),
        MakeJitConstant("ACCUMULATOR_TYPE", ACC_TYPE_NAME),
        MakeJitConstant("ACCUMULATOR_TYPE_SIZE", ACC_TYPE_SIZE),
    });

    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    
    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"",
                                      {"b", "f", "y_start + y", "x_start + x"},
                                      "out_val",
                                      input_dt,
                                      1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    
    return jit;
}

MVNKernelByxfWithinChannelsSubgroupOverX::Parent::DispatchData MVNKernelByxfWithinChannelsSubgroupOverX::SetDefault(const mvn_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);

    const auto& input = params.inputs[0];

    const auto& dataset_size = GetDatasetSize(params);
    const auto& subgroup_size = GetSubgroupSize(params);
    const auto& num_sg_per_wg = GetNumSubgroupsPerWorkgroup(params);
    const auto& tile_cfg = GetTileConfig(params);

    dispatchData.gws[0] = input.Batch().v;
    dispatchData.gws[1] = num_sg_per_wg * subgroup_size;
    dispatchData.gws[2] = input.Feature().v;    

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = dispatchData.gws[1];
    dispatchData.lws[2] = 1;

    dispatchData.maxSlmSize = SLM_ELEMS_PER_WI_DATA_ELEM * num_sg_per_wg * subgroup_size * ACC_TYPE_SIZE;

    return dispatchData;
}

bool MVNKernelByxfWithinChannelsSubgroupOverX::Validate(const Params& p) const {
    if (!MVNKernelBase::Validate(p)) {
        return false;
    }
    const mvn_params& params = static_cast<const mvn_params&>(p);

    if (!GetSubgroupSize(params))
        return false;

    return true;
}

KernelsData MVNKernelByxfWithinChannelsSubgroupOverX::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority MVNKernelByxfWithinChannelsSubgroupOverX::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
