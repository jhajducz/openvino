// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_bzyxf_opt.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr char ACC_TYPE_NAME[] = "float";
static constexpr size_t ACC_TYPE_SIZE = 4;
static constexpr size_t MAX_F_BLOCK_SIZE = 16;
static constexpr size_t SLM_ELEMS_PER_WI_DATA_ELEM = 2;

ParamsKey MVNKernelBzyxfOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bzyxf);
    k.EnableOutputLayout(DataLayout::bzyxf);
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNNormalizeVariance();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

size_t MVNKernelBzyxfOpt::GetWorkGroupSize(const mvn_params& params, const size_t& f_block_size) const {
    const auto& input = params.inputs[0];

    size_t wg_size = 1;

    size_t max_wg_size_due_to_dims = 1;
    if (params.mvnMode == MVNMode::ACROSS_CHANNELS)
        max_wg_size_due_to_dims *= input.Feature().v / f_block_size;
    max_wg_size_due_to_dims *= input.X().v * input.Y().v;

    if (input.GetDims().size() == 5)
        max_wg_size_due_to_dims *= input.Z().v;

    while ((2 * wg_size <= params.engineInfo.maxWorkGroupSize) && (2 * wg_size <= max_wg_size_due_to_dims)) {
        wg_size *= 2;
    }

    return wg_size;
}

size_t MVNKernelBzyxfOpt::GetBlockSize(const mvn_params& params) const {
    const auto& input = params.inputs[0];

    const auto& spatial_wg_size = GetWorkGroupSize(params);
    size_t f_block_size = 1;
    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        while ((SLM_ELEMS_PER_WI_DATA_ELEM * spatial_wg_size * f_block_size * ACC_TYPE_SIZE <
                params.engineInfo.maxLocalMemSize) &&
               (2 * f_block_size <= MAX_F_BLOCK_SIZE) && (2 * f_block_size <= input.Feature().v)) {
            f_block_size *= 2;
        }
    } else {
        while ((2 * f_block_size <= input.Feature().v) && (2 * f_block_size <= MAX_F_BLOCK_SIZE)) {
            f_block_size *= 2;
        }
    }

    return f_block_size;
}

std::map<Tensor::DataChannelName, size_t> MVNKernelBzyxfOpt::GetNumDataElementsPerWorkItem(
    const mvn_params& params) const {
    std::map<Tensor::DataChannelName, size_t> res;

    const auto& input = params.inputs[0];

    size_t num_z_per_wi = 1;
    size_t num_y_per_wi = 1;
    size_t num_x_per_wi = 1;
    size_t num_f_blocks_per_wi = 1;

    const auto& f_block_size = GetBlockSize(params);
    const auto& wg_size = GetWorkGroupSize(params, f_block_size);
    auto spatial_wg_size = wg_size;

    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        while ((SLM_ELEMS_PER_WI_DATA_ELEM * spatial_wg_size * num_f_blocks_per_wi * f_block_size * ACC_TYPE_SIZE <
                params.engineInfo.maxLocalMemSize) &&
               (2 * num_f_blocks_per_wi * f_block_size <= input.Feature().v)) {
            num_f_blocks_per_wi *= 2;
        }
    } else {
        num_f_blocks_per_wi = input.Feature().v / f_block_size;
    }

    if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
        spatial_wg_size /= num_f_blocks_per_wi;
    }

    num_x_per_wi = (input.X().v + spatial_wg_size - 1) / spatial_wg_size;

    if (num_x_per_wi > 1) {
        num_y_per_wi = input.Y().v;
    } else {
        size_t ceil_spatial_wg_size_div_by_x = (spatial_wg_size + input.X().v - 1) / input.X().v;
        num_y_per_wi = (input.Y().v + ceil_spatial_wg_size_div_by_x - 1) / ceil_spatial_wg_size_div_by_x;
    }

    if (input.GetDims().size() == 5) {
        if ((num_x_per_wi > 1) && (num_y_per_wi > 1)) {
            num_z_per_wi = input.Z().v;
        } else {
            size_t ceil_spatial_wg_size_div_by_xy =
                (spatial_wg_size + input.X().v * input.Y().v - 1) / (input.X().v * input.Y().v);
            num_z_per_wi = (input.Y().v + ceil_spatial_wg_size_div_by_xy - 1) / ceil_spatial_wg_size_div_by_xy;
        }
    } else {
        num_z_per_wi = 1;
    }

    res[Tensor::DataChannelName::BATCH] = 1;
    if (params.mvnMode == MVNMode::WITHIN_CHANNELS)
        res[Tensor::DataChannelName::FEATURE] = num_f_blocks_per_wi * f_block_size;
    else
        res[Tensor::DataChannelName::FEATURE] = f_block_size;
    res[Tensor::DataChannelName::X] = num_x_per_wi;
    res[Tensor::DataChannelName::Y] = num_y_per_wi;
    res[Tensor::DataChannelName::Z] = num_z_per_wi;

    return res;
}

JitConstants MVNKernelBzyxfOpt::GetJitConstants(const mvn_params& params,
                                                const MVNKernelBase::DispatchData dispatchData) const {
    const auto& input_dt = params.inputs[0].GetDType();

    const auto& input = params.inputs[0];

    const auto& f_block_size = GetBlockSize(params);
    const auto& wg_size = GetWorkGroupSize(params, f_block_size);
    const auto& data_elems_per_wi = GetNumDataElementsPerWorkItem(params);

    const auto& num_f_blocks_per_wi = data_elems_per_wi.at(Tensor::DataChannelName::FEATURE) / f_block_size;
    size_t spatial_wg_size = wg_size;
    if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
        spatial_wg_size /= input.Feature().v / f_block_size;
    }

    DimensionAccessHelperJit dims(input);

    std::string data_set_size;

    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z()});
    } else {
        data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
    }

    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstants({
        MakeJitConstant("DATA_SET_SIZE", data_set_size),
        MakeJitConstant("WG_SIZE", wg_size),
        MakeJitConstant("SPATIAL_WG_SIZE", spatial_wg_size),
        MakeJitConstant("F_BLOCK_SIZE", f_block_size),
        MakeJitConstant("NUM_F_BLOCKS_PER_WI", num_f_blocks_per_wi),
        MakeJitConstant("NUM_X_PER_WI", data_elems_per_wi.at(Tensor::DataChannelName::X)),
        MakeJitConstant("NUM_Y_PER_WI", data_elems_per_wi.at(Tensor::DataChannelName::Y)),
        MakeJitConstant("NUM_Z_PER_WI", data_elems_per_wi.at(Tensor::DataChannelName::Z)),
        MakeJitConstant("ACCUMULATOR_TYPE", ACC_TYPE_NAME),
        MakeJitConstant("ACCUMULATOR_TYPE_SIZE", ACC_TYPE_SIZE),
    });

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"",
                                      {"b_start", "f_start + fb * F_BLOCK_SIZE + i", "y_start + y", "x_start + x"},
                                      "out_vals.f[i]",
                                      input_dt,
                                      1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

MVNKernelBzyxfOpt::Parent::DispatchData MVNKernelBzyxfOpt::SetDefault(const mvn_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);

    const auto& input = params.inputs[0];

    const auto& f_block_size = GetBlockSize(params);
    const auto& wg_size = GetWorkGroupSize(params, f_block_size);
    const auto& data_elems_per_wi = GetNumDataElementsPerWorkItem(params);

    const auto& num_f_blocks_per_wi = data_elems_per_wi.at(Tensor::DataChannelName::FEATURE) / f_block_size;
    size_t spatial_wg_size = wg_size;
    if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
        spatial_wg_size /= input.Feature().v / f_block_size;
    }

    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        dispatchData.gws[0] = input.Batch().v;
        dispatchData.gws[1] = input.Feature().v / data_elems_per_wi.at(Tensor::DataChannelName::FEATURE);
        dispatchData.gws[2] = spatial_wg_size;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = spatial_wg_size;
    } else {
        dispatchData.gws[0] = input.Batch().v;
        dispatchData.gws[1] = input.Feature().v / f_block_size;
        dispatchData.gws[2] = spatial_wg_size;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = input.Feature().v / f_block_size;
        dispatchData.lws[2] = spatial_wg_size;
    }

    dispatchData.maxSlmSize = SLM_ELEMS_PER_WI_DATA_ELEM * spatial_wg_size *
                              data_elems_per_wi.at(Tensor::DataChannelName::FEATURE) * ACC_TYPE_SIZE;

    return dispatchData;
}

bool MVNKernelBzyxfOpt::Validate(const Params& p) const {
    if (!MVNKernelBase::Validate(p)) {
        return false;
    }
    const mvn_params& params = static_cast<const mvn_params&>(p);

    if (!(params.inputs[0].Feature().v && !(params.inputs[0].Feature().v & (params.inputs[0].Feature().v - 1))))
        return false;

    if (!(params.inputs[0].X().v && !(params.inputs[0].X().v & (params.inputs[0].X().v - 1))))
        return false;

    if ((params.inputs[0].X().v > params.engineInfo.maxWorkGroupSize) &&
        (params.inputs[0].X().v % params.engineInfo.maxWorkGroupSize))
        return false;

    if (!(params.inputs[0].Y().v && !(params.inputs[0].Y().v & (params.inputs[0].Y().v - 1))))
        return false;

    if ((params.inputs[0].Y().v > params.engineInfo.maxWorkGroupSize) &&
        (params.inputs[0].Y().v % params.engineInfo.maxWorkGroupSize))
        return false;

    if (params.inputs[0].GetDims().size() == 5) {
        if (!(params.inputs[0].Z().v && !(params.inputs[0].Z().v & (params.inputs[0].Z().v - 1))))
            return false;

        if ((params.inputs[0].Z().v > params.engineInfo.maxWorkGroupSize) &&
            (params.inputs[0].Z().v % params.engineInfo.maxWorkGroupSize))
            return false;
    }

    return true;
}

KernelsData MVNKernelBzyxfOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority MVNKernelBzyxfOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
