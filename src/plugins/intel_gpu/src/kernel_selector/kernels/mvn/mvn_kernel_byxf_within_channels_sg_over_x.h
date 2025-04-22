// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"
#include "vector"

namespace kernel_selector {
class MVNKernelByxfWithinChannelsSubgroupOverX : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernelByxfWithinChannelsSubgroupOverX() : MVNKernelBase("mvn_gpu_byxf_within_channels_sg_over_x") {}
    virtual ~MVNKernelByxfWithinChannelsSubgroupOverX() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

private:
    struct tile_config {
        size_t tile_size = 0;
        size_t tile_size_x = 0;
        size_t tile_size_y = 0;
    };
    DispatchData SetDefault(const mvn_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE, FusedOpType::ELTWISE, FusedOpType::ACTIVATION};
    }
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params,
                                 const MVNKernelBase::DispatchData dispatchData) const override;
    size_t GetDatasetSize(const mvn_params& params) const;
    size_t GetSubgroupSize(const mvn_params& params) const;
    size_t GetNumSubgroupsPerWorkgroup(const mvn_params& params) const;
    tile_config GetTileConfig(const mvn_params& params) const;
};
}  // namespace kernel_selector
