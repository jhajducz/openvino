// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"
#include "vector"

namespace kernel_selector {
class MVNKernelBzyxfOpt : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernelBzyxfOpt() : MVNKernelBase("mvn_gpu_bzyxf_opt") {}
    virtual ~MVNKernelBzyxfOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

private:
    DispatchData SetDefault(const mvn_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE, FusedOpType::ELTWISE, FusedOpType::ACTIVATION};
    }
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params,
                                 const MVNKernelBase::DispatchData dispatchData) const override;
    size_t GetWorkGroupSize(const mvn_params& params, const size_t& f_block_size = 1) const;
    size_t GetBlockSize(const mvn_params& params) const;
    std::map<Tensor::DataChannelName, size_t> GetNumDataElementsPerWorkItem(const mvn_params& params) const;
};
}  // namespace kernel_selector
