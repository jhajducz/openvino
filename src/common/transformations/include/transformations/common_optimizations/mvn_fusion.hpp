// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MVNFusion;
class TRANSFORMATIONS_API MVNFusionWithoutConstants;
class TRANSFORMATIONS_API MVNFusionWithConstantsInside;
class TRANSFORMATIONS_API MVNFusionWithConstantsInside_v2;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) to MVN op.
 */
class ov::pass::MVNFusionWithoutConstants : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MVNFusionWithoutConstants");
    MVNFusionWithoutConstants();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: gamma * (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta to MVN
 * op.
 */
class ov::pass::MVNFusionWithConstantsInside : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MVNFusionWithConstantsInside");
    MVNFusionWithConstantsInside();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief MVNFusion transformation replaces following operations to MVN op.
 * y = x * inv_sqrt_var * gamma + bias - mean * inv_sqrt_var * gamma
 * where:
 *     mean = Reshape(ReduceMean(x, axes), new_shape)
 *     mean_centered_x = x - mean
 *     inv_sqrt_var = Sqrt(ReduceMean(mean_centered_x * mean_centered_x) + eps)
 */
class ov::pass::MVNFusionWithConstantsInside_v2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MVNFusionWithConstantsInside_v2");
    MVNFusionWithConstantsInside_v2();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief MVNFusion transformation replaces various sub-graphs with a MVN op.
 */
class ov::pass::MVNFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MVNFusion");
    MVNFusion() {
        add_matcher<ov::pass::MVNFusionWithoutConstants>();
        add_matcher<ov::pass::MVNFusionWithConstantsInside>();
        add_matcher<ov::pass::MVNFusionWithConstantsInside_v2>();
    }
};
