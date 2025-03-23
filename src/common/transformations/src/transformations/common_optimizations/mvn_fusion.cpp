// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mvn_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

template <class T>
std::function<bool(ov::Output<ov::Node>)> value_is_equal_to(const std::vector<T>& ref_values) {
    return [ref_values](ov::Output<ov::Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

ov::pass::MVNFusionWithoutConstants::MVNFusionWithoutConstants() {
    MATCHER_SCOPE(MVNFusionWithoutConstants);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean1 = pattern::wrap_type<ov::op::v1::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<ov::op::v1::Subtract>({x, mean1});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean2 = pattern::wrap_type<ov::op::v1::ReduceMean>({x, mean2_axes});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = pattern::wrap_type<ov::op::v1::Subtract>({x, mean2});

    const auto reuseSub1OrNot = std::make_shared<pattern::op::Or>(OutputVector{sub1, sub2});
    const auto optionalConvert = pattern::optional<ov::op::v0::Convert>(reuseSub1OrNot);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<ov::op::v0::Constant>(value_is_equal_to<float>({2.0}));
    auto power = pattern::wrap_type<ov::op::v1::Power>({optionalConvert, const_2});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean3 = pattern::wrap_type<ov::op::v1::ReduceMean>({power, mean3_axes});

    auto const_0_5 = pattern::wrap_type<ov::op::v0::Constant>(value_is_equal_to<float>({0.5}));
    auto eps = pattern::wrap_type<ov::op::v0::Constant>();
    // ------------------- OUTSIDE_SQRT ----------------------

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_os = pattern::wrap_type<ov::op::v1::Power>({mean3, const_0_5});
    auto sqrt_os = pattern::wrap_type<ov::op::v0::Sqrt>({mean3});
    const auto powerOrSqrt_os = std::make_shared<pattern::op::Or>(OutputVector{power_sqrt_os, sqrt_os});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `----------------------------------------------Add---'
    auto add_eps_os = pattern::wrap_type<ov::op::v1::Add>({powerOrSqrt_os, eps});

    // ------------------- INSIDE_SQRT ----------------------

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    auto add_eps_is = pattern::wrap_type<ov::op::v1::Add>({mean3, eps});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_is = pattern::wrap_type<ov::op::v1::Power>({add_eps_is, const_0_5});
    auto sqrt_is = pattern::wrap_type<ov::op::v0::Sqrt>({add_eps_is});
    const auto powerOrSqrt_is = std::make_shared<pattern::op::Or>(OutputVector{power_sqrt_is, sqrt_is});

    auto outsideOrInside = std::make_shared<pattern::op::Or>(OutputVector{add_eps_os, powerOrSqrt_is});

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<ov::op::v0::Constant>(value_is_equal_to<float>({-1}));
    auto power_div = pattern::wrap_type<ov::op::v1::Power>({outsideOrInside, const_neg_1});
    auto div = pattern::wrap_type<ov::op::v1::Multiply>({sub1, power_div});

    auto div_alt = pattern::wrap_type<ov::op::v1::Divide>({sub1, outsideOrInside});
    const auto powerMulOrDiv = std::make_shared<pattern::op::Or>(OutputVector{div, div_alt});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);

        auto const_eps_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes_1_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_3_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (axes_1_value != axes_3_value) {
            return false;
        }
        if (pattern_to_output.count(mean2_axes)) {
            auto axes_2_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
            if (!axes_2_node) {
                return false;
            }
            auto axes_2_value = axes_2_node->cast_vector<int64_t>();
            if (axes_1_value != axes_2_value) {
                return false;
            }
        }

        ov::NodeVector nodes_to_copy_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                                           pattern_to_output.at(sub1).get_node_shared_ptr(),
                                           pattern_to_output.at(power).get_node_shared_ptr(),
                                           pattern_to_output.at(mean3).get_node_shared_ptr()});

        op::MVNEpsMode mode;
        if (pattern_to_output.count(add_eps_os)) {
            mode = op::MVNEpsMode::OUTSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_os).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_os).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_os).get_node_shared_ptr());
            }
        } else if (pattern_to_output.count(powerOrSqrt_is)) {
            mode = op::MVNEpsMode::INSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_is).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_is).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_is).get_node_shared_ptr());
            }
        } else {
            return false;
        }
        auto mvn = std::make_shared<ov::op::v6::MVN>(exp_input, axes_1_node, true, eps_value, mode);

        if (pattern_to_output.count(mean2) && pattern_to_output.count(sub2)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(mean2).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(sub2).get_node_shared_ptr());
        }

        if (pattern_to_output.count(optionalConvert)) {
            auto cast = pattern_to_output.at(optionalConvert).get_node_shared_ptr();
            nodes_to_copy_info.push_back(cast);
        }

        if (pattern_to_output.count(div_alt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(div_alt).get_node_shared_ptr());
        } else if (pattern_to_output.count(power_div) && pattern_to_output.count(div)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_div).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(div).get_node_shared_ptr());
        }

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(nodes_to_copy_info, mvn);
        ov::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(powerMulOrDiv, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::MVNFusionWithConstantsInside::MVNFusionWithConstantsInside() {
    MATCHER_SCOPE(MVNFusionWithConstantsInside);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) * gamma / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))^2
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean1 = pattern::wrap_type<ov::op::v1::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))^2
    // `-squared_difference------'
    auto squared_difference = pattern::wrap_type<ov::op::v0::SquaredDifference>({x, mean1});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `---mean2--------------------------------'
    auto mean2_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean2 = pattern::wrap_type<ov::op::v1::ReduceMean>({squared_difference, mean2_axes});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `------------------------------------------add--'
    auto eps = pattern::wrap_type<ov::op::v0::Constant>();
    auto add_eps = pattern::wrap_type<ov::op::v1::Add>({mean2, eps});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `-power-------------------------------------------------'
    auto const_0_5 = pattern::wrap_type<ov::op::v0::Constant>(value_is_equal_to<float>({-0.5}));
    auto power = pattern::wrap_type<ov::op::v1::Power>({add_eps, const_0_5});

    // gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul1----------------------------------------------------'
    auto gamma = pattern::wrap_type<ov::op::v0::Constant>();
    auto mul1 = pattern::wrap_type<ov::op::v1::Multiply>({power, gamma});

    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul2--------------------------------------------------------'
    auto mul2 = pattern::wrap_type<ov::op::v1::Multiply>({x, mul1});

    // ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) - beta
    // `-------------------mul3----------------------------------------------------------'
    auto mul3 = pattern::wrap_type<ov::op::v1::Multiply>({mul1, mean1});

    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---sub-----------------------------------------------------------------------------------'
    auto beta = pattern::wrap_type<ov::op::v0::Constant>();
    auto sub = pattern::wrap_type<ov::op::v1::Subtract>({beta, mul3});

    // Final Add
    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) +
    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) =
    // gamma * (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) + beta
    auto add = pattern::wrap_type<ov::op::v1::Add>({mul2, sub});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(x);

        auto const_0_5_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(const_0_5).get_node_shared_ptr());
        auto const_gamma_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(gamma).get_node_shared_ptr());
        auto const_beta_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(beta).get_node_shared_ptr());
        auto const_eps_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        if (!const_0_5_node || !const_beta_node || !const_gamma_node || !const_eps_node) {
            return false;
        }

        float eps_value;
        bool valid_constant_values = op::util::has_constant_value<float>(const_0_5_node, -0.5) &&
                                     op::util::get_single_value(const_eps_node, eps_value);
        if (!valid_constant_values) {
            return false;
        }

        auto axes_1_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_2_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
        if (!axes_1_node || !axes_2_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_2_value = axes_2_node->cast_vector<int64_t>();
        if (axes_1_value != axes_2_value) {
            return false;
        }

        auto mvn =
            std::make_shared<ov::op::v6::MVN>(x_output, axes_1_node, true, eps_value, op::MVNEpsMode::INSIDE_SQRT);
        auto mul_gamma = std::make_shared<ov::op::v1::Multiply>(mvn, const_gamma_node);
        auto add_beta = std::make_shared<ov::op::v1::Add>(mul_gamma, const_beta_node);

        ov::copy_runtime_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                               pattern_to_output.at(squared_difference).get_node_shared_ptr(),
                               pattern_to_output.at(add_eps).get_node_shared_ptr(),
                               pattern_to_output.at(power).get_node_shared_ptr(),
                               pattern_to_output.at(mul1).get_node_shared_ptr(),
                               pattern_to_output.at(mul2).get_node_shared_ptr(),
                               pattern_to_output.at(mul3).get_node_shared_ptr(),
                               pattern_to_output.at(sub).get_node_shared_ptr(),
                               pattern_to_output.at(add).get_node_shared_ptr()},
                              {mvn, const_gamma_node, mul_gamma, const_beta_node, add_beta});
        add_beta->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::replace_node(m.get_match_root(), add_beta);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::MVNFusionWithConstantsInside_v2::MVNFusionWithConstantsInside_v2() {
    MATCHER_SCOPE(MVNFusionWithConstantsInside_v2);

    auto has_real_not_quantized_type = [](const ov::Output<ov::Node>& output) -> bool {
        const auto& T = output.get_element_type();
        return (T.is_real() && (!T.is_quantized()));
    };

    auto has_integral_type = [](const ov::Output<ov::Node>& output) -> bool {
        const auto& T = output.get_element_type();
        return (T.is_integral_number());
    };

    auto has_single_value = [](const ov::Output<ov::Node>& output) -> bool {
        const auto& output_shape = output.get_shape();
        return (ov::shape_size(output_shape) == 1);
    };

    // Detect following MVN decomposition pattern:
    // y = x * inv_sqrt_var * gamma + bias - mean * inv_sqrt_var * gamma
    // where:
    //     mean = Reshape(ReduceMean(x, axes), new_shape)
    //     mean_centered_x = x - mean
    //     inv_sqrt_var = Sqrt(ReduceMean(mean_centered_x * mean_centered_x) + eps)

    auto x = pattern::any_input();

    auto mean_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean_before_opt_reshape = pattern::wrap_type<ov::op::v1::ReduceMean>({x, mean_axes});
    auto opt_reshape_new_shape = pattern::wrap_type<ov::op::v0::Constant>();
    auto mean = pattern::optional<ov::op::v1::Reshape>({mean_before_opt_reshape, opt_reshape_new_shape});

    auto mean_centered_x = pattern::wrap_type<ov::op::v1::Subtract>({x, mean});

    auto mean_centered_x_squared = pattern::wrap_type<ov::op::v1::Multiply>({mean_centered_x, mean_centered_x});
    auto var_axes = pattern::wrap_type<ov::op::v0::Constant>();
    auto var = pattern::wrap_type<ov::op::v1::ReduceMean>({mean_centered_x_squared, var_axes});
    auto eps_const = pattern::wrap_type<ov::op::v0::Constant>(
        ov::pass::pattern::all_of({pattern::has_static_shape(), has_single_value, has_real_not_quantized_type}));
    auto eps_quantized = pattern::wrap_type<ov::op::v0::Constant>(
        ov::pass::pattern::all_of({pattern::has_static_shape(), has_single_value, has_integral_type}));
    auto eps_quant_convert = pattern::wrap_type<ov::op::v0::Convert>({eps_quantized});
    auto eps_zp_quantized = pattern::wrap_type<ov::op::v0::Constant>(
        ov::pass::pattern::all_of({pattern::has_static_shape(), has_single_value, has_integral_type}));
    auto eps_zp_convert = pattern::wrap_type<ov::op::v0::Convert>({eps_zp_quantized});
    auto eps_zp_subtract = pattern::wrap_type<ov::op::v1::Subtract>({eps_quant_convert, eps_zp_convert});
    auto eps_scale = pattern::wrap_type<ov::op::v0::Constant>(
        ov::pass::pattern::all_of({pattern::has_static_shape(), has_single_value, has_real_not_quantized_type}));
    auto eps_dequantized = pattern::wrap_type<ov::op::v1::Multiply>({eps_zp_subtract, eps_scale});
    auto eps = std::make_shared<pattern::op::Or>(OutputVector{eps_const, eps_dequantized});
    auto var_plus_eps = pattern::wrap_type<ov::op::v1::Add>({var, eps});
    auto sqrt = pattern::wrap_type<ov::op::v0::Sqrt>({var_plus_eps});
    auto ones = pattern::wrap_type<ov::op::v0::Constant>(
        ov::pass::pattern::all_of({pattern::has_static_shape(), has_single_value, has_real_not_quantized_type}));
    auto inv_sqrt_var = pattern::wrap_type<ov::op::v1::Divide>({ones, sqrt});

    // x * inv_sqrt_var * gamma
    auto gamma = pattern::any_input();
    auto scaled_inv_sqrt_var = pattern::wrap_type<ov::op::v1::Multiply>({inv_sqrt_var, gamma});
    auto x_mult_by_scaled_inv_sqrt_var = pattern::wrap_type<ov::op::v1::Multiply>({x, scaled_inv_sqrt_var});

    // beta - mean * inv_sqrt_var * gamma
    auto mean_mult_by_scaled_inv_sqrt_var = pattern::wrap_type<ov::op::v1::Multiply>({mean, scaled_inv_sqrt_var});
    auto beta = pattern::any_input();
    auto beta_minus_mean_mult_by_scaled_inv_sqrt_var =
        pattern::wrap_type<ov::op::v1::Subtract>({beta, mean_mult_by_scaled_inv_sqrt_var});

    // y = x * inv_sqrt * gamma + beta - mean * inv_sqrt_var * gamma
    auto y = pattern::wrap_type<ov::op::v1::Add>(
        {x_mult_by_scaled_inv_sqrt_var, beta_minus_mean_mult_by_scaled_inv_sqrt_var});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(x);

        float eps_value;

        if (pattern_to_output.count(eps_const) > 0) {
            auto const_eps_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());
            if (!const_eps_node)
                return false;
            bool valid_eps_value = op::util::get_single_value(const_eps_node, eps_value);
            if (!valid_eps_value)
                return false;
        } else if ((pattern_to_output.count(eps_dequantized) > 0) && (pattern_to_output.count(eps_zp_quantized) > 0) &&
                   (pattern_to_output.count(eps_scale) > 0)) {
            auto const_eps_quantized_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps_quantized).get_node_shared_ptr());
            auto const_eps_zp_quantized_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps_zp_quantized).get_node_shared_ptr());
            auto const_eps_scale_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps_scale).get_node_shared_ptr());

            if ((!const_eps_quantized_node) || (!const_eps_zp_quantized_node) || (!const_eps_scale_node))
                return false;

            if (const_eps_quantized_node->get_output_element_type(0) !=
                const_eps_zp_quantized_node->get_output_element_type(0))
                return false;

            float eps_quantized_flt_val, eps_zp_flt_val, eps_scale_val;
            bool valid_eps_quantized_flt_val =
                op::util::get_single_value(const_eps_quantized_node, eps_quantized_flt_val);
            if (!valid_eps_quantized_flt_val)
                return false;
            bool valid_eps_zp_flt_val = op::util::get_single_value(const_eps_zp_quantized_node, eps_zp_flt_val);
            if (!valid_eps_zp_flt_val)
                return false;
            bool valid_eps_scale_val = op::util::get_single_value(const_eps_scale_node, eps_scale_val);
            if (!valid_eps_scale_val)
                return false;

            eps_value = eps_scale_val * (eps_quantized_flt_val - eps_zp_flt_val);
        } else {
            return false;
        }

        auto const_ones_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(ones).get_node_shared_ptr());
        if (!const_ones_node) {
            return false;
        }

        bool valid_ones_values = op::util::has_constant_value<float>(const_ones_node, 1);
        if (!valid_ones_values) {
            return false;
        }

        auto const_mean_axes_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mean_axes).get_node_shared_ptr());
        auto const_variance_axes_node =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(var_axes).get_node_shared_ptr());
        if (!const_mean_axes_node || !const_variance_axes_node) {
            return false;
        }

        auto mean_axes_value = const_mean_axes_node->cast_vector<int64_t>();
        auto variance_axes_value = const_variance_axes_node->cast_vector<int64_t>();
        if (mean_axes_value != variance_axes_value) {
            return false;
        }

        auto mvn = std::make_shared<ov::op::v6::MVN>(x_output,
                                                     const_mean_axes_node,
                                                     true,
                                                     eps_value,
                                                     op::MVNEpsMode::INSIDE_SQRT);
        auto mul_gamma = std::make_shared<ov::op::v1::Multiply>(mvn, pattern_to_output.at(gamma));
        auto add_beta = std::make_shared<ov::op::v1::Add>(mul_gamma, pattern_to_output.at(beta));
        ov::copy_runtime_info({pattern_to_output.at(mean_before_opt_reshape).get_node_shared_ptr(),
                               pattern_to_output.at(mean).get_node_shared_ptr(),
                               pattern_to_output.at(mean_centered_x).get_node_shared_ptr(),
                               pattern_to_output.at(mean_centered_x_squared).get_node_shared_ptr(),
                               pattern_to_output.at(var).get_node_shared_ptr(),
                               pattern_to_output.at(var_plus_eps).get_node_shared_ptr(),
                               pattern_to_output.at(inv_sqrt_var).get_node_shared_ptr(),
                               pattern_to_output.at(scaled_inv_sqrt_var).get_node_shared_ptr(),
                               pattern_to_output.at(x_mult_by_scaled_inv_sqrt_var).get_node_shared_ptr(),
                               pattern_to_output.at(mean_mult_by_scaled_inv_sqrt_var).get_node_shared_ptr(),
                               pattern_to_output.at(beta_minus_mean_mult_by_scaled_inv_sqrt_var).get_node_shared_ptr(),
                               pattern_to_output.at(y).get_node_shared_ptr()},
                              {mvn, mul_gamma, add_beta});
        add_beta->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::replace_node(m.get_match_root(), add_beta);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(y, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
