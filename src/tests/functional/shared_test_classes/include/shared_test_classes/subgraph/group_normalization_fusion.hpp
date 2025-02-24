// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;

namespace ov {
namespace test {

using GroupNormalizationFusionTestBaseValues =
    std::tuple<PartialShape,        // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,               // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,               // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,               // shape of group norm gamma tensor
               Shape,               // shape of group norm beta tensor
               size_t,              // number of groups
               float>;              // epsilon

using GroupNormalizationFusionTransformationsTestValues =
    std::tuple<PartialShape,        // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,               // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,               // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,               // shape of group norm gamma tensor
               Shape,               // shape of group norm beta tensor
               size_t,              // number of groups
               float,               // epsilon
               bool,                // whether it's a positive test that should run reference model or a negative test
               std::string,         // taget device name
               AnyMap,              // taget device properties
               std::string,         // reference device name
               AnyMap>;             // reference device properties

template <typename... T_old_vals, typename... T_added_vals>
std::vector<std::tuple<T_old_vals..., T_added_vals...>> expand_vals(std::vector<std::tuple<T_old_vals...>> old_vals,
                                                                    std::tuple<T_added_vals...> added_vals) {
    std::vector<std::tuple<T_old_vals..., T_added_vals...>> res;
    for (const std::tuple<T_old_vals...>& t : old_vals) {
        auto new_tuple = std::tuple_cat(t, added_vals);
        res.push_back(new_tuple);
    }
    return res;
}

template <element::Type_t T_elem_t>
class GroupNormalizationFusionTestBase {
public:
    static constexpr element::Type T_elem = T_elem_t;
    typedef typename element_type_traits<T_elem_t>::value_type T_store_t;

protected:
    size_t numChannels;
    bool instanceNormGammaPresent;
    bool instanceNormBetaPresent;

    std::vector<T_store_t> instanceNormGammaVals;
    std::vector<T_store_t> instanceNormBetaVals;
    std::vector<T_store_t> groupNormGammaVals;
    std::vector<T_store_t> groupNormBetaVals;

    PartialShape dataShape;
    Shape instanceNormGammaShape;
    Shape instanceNormBetaShape;
    Shape groupNormGammaShape;
    Shape groupNormBetaShape;
    size_t numGroups;
    float epsilon;

    virtual void read_test_parameters() = 0;

    void generate_weights_init_values() {
        if (instanceNormGammaPresent)
            instanceNormGammaVals = test::utils::generateVector<T_elem_t>(shape_size(instanceNormGammaShape), 10, 1, 1);
        if (instanceNormBetaPresent)
            instanceNormBetaVals = test::utils::generateVector<T_elem_t>(shape_size(instanceNormBetaShape), 10, 1, 2);
        groupNormGammaVals = test::utils::generateVector<T_elem_t>(shape_size(groupNormGammaShape), 10, 1, 3);
        groupNormBetaVals = test::utils::generateVector<T_elem_t>(shape_size(groupNormBetaShape), 10, 1, 4);
    }

    std::shared_ptr<Model> create_model() {
        auto input = std::make_shared<op::v0::Parameter>(T_elem, dataShape);
        auto pre_mvn_shape_const =
            op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, static_cast<long long>(numGroups), -1});
        auto pre_mvn_reshape = std::make_shared<op::v1::Reshape>(input, pre_mvn_shape_const, true);

        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {2});
        auto mvn =
            std::make_shared<op::v6::MVN>(pre_mvn_reshape, mvn_axes_const, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instanceNormGammaPresent) {
            auto instance_norm_gamma_const =
                op::v0::Constant::create(T_elem, instanceNormGammaShape, instanceNormGammaVals);
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
        }

        std::shared_ptr<Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instanceNormBetaPresent) {
            auto instance_norm_beta_const =
                op::v0::Constant::create(T_elem, instanceNormBetaShape, instanceNormBetaVals);
            opt_instance_norm_beta_add =
                std::make_shared<op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
        }

        auto post_instance_norm_shape = std::make_shared<op::v0::ShapeOf>(input);

        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);

        auto group_norm_gamma_const = op::v0::Constant::create(T_elem, groupNormGammaShape, groupNormGammaVals);
        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);

        auto group_norm_beta_const = op::v0::Constant::create(T_elem, groupNormBetaShape, groupNormBetaVals);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        return std::make_shared<Model>(NodeVector{group_norm_beta_add}, ParameterVector{input});
    }
};

template <element::Type_t T_elem_t>
class GroupNormalizationFusionSubgraphTestsF
    : public GroupNormalizationFusionTestBase<T_elem_t>,
      public test::SubgraphBaseTest,
      public testing::WithParamInterface<GroupNormalizationFusionTransformationsTestValues> {
public:
    static constexpr element::Type T_elem = T_elem_t;
    static std::string getTestCaseName(
        const testing::TestParamInfo<GroupNormalizationFusionTransformationsTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& positive_test = std::get<7>(params);
        const auto& device_name = std::get<8>(params);
        const auto& device_properties = std::get<9>(params);
        const auto& ref_device_name = std::get<10>(params);
        const auto& ref_device_properties = std::get<11>(params);

        std::ostringstream results;

        results << "T=" << T_elem_t << "_";
        results << "Input=" << test::utils::partialShape2str({data_shape}) << "_";
        results << "InstNormGamma=" << test::utils::partialShape2str({instance_norm_gamma_shape}) << "_";
        results << "InstNormBeta=" << test::utils::partialShape2str({instance_norm_beta_shape}) << "_";
        results << "GroupNormGamma=" << test::utils::partialShape2str({group_norm_gamma_shape}) << "_";
        results << "GroupNormBeta=" << test::utils::partialShape2str({group_norm_beta_shape}) << "_";
        results << "NumGroups=" << num_groups << "_";
        results << "Epsilon=" << epsilon << "_";
        results << "PositiveTest=" << std::boolalpha << positive_test << "_";
        results << "Device=" << device_name << "_";
        results << "DeviceCfg=(";
        for (auto iter = device_properties.begin(); iter != device_properties.end(); iter++) {
            results << iter->first << "=" << iter->second.as<std::string>();
            if (std::next(iter) != device_properties.end())
                results << "_";
        }
        results << ")_";
        results << "RefDevice=" << ref_device_name << "_";
        results << "RefDeviceCfg=(";
        for (auto iter = ref_device_properties.begin(); iter != ref_device_properties.end(); iter++) {
            results << iter->first << "=" << iter->second.as<std::string>();
            if (std::next(iter) != ref_device_properties.end())
                results << "_";
        }
        results << ")";
        return results.str();
    }

protected:
    bool positiveTest;
    std::string targetDeviceName;
    AnyMap targetConfiguration;
    std::string refDevice;
    AnyMap refConfiguration;

    ElementType refInferencePrecision;
    CompiledModel compiledRefModel;
    InferRequest refInferRequest;

    void TearDown() override {
        SubgraphBaseTest::TearDown();
    }

    void read_test_parameters() override {
        const auto& params = GetParam();

        this->dataShape = std::get<0>(params);
        if (!this->dataShape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (this->dataShape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!this->dataShape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");

        this->numChannels = static_cast<size_t>(this->dataShape[1].get_max_length());
        this->instanceNormGammaShape = std::get<1>(params);
        this->instanceNormBetaShape = std::get<2>(params);
        this->groupNormGammaShape = std::get<3>(params);
        this->groupNormBetaShape = std::get<4>(params);
        this->numGroups = std::get<5>(params);
        this->epsilon = std::get<6>(params);
        positiveTest = std::get<7>(params);
        targetDeviceName = std::get<8>(params);
        targetConfiguration = std::get<9>(params);
        refDevice = std::get<10>(params);
        refConfiguration = std::get<11>(params);

        this->instanceNormGammaPresent = (this->instanceNormGammaShape != Shape{});
        this->instanceNormBetaPresent = (this->instanceNormBetaShape != Shape{});

        inType = T_elem;
        outType = T_elem;
        targetDevice = targetDeviceName;
        configuration = targetConfiguration;

        if (positiveTest) {
            if ((this->instanceNormGammaShape != Shape{}) &&
                (shape_size(this->instanceNormGammaShape) != this->numGroups))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <numGroups> elements");
            if ((this->instanceNormBetaShape != Shape{}) &&
                (shape_size(this->instanceNormBetaShape) != this->numGroups))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <numGroups> elements");
            if (shape_size(this->groupNormGammaShape) != this->numChannels)
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <numChannels> elements");
            if (shape_size(this->groupNormBetaShape) != this->numChannels)
                throw std::runtime_error("Shape of group norm beta has to contain exactly <numChannels> elements");

            this->instanceNormGammaPresent =
                this->instanceNormGammaPresent && (shape_size(this->instanceNormGammaShape) == this->numGroups);
            this->instanceNormBetaPresent =
                this->instanceNormBetaPresent && (shape_size(this->instanceNormBetaShape) == this->numGroups);
        }
    }

    void configure_device() {
        if (targetConfiguration.count(hint::inference_precision.name()) <= 0) {
            targetConfiguration.insert({hint::inference_precision.name(), T_elem});
        }
    }

    void configure_ref_device() {
        if (refConfiguration.count(hint::inference_precision.name()) <= 0) {
            refConfiguration.insert({hint::inference_precision.name(), T_elem});
        }
    }

    void configure_ref_model() {
        // configure input precision
        preprocess::PrePostProcessor p(functionRefs);
        {
            auto& params = functionRefs->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (inType != element::Type_t::undefined) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }

        // configure output precision
        {
            auto results = functionRefs->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != element::Type_t::undefined) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        functionRefs = p.build();
    }

    void compile_ref_model() {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::compile_ref_model()` is started"
                      << std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        configure_ref_model();
        core_configuration(this);
        compiledRefModel = core->compile_model(functionRefs, refDevice, refConfiguration);
        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::compile_ref_model()` is finished "
                         "successfully. Duration is "
                      << duration.count() << "s" << std::endl;
        }
        try {
            refInferencePrecision = core->get_property(refDevice, hint::inference_precision);
        } catch (std::exception& e) {
            std::cout << "[ WARNING ] Impossible to get Inference Precision with exception: " << e.what() << std::endl;
        }
    }

    void init_thresholds() override {
        if (!targetStaticShapes.empty()) {
            size_t problem_size = shape_size(this->dataShape.get_shape());

            abs_threshold = pow(problem_size, 0.5) * test::utils::get_eps_by_ov_type(outType);
            rel_threshold = abs_threshold;
        }
    }

    void infer_ref(const std::map<std::shared_ptr<Node>, Tensor>& inputs_ref) {
        refInferRequest = compiledRefModel.create_infer_request();
        for (const auto& input : inputs_ref) {
            refInferRequest.set_tensor(input.first, input.second);
        }
        refInferRequest.infer();
    }

    std::vector<Tensor> calculate_refs() override {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::calculate_refs()` is started"
                      << std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        update_ref_model();
        match_parameters(function->get_parameters(), functionRefs->get_parameters());

        std::map<std::shared_ptr<Node>, Tensor> inputs_ref;
        for (const auto& param : functionRefs->get_parameters()) {
            inputs_ref[param] = inputs.at(matched_parameters[param]);
        }

        infer_ref(inputs_ref);
        auto outputs = std::vector<Tensor>{};
        for (const auto& output : functionRefs->outputs()) {
            outputs.push_back(refInferRequest.get_tensor(output));
        }
        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::calculate_refs()` is finished "
                         "successfully. Duration is "
                      << duration.count() << "s" << std::endl;
        }
        return outputs;
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        inputs.clear();

        auto itTargetShape = targetInputStaticShapes.begin();
        for (const auto& param : function->get_parameters()) {
            std::shared_ptr<Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto& node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<Node> nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                            const auto& tensor = test::utils::create_and_fill_tensor(inType, *itTargetShape);
                            inputs.insert({param, tensor});
                            break;
                        }
                    }
                }
            }
            itTargetShape++;
        }
    }

public:
    void run() override {
        is_reported = true;
        bool isCurrentTestDisabled = test::utils::current_test_is_disabled();

        test::utils::PassRate::Statuses status =
            isCurrentTestDisabled ? test::utils::PassRate::Statuses::SKIPPED : test::utils::PassRate::Statuses::CRASHED;

        if (isCurrentTestDisabled)
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

        // in case of crash jump will be made and work will be continued
        auto crashHandler = std::unique_ptr<test::utils::CrashHandler>(new test::utils::CrashHandler());

        // place to jump in case of a crash
        int jmpRes = 0;
#ifdef _WIN32
        jmpRes = setjmp(test::utils::env);
#else
        jmpRes = sigsetjmp(test::utils::env, 1);
#endif
        if (jmpRes == test::utils::JMP_STATUS::ok) {
            crashHandler->StartTimer();
            std::string errorMessage;
            try {
                read_test_parameters();
                this->generate_weights_init_values();
                functionRefs = this->create_model();
                function = functionRefs->clone();
                pass::Manager m;
                m.register_pass<pass::GroupNormalizationFusion>();
                OV_ASSERT_NO_THROW(m.run_passes(function));

                summary.setDeviceName(targetDevice);
                summary.updateOPsStats(function, status, rel_influence_coef);
                if (positiveTest) {
                    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(functionRefs), 0);
                    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(function), 1);

                    if (!function->is_dynamic()) {
                        configure_device();
                        configure_ref_device();
                        auto input_shapes = static_partial_shapes_to_test_representation({this->dataShape});
                        init_input_shapes(input_shapes);
                        ASSERT_FALSE(targetStaticShapes.empty() && !function->get_parameters().empty())
                            << "Target Static Shape is empty!!!";
                        compile_model();
                        compile_ref_model();
                        for (const auto& targetStaticShapeVec : targetStaticShapes) {
                            generate_inputs(targetStaticShapeVec);
                            validate();
                        }
                    }
                } else {
                    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(functionRefs), 0);
                    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(function), 0);
                }
                status = test::utils::PassRate::Statuses::PASSED;
            } catch (const std::exception& ex) {
                if (callback_exception != nullptr) {
                    // exception will be checked by callback.
                    callback_exception(ex);
                    return;
                } else {
                    status = test::utils::PassRate::Statuses::FAILED;
                    errorMessage = ex.what();
                }
            } catch (...) {
                status = test::utils::PassRate::Statuses::FAILED;
                errorMessage = "Unknown failure occurred.";
            }
            summary.updateOPsStats(function, status, rel_influence_coef);
            if (status != test::utils::PassRate::Statuses::PASSED) {
                GTEST_FATAL_FAILURE_(errorMessage.c_str());
            }
        } else if (jmpRes == test::utils::JMP_STATUS::anyError) {
            OPENVINO_THROW("Crash happens");
        } else if (jmpRes == test::utils::JMP_STATUS::alarmErr) {
            summary.updateOPsStats(function, test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
            OPENVINO_THROW("Crash happens");
        }
    }
};

class GroupNormalizationFusionSubgraphTestsF_f32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f32> {
};
class GroupNormalizationFusionSubgraphTestsF_f16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f16> {
};
class GroupNormalizationFusionSubgraphTestsF_bf16
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::bf16> {};

class GroupNormalizationFusionSubgraphTestsF_u8 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u8> {};
class GroupNormalizationFusionSubgraphTestsF_u16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u16> {
};
class GroupNormalizationFusionSubgraphTestsF_u32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u32> {
};
class GroupNormalizationFusionSubgraphTestsF_u64 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u64> {
};
class GroupNormalizationFusionSubgraphTestsF_i8 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i8> {};
class GroupNormalizationFusionSubgraphTestsF_i16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i16> {
};
class GroupNormalizationFusionSubgraphTestsF_i32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i32> {
};
class GroupNormalizationFusionSubgraphTestsF_i64 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i64> {
};
class GroupNormalizationFusionSubgraphTestsF_f8e4m3
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e4m3> {};
class GroupNormalizationFusionSubgraphTestsF_f8e5m2
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e5m2> {};
class GroupNormalizationFusionSubgraphTestsF_f4e2m1
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f4e2m1> {};
class GroupNormalizationFusionSubgraphTestsF_f8e8m0
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e8m0> {};

}  // namespace test
}  // namespace ov
