#include "ext_ops.hpp"

#define REGISTER_OPERATOR_CUSOTM(name_, fn_) \
    ngraph::onnx_import::register_operator(name_, 1, "", fn_)

ngraph::OutputVector op_example(const ngraph::onnx_import::Node &onnx_node)
{
    namespace opset = ngraph::opset5;

    ngraph::OutputVector ng_inputs{onnx_node.get_ng_inputs()};
    const ngraph::Output<ngraph::Node> &data = ng_inputs.at(0);
    // create constant node with a single element that's equal to zero
    std::shared_ptr<ngraph::Node> zero_node = opset::Constant::create(data.get_element_type(), ngraph::Shape{}, {0});
    // create a negative map for 'data' node, 1 for negative values , 0 for positive values or zero
    // then convert it from boolean type to `data.get_element_type()`
    std::shared_ptr<ngraph::Node> negative_map = std::make_shared<opset::Convert>(
        std::make_shared<opset::Less>(data, zero_node), data.get_element_type());
    // create a positive map for 'data' node, 0 for negative values , 1 for positive values or zero
    // then convert it from boolean type to `data.get_element_type()`
    std::shared_ptr<ngraph::Node> positive_map = std::make_shared<opset::Convert>(
        std::make_shared<opset::GreaterEqual>(data, zero_node), data.get_element_type());

    // fetch alpha and beta attributes from ONNX node
    float alpha = onnx_node.get_attribute_value<float>("alpha", 1); // if 'alpha' attribute is not provided in the model, then the default value is 1
    float beta = onnx_node.get_attribute_value<float>("beta");
    // create constant node with a single element 'alpha' with type f32
    std::shared_ptr<ngraph::Node> alpha_node = opset::Constant::create(ngraph::element::f32, ngraph::Shape{}, {alpha});
    // create constant node with a single element 'beta' with type f32
    std::shared_ptr<ngraph::Node> beta_node = opset::Constant::create(ngraph::element::f32, ngraph::Shape{}, {beta});

    return {
        std::make_shared<opset::Add>(
            std::make_shared<opset::Multiply>(alpha_node, std::make_shared<opset::Multiply>(data, positive_map)),
            std::make_shared<opset::Multiply>(beta_node, std::make_shared<opset::Multiply>(data, negative_map)))};
}

void registerRoiAlign()
{

    REGISTER_OPERATOR_CUSOTM("OpExampl", op_example);
}