#include "export_onnx.h"
#include "export_pth.h"

class ModelConverter
{
public:
    ModelConverter(const string &model_path, const string &onnx_path) : model_path(model_path), onnx_path(onnx_path) {}

    void convert()
    {
        // Load model
        auto model = make_shared<BetterModel>();
        torch::load(model, model_path);
        model->eval();

        // Create dummy input
        auto dummy_input = torch::randn({1, 1, 28, 28});

        // Export to ONNX
        torch::jit::script::Module module;
        module.register_module("model", model);
        module.to(torch::kCPU);
        module.eval();
        module.save(onnx_path);

        cout << "Model đã được chuyển đổi sang ONNX và lưu tại " << onnx_path << endl;
    }

    void init()
    {
        string model_path = "mnist_model.pth";
        string onnx_path = "mnist_model.onnx";

        ModelConverter converter(model_path, onnx_path);
        converter.convert();
    }

private:
    string model_path;
    string onnx_path;
};