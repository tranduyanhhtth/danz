#ifndef MODEL_CONVERTER_H
#define MODEL_CONVERTER_H

#include "export_pth.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

class ModelConverter
{
public:
    ModelConverter(const string &model_path, const string &onnx_path);
    void load_model();
    void create_dummy_input();
    void export_to_onnx();

private:
    string model_path_;
    string onnx_path_;
    shared_ptr<torch::jit::script::Module> model_;
    torch::Tensor dummy_input_;
};

#endif