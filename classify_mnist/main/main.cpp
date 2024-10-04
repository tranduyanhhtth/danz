#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

// Define the ModelInference class
class ModelInference
{
public:
    ModelInference(const std::string &model_path)
    {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Load the ONNX model
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    }

    std::vector<float> preprocess(const std::string &image_path)
    {
        // Load image using OpenCV
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            throw std::runtime_error("Could not open or find the image!");
        }

        // Resize image to 28x28
        cv::resize(image, image, cv::Size(28, 28));

        // Normalize image
        image.convertTo(image, CV_32F, 1.0 / 255);
        image = (image - 0.1307) / 0.3081;

        // Convert to tensor format
        std::vector<float> input_tensor_values(image.begin<float>(), image.end<float>());
        return input_tensor_values;
    }

    void infer(const std::string &image_path)
    {
        // Preprocess the input image
        std::vector<float> input_tensor_values = preprocess(image_path);

        // Create input tensor
        std::array<int64_t, 4> input_shape = {1, 1, 28, 28};
        size_t input_tensor_size = input_tensor_values.size();
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, &input_node_names[0], &input_tensor, 1, &output_node_names[0], 1);

        // Process output
        float *floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::vector<float> output_tensor_values(floatarr, floatarr + output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount());

        // Get the predicted class
        auto max_element_iter = std::max_element(output_tensor_values.begin(), output_tensor_values.end());
        int predicted_class = std::distance(output_tensor_values.begin(), max_element_iter);
        std::cout << "Predicted class: " << predicted_class << std::endl;
    }

private:
    std::unique_ptr<Ort::Session> session;
    const char *input_node_names[1] = {"input"};
    const char *output_node_names[1] = {"output"};
};

// Main function
int main()
{
    const std::string model_path = "mnist_model.onnx";
    const std::string image_path = "content/siz.png";

    try
    {
        ModelInference model_inference(model_path);
        model_inference.infer(image_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}