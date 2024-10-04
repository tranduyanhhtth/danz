#include "export_pth.h"
// Constants
const int64_t kBatchSize = 32;
const double kLearningRate = 0.01;
const int64_t kNumEpochs = 20;

// DataLoader class to handle data loading
class DataLoader
{
public:
    DataLoader()
    {
        train_dataset = torch::data::datasets::MNIST("./data")
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());
        train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            move(train_dataset), kBatchSize);

        val_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
        val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            move(val_dataset), kBatchSize);
    }

    auto &get_train_loader() { return train_loader; }
    auto &get_val_loader() { return val_loader; }

private:
    torch::data::datasets::MNIST train_dataset, val_dataset;
    shared_ptr<torch::data::DataLoader<torch::data::datasets::MNIST>> train_loader, val_loader;
};

// Define the CNN model
struct BetterModel : torch::nn::Module
{
    BetterModel()
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 32, 3).padding(1));
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 3).padding(1));
        conv3 = register_module("conv3", torch::nn::Conv2d(64, 128, 3).padding(1));
        pool = register_module("pool", torch::nn::MaxPool2d(2));
        fc1 = register_module("fc1", torch::nn::Linear(128 * 3 * 3, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, 10));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = pool(torch::relu(conv1(x)));
        x = pool(torch::relu(conv2(x)));
        x = pool(torch::relu(conv3(x)));
        x = x.view({-1, 128 * 3 * 3});
        x = torch::relu(fc1(x));
        x = dropout(x);
        x = torch::relu(fc2(x));
        x = dropout(x);
        x = fc3(x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

// Trainer class to handle training and validation
class Trainer
{
public:
    Trainer(shared_ptr<BetterModel> model, torch::Device device)
        : model(model), device(device), optimizer(model->parameters(), torch::optim::SGDOptions(kLearningRate)) {}

    void train(DataLoader &data_loader)
    {
        model->train();
        size_t batch_idx = 0;
        for (auto &batch : *data_loader.get_train_loader())
        {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, targets);
            loss.backward();
            optimizer.step();
            if (++batch_idx % 100 == 0)
            {
                cout << "Train Epoch: " << setw(3) << batch_idx << " Loss: " << loss.item<double>() << endl;
            }
        }
    }

    void validate(DataLoader &data_loader)
    {
        torch::NoGradGuard no_grad;
        model->eval();
        double val_loss = 0;
        int correct = 0;
        for (const auto &batch : *data_loader.get_val_loader())
        {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = model->forward(data);
            val_loss += torch::nn::functional::cross_entropy(output, targets).item<double>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().item<int64_t>();
        }
        val_loss /= data_loader.get_val_loader()->size().value();
        cout << "Validation set: Average loss: " << val_loss << ", Accuracy: " << correct << "/" << data_loader.get_val_loader()->size().value() << " (" << 100. * correct / data_loader.get_val_loader()->size().value() << "%)\n";
    }

    void Show_trainning()
    {
        torch::manual_seed(1);
        torch::Device device(torch::kCUDA);

        DataLoader data_loader;
        auto model = make_shared<BetterModel>();
        model->to(device);

        Trainer trainer(model, device);

        for (int epoch = 1; epoch <= kNumEpochs; ++epoch)
        {
            cout << "Epoch: " << epoch << endl;
            trainer.train(data_loader);
            trainer.validate(data_loader);
        }

        torch::save(model, "mnist_model.pt");
        cout << "Model saved to mnist_model.pt" << endl;
    }

private:
    shared_ptr<BetterModel> model;
    torch::Device device;
    torch::optim::SGD optimizer;
};