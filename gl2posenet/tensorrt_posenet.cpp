#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>


template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
nvinfer1::Dims mInputDims;
IExecutionContext *context;


//!
//! \brief Uses a Uff parser to create the MNIST Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
void
constructNetwork(
    SampleUniquePtr<nvuffparser::IUffParser>& parser,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string uffFileName("posenet_model/model-mobilenet_v1_101_257.uff"); //!< Filename of uff file of a network
    
    inputTensorNames.push_back("in");
    outputTensorNames.push_back("out");
    
    // Register tensorflow input
    parser->registerInput(inputTensorNames[0].c_str(),
                          nvinfer1::Dims3(1, 28, 28),
                          nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(outputTensorNames[0].c_str());

    parser->parse(uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

#if 0 /* Run in Int8 mode */
    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
#endif
}
    
int
build_tensorrt_network()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }
    constructNetwork(parser, network);
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

#if 0 /* Run in fp16 mode */
    config->setFlag(BuilderFlag::kFP16);
#endif
    
#if 0 /* Run in Int8 mode */
    config->setFlag(BuilderFlag::kINT8);
#endif

#if 0 /* use DLA core. */
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
#endif

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}



int
infer_tensorrt()
{
    // Create RAII buffer manager object
    static samplesCommon::BufferManager buffers(mEngine, 1);

    std::string inputTensorName("image");
    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));
    
    context = mEngine->createExecutionContext();
    if (!context)
    {
        return -1;
    }

    bool outputCorrect = true;
    float total = 0;

    // Try to infer each digit 0-9
//    for (int digit = 0; digit < kDIGITS; digit++)
    {
//        if (!processInput(buffers, mParams.inputTensorNames[0], digit))
//        {
//            return -1;
//        }
        // Copy data from host input buffers to device input buffers
//        buffers.copyInputToDevice();

        // Execute the inference work
        if (!context->execute(1, buffers.getDeviceBindings().data()))
        {
            return -1;
        }

        // Copy data from device output buffers to host output buffers
//        buffers.copyOutputToHost();

        // Check and print the output of the inference
//        outputCorrect &= verifyOutput(buffers, mParams.outputTensorNames[0], digit);
    }

//    total /= kDIGITS;

//    gLogInfo << "Average over " << kDIGITS << " runs is " << total << " ms."
//             << std::endl;

    return 0;
}



