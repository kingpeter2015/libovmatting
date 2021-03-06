#include "samples.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    try
    {
        //Inference_Camera();
        //Inference_Video();
        //Inference_Modnet();
        Inference_demo1(argc, argv);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "main(int argc, char *argv[]):" << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
