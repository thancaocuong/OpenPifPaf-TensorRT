#include "openpifpaf/tensorrt_net.hpp"
#include "openpifpaf/args_parser.hpp"
#include "openpifpaf/poseplugin_lib.hpp"
#include <iostream>
#include <string>

using namespace argsParser;

int main(int argc, char** argv)
{
    parser::ADD_ARG_INT("batchsize",Desc("batch size for input"),DefaultValue("1"));
    parser::ADD_ARG_INT("pwidth",Desc("processing width"),DefaultValue("640"));
    parser::ADD_ARG_INT("pheight",Desc("processing height"),DefaultValue("480"));
    parser::ADD_ARG_INT("fullframe", Desc("full frame processing"), DefaultValue("1"));
    parser::ADD_ARG_STRING("config", Desc("config file path"), DefaultValue("../../configs/net_configs.json"));
    parser::ADD_ARG_STRING("engine", Desc("path to engine"), DefaultValue("./pifpaf.engine"));
    parser::ADD_ARG_STRING("imgpath", Desc("path to image"), DefaultValue("./1.jpg"));
    parser::ADD_ARG_BOOL("imshow", Desc("show image or not"), DefaultValue("0"));

    parser::parseArgs(argc,argv);

    int batchsize = parser::getIntValue("batchsize");
    int processingWidth = parser::getIntValue("pwidth");
    int processingHeight = parser::getIntValue("pheight");
    int fullFrame = parser::getIntValue("fullframe");
    std::string configFilePath =  parser::getStringValue("config");
    std::string engineName =  parser::getStringValue("engine");
    std::string image_path =  parser::getStringValue("imgpath");
    bool is_imshow = parser::getBoolValue("imshow");

    PosePluginInitParams initParams {   .processingWidth=processingWidth,
                                        .processingHeight=processingHeight,
                                        .fullFrame=fullFrame,
                                        .configFilePath=configFilePath,
                                        .engineFilePath=engineName};

    PosePluginCtx* estimator = PosePluginCtxInit(&initParams, batchsize);

    auto image1 = cv::imread(image_path);
    std::vector<cv::Mat> imgvector;
    for(int i=0; i<batchsize; i++)
        imgvector.push_back(image1);
    for(int i=0; i<100; i++){
    auto batch_results = PosePluginProcess(estimator, imgvector);
    }
    std::cout << "Done inference" << std::endl;
    PosePluginCtxDeinit(estimator);
    return 0;
}
