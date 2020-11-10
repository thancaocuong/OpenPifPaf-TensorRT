
#include "openpifpaf/trt_utils.hpp"
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>


static void leftTrim(std::string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName)
{
    struct stat buffer;
  return (stat (fileName.c_str(), &buffer) == 0);
}


std::vector<std::string> loadListFromTextFile(const std::string filename)
{
    assert(fileExists(filename));
    std::vector<std::string> list;

    FILE* f = fopen(filename.c_str(), "r");
    if (!f)
    {
        std::cout << "failed to open " << filename;
        assert(0);
    }

    char str[512];
    while (fgets(str, 512, f) != NULL)
    {
        for (int i = 0; str[i] != '\0'; ++i)
        {
            if (str[i] == '\n')
            {
                str[i] = '\0';
                break;
            }
        }
        list.push_back(str);
    }
    fclose(f);
    return list;
}


nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, Logger& logger)
{
    // reading the model in memory
    std::cout << "Loading TRT Engine from " << planFilePath << " ..." << std::endl;
    assert(fileExists(planFilePath));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine
        = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}


std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    for (int i = 0; i < d.nbDims - 1; ++i)
    {
        s << std::setw(4) << d.d[i] << " x";
    }
    s << std::setw(4) << d.d[d.nbDims - 1];

    return s.str();
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

uint64_t get3DTensorVolume(nvinfer1::Dims inputDims)
{
    assert(inputDims.nbDims == 3);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2];
}

nlohmann::json readJsonFile(const std::string file_name)
{

    std::cout << "loading config path from  " << file_name << "." << std::endl;
    std::ifstream inputJsonFile(file_name);
    if(!inputJsonFile)
        {
            std::cout << "Failed to open file" << std::endl;
            return 0;
        }
    nlohmann::json jsonConfig;
    inputJsonFile >> jsonConfig;
    return jsonConfig;
}
int* topologyFromJson(const nlohmann::json json, const int pafChannels)
{
    int* topology = new int[2*pafChannels];
    auto skeleton = json["parser"]["skeleton"];
    for(int k=0; k<pafChannels/2; k++)
    {
        topology[4*k] = 2*k;
        topology[4*k+1] = 2*k+1;
        topology[4*k+2] = int(skeleton[k][0]) - 1;
        topology[4*k+3] = int(skeleton[k][1]) - 1;

    }
    return topology;

}

int** create2DArray(int height, int width)
    {
      int** array2D = 0;
      array2D = new int*[height];

      for (int h = 0; h < height; h++)
      {
            array2D[h] = new int[width];

            for (int w = 0; w < width; w++)
            {
                  array2D[h][w] = w + width * h;
            }
      }

      return array2D;
    }

vector<vector<int>> skeletonFromJson(const nlohmann::json json, const int pafChannels)
{
    // skeleton : num_parts x 2
    vector<vector<int>> skeleton(pafChannels/2, vector<int>(2, -1));
    auto json_skeleton = json["parser"]["skeleton"];
    for(int k=0; k<pafChannels/2; k++)
    {
        skeleton[k][0] = int(json_skeleton[k][0]) - 1;
        skeleton[k][1] = int(json_skeleton[k][1]) - 1;
    }
    return skeleton;

}

void serialize(const string outFileName, float* arr, int rows, int cols) {
    ofstream outfile;
    outfile.open (outFileName, ios::out | ios::trunc);
    for (int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                outfile << std::fixed << std::setprecision(3) << arr[i*cols + j] << " ";
            outfile << std::endl;
        }
}
