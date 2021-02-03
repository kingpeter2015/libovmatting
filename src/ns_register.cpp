#include "ns_register.hpp"
#include "ns_utils.hpp"
#include "ns_bmp.hpp"

using namespace std;
using namespace ns_utils_my;

/***************************************MnistUbyte*******************************************************************/
int MnistUbyte::reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = (unsigned char)(i & 255);
    ch2 = (unsigned char)((i >> 8) & 255);
    ch3 = (unsigned char)((i >> 16) & 255);
    ch4 = (unsigned char)((i >> 24) & 255);
    return (static_cast<int>(ch1) << 24) + (static_cast<int>(ch2) << 16) + (static_cast<int>(ch3) << 8) + ch4;
}

MnistUbyte::MnistUbyte(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        return;
    }
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2051)
    {
        return;
    }
    file.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    _height = (size_t)n_rows;
    file.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    _width = (size_t)n_cols;
    if (number_of_images > 1)
    {
        std::cout << "[MNIST] Warning: number_of_images  in mnist file equals " << number_of_images
                  << ". Only a first image will be read." << std::endl;
    }

    size_t size = _width * _height * 1;

    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    size_t count = 0;
    if (0 < number_of_images)
    {
        for (int r = 0; r < n_rows; ++r)
        {
            for (int c = 0; c < n_cols; ++c)
            {
                unsigned char temp = 0;
                file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
                _data.get()[count++] = temp;
            }
        }
    }

    file.close();
}

/*************************************************OCVReader**********************************************************/

OCVReader::OCVReader(const string &filename)
{
    img = cv::imread(filename);
    _size = 0;

    if (img.empty())
    {
        return;
    }

    _size = img.size().width * img.size().height * img.channels();
    _width = img.size().width;
    _height = img.size().height;
}

std::shared_ptr<unsigned char> OCVReader::getData(size_t width = 0, size_t height = 0)
{
    cv::Mat resized(img);
    if (width != 0 && height != 0)
    {
        size_t iw = img.size().width;
        size_t ih = img.size().height;
        if (width != iw || height != ih)
        {
            slog::warn << "Image is resized from (" << iw << ", " << ih << ") to (" << width << ", " << height << ")" << slog::endl;
        }
        cv::resize(img, resized, cv::Size(width, height));
    }

    size_t size = resized.size().width * resized.size().height * resized.channels();
    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < size; ++id)
    {
        _data.get()[id] = resized.data[id];
    }
    return _data;
}

/********************************************Registry************************************************************/
std::vector<Registry::CreatorFunction> Registry::_data;

Register<MnistUbyte> MnistUbyte::reg;
#ifdef NS_USE_OPENCV
Register<OCVReader> OCVReader::reg;
#else
Register<BitMap> BitMap::reg;
#endif

Reader *Registry::CreateReader(const char *filename)
{
    for (auto maker : _data)
    {
        Reader *ol = maker(filename);
        if (ol != nullptr && ol->size() != 0)
            return ol;
        if (ol != nullptr)
            ol->Release();
    }
    return nullptr;
}

void Registry::RegisterReader(CreatorFunction f)
{
    _data.push_back(f);
}

ReaderPtr::ReaderPtr(const char *imageName)
 : reader(CreateFormatReader(imageName), [](Reader *p) { p->Release(); }) 
{}

FORMAT_READER_API(Reader *)
CreateFormatReader(const char *filename)
{
    return Registry::CreateReader(filename);
}
