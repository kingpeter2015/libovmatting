#ifndef __NS__REGISTER__HPP___
#define __NS__REGISTER__HPP___

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <fstream>
#include <opencv2/opencv.hpp>

#if defined(_WIN32)
#ifdef IMPLEMENT_FORMAT_READER
#define FORMAT_READER_API(type) extern "C" __declspec(dllexport) type
#else
#define FORMAT_READER_API(type) extern "C" type
#endif
#elif (__GNUC__ >= 4)
#ifdef IMPLEMENT_FORMAT_READER
#define FORMAT_READER_API(type) extern "C" __attribute__((visibility("default"))) type
#else
#define FORMAT_READER_API(type) extern "C" type
#endif
#else
#define FORMAT_READER_API(TYPE) extern "C" TYPE
#endif

#define NS_USE_OPENCV 1

namespace ovlib
{

    /**
 * \class FormatReader
 * \brief This is an abstract class for reading input data
 */
    class Reader
    {
    protected:
        /// \brief height
        size_t _height = 0;
        /// \brief width
        size_t _width = 0;
        /// \brief data
        std::shared_ptr<unsigned char> _data;

    public:
        /**
     * \brief Get width
     * @return width
     */
        size_t width() const { return _width; }

        /**
     * \brief Get height
     * @return height
     */
        size_t height() const { return _height; }

        /**
     * \brief Get input data ptr
     * @return shared pointer with input data
     * @In case of using OpenCV, parameters width and height will be used for image resizing
     */
        virtual std::shared_ptr<unsigned char> getData(size_t width = 0, size_t height = 0) = 0;

        /**
     * \brief Get size
     * @return size
     */
        virtual size_t size() const = 0;

        virtual void Release() noexcept = 0;
    };

    class ReaderPtr
    {
    public:
        explicit ReaderPtr(const char *imageName);
        
        /**
     * @brief dereference operator overload
     * @return Reader
     */
        Reader *operator->() const noexcept
        {
            return reader.get();
        }

        /**
     * @brief dereference operator overload
     * @return Reader
     */
        Reader *operator*() const noexcept
        {
            return reader.get();
        }

        Reader *get()
        {
            return reader.get();
        }

    protected:
        std::unique_ptr<Reader, std::function<void(Reader *)>> reader;
    };

    /**
 * \class Registry
 * \brief Create reader from fabric
 */
    class Registry
    {
    private:
        typedef std::function<Reader *(const std::string &filename)> CreatorFunction;
        static std::vector<CreatorFunction> _data;

    public:
        /**
     * \brief Create reader
     * @param filename - path to input data
     * @return Reader for input data or nullptr
     */
        static Reader *CreateReader(const char *filename);

        /**
     * \brief Registers reader in fabric
     * @param f - a creation function
     */
        static void RegisterReader(CreatorFunction f);
    };

    /**
 * \class Register
 * \brief Registers reader in fabric
 */
    template <typename To>
    class Register
    {
    public:
        /**
     * \brief Constructor creates creation function for fabric
     * @return Register object
     */
        Register()
        {
            Registry::RegisterReader([](const std::string &filename) -> Reader * {
                return new To(filename);
            });
        }
    };

    /**
 * \class OCVMAT
 * \brief OpenCV Wraper
 */
    class OCVReader : public Reader
    {
    private:
        cv::Mat img;
        size_t _size;
        static Register<OCVReader> reg;

    public:
        /**
    * \brief Constructor of BMP reader
    * @param filename - path to input data
    * @return BitMap reader object
    */
        explicit OCVReader(const std::string &filename);
        virtual ~OCVReader()
        {
        }

        /**
    * \brief Get size
    * @return size
    */
        size_t size() const override
        {
            return _size;
        }

        void Release() noexcept override
        {
            delete this;
        }

        std::shared_ptr<unsigned char> getData(size_t width, size_t height) override;
    };

    /**
 * \class MnistUbyte
 * \brief Reader for mnist db files
 */
    class MnistUbyte : public Reader
    {
    private:
        int reverseInt(int i);

        static Register<MnistUbyte> reg;

    public:
        /**
     * \brief Constructor of Mnist reader
     * @param filename - path to input data
     * @return MnistUbyte reader object
     */
        explicit MnistUbyte(const std::string &filename);
        virtual ~MnistUbyte()
        {
        }

        /**
     * \brief Get size
     * @return size
     */
        size_t size() const override
        {
            return _width * _height * 1;
        }

        void Release() noexcept override
        {
            delete this;
        }

        std::shared_ptr<unsigned char> getData(size_t width, size_t height) override
        {
            if ((width * height != 0) && (_width * _height != width * height))
            {
                std::cout << "[ WARNING ] Image won't be resized! Please use OpenCV.\n";
                return nullptr;
            }
            return _data;
        }
    };

} // namespace ovlib


/**
 * \brief Function for create reader
 * @return FormatReader pointer
 */
FORMAT_READER_API(ovlib::Reader *)
CreateFormatReader(const char *filename);

#endif