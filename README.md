# libovmatting
A C++ library for Background Matting using openvino and deep learning models.<br>

Now support models such as BackgroundMattingV2, MODNet.<br>

## 1 Application scenes
<br>
...<br>

## 2 License
The code of libovmatting is released under the MIT License. 

## 3 Development environment
CentOS 7<br>
Ubuntu<br>
Windows 10<br>

## 4 Usage
### 4.1 Install openvino<br>
Please refer https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html<br>
Use version: openvino_2021.2.185<br>
<br>
### 4.2 Build libovmatting library and test program<br>
Clone the project<br>
$git clone https://github.com/kingpeter2015/libovmatting.git<br>

#### 4.2.1 Ubuntu/CentOS

$cd libovmatting <br>
$mkdir build <br>
$cmake .. <br>
$make <br>
<br>
Run the test program <br>
$cd ./build/ <br>
$LD_LIBRARY_PATH=./:$(LD_LIBRARY_PATH) ./sample<br>

#### 4.2.2 Windows 10
Install Visual Studio 2019 <br>
open Win64\ovmatting.sln <br>
build and run in the Visual Studio IDE <br>

## 5 About Source Code

### 5.1 Directory Structure

* Win64: Windows Solution and projects <br/>
* include: contains header files. **ovmatter.h is api header file** <br/>
* libovmatting: CMakefiles for compile libovmatting.so under linux <br/>
* samples: four samples using libovmatting library <br/>
* share: contains model files which are used by openvino inference engine. <br/>
* src: contains source code files <br/>

main.cpp:  startup point of demos <br/>

### 5.2 Please Attention !!!

* 'include/ovmatter.h' is one only sdk header file which exposes api to applications.
* libovmatting library only supports two matting method: METHOD_BACKGROUND_MATTING_V2 and METHOD_MODNET. They are defined in include/ovmatter.h
*  METHOD_BACKGROUND_MATTING_V2 method uses cnn model defined 'pytorch_mobilenetv2.bin/pytorch_mobilenetv2.xml'.
*  METHOD_MODNET method uses cnn model defined in 'modnet.bin/modnet.xml' 

if you want to use your own model, you can do following steps:
* Add a enum item in enum OV_MATTER_API MattingMethod which is defined in include/ovmatter.h
* declare and implement a new ovmatter class, and inherit 'MatterBaseImpl', then write a new cnn that uses your own models

That's all, have fun !

## 6 Reference
OpenVINO https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html <br>
BackgroundMattingV2 https://github.com/PeterL1n/BackgroundMattingV2 <br>
MODNet https://github.com/ZHKKKe/MODNet <br>
