# libovmatting
A C++ library for Background Matting using openvino and deep learning models.<br>

Now support models such as BackgroundMattingV2, MODNet.<br>

## Application scenes
<br>
...<br>

## License
The code of libovmatting is released under the MIT License. 

## Development environment
CentOS 7<br>
Ubuntu<br>
Windows 10<br>

## Usage
### 1 Install openvino<br>
Please refer https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html<br>
Use version: openvino_2021.2.185<br>
<br>
### 2 Build libovmatting library and test program<br>
Clone the project<br>
$git clone https://github.com/kingpeter2015/libovmatting.git<br>

#### Ubuntu/CentOS

$cd libovmatting <br>
$mkdir build <br>
$cmake .. <br>
$make <br>
<br>
Run the test program <br>
$cd ./build/ <br>
$LD_LIBRARY_PATH=./:$(LD_LIBRARY_PATH) ./sample<br>

#### Windows 10
Install Visual Studio 2019 <br>
open Win64\ovmatting.sln <br>
build and run in the Visual Studio IDE <br>

## Reference
OpenVINO https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html <br>
BackgroundMattingV2 https://github.com/PeterL1n/BackgroundMattingV2 <br>
MODNet https://github.com/ZHKKKe/MODNet <br>
