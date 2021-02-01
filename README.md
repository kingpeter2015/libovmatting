# libovmatting
A C++ library for Background Matting using openvino and BackgroundMattingV2 Model.<br>

## Application scenes
<br>
...<br>

## License
The code of libovmatting is released under the MIT License. 

## Development environment
CentOS 7<br>
Ubuntu<br>
Windows(TODO)<br>

## Usage
1 Install openvino<br>
Please refer https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html<br>
Use version: openvino_2021.2.185<br>
<br>
2 Build libovmatting library and test program<br>
Clone the project<br>
$git clone https://github.com/kingpeter2015/libovmatting.git<br>
$cd libovmatting<br>
$mkdir build<br>
$cmake ..<br>
$make<br>
<br>
4 Run the test program<br>
$cd out/
$LD_LIBRARY_PATH=./:$(LD_LIBRARY_PATH) ./sample<br>

## Reference
OpenVINO https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html <br>
BackgroundMattingV2 https://github.com/PeterL1n/BackgroundMattingV2 <br>
