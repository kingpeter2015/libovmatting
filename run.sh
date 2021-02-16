#/bin/bash

export LD_LIBRARY_PATH=./:./build/:/opt/intel/openvino_2021/inference_engine/lib/intel64/:/opt/intel/openvino_2021/opencv/lib/:/opt/intel/openvino_2021/deployment_tools/ngraph/lib/:/opt/intel/openvino_2021/inference_engine/external/tbb/lib/:$LD_LIBRARY_PATH

# echo $LD_LIBRARY_PATH

./build/sample -model ./share/pytorch_mobilenetv2.xml \
         -bin ./share/pytorch_mobilenetv2.bin \
         -dev CPU \
         -src ./share/src.mp4 \
         -sbgr ./share/src.png \
         -dst ./share/dst.mp4 \
         -dbgr ./share/replace.jpg \
		 -in_width 320 \
		 -in_height 180 \
		 -method 0 \
		 -interval 3 \
         -cpu_thread 0 \
         -cpu_stream 1