# export LD_LIBRARY_PATH=./:./build/:/opt/intel/openvino_2021/inference_engine/lib/intel64/:/opt/intel/openvino_2021/opencv/lib/:/opt/intel/openvino_2021/deployment_tools/ngraph/lib/:/opt/intel/openvino_2021/inference_engine/external/tbb/lib/:$LD_LIBRARY_PATH

# echo $LD_LIBRARY_PATH

./build/sample -model ./share/modnet.xml \
         -bin ./share/modnet.bin \
         -dev CPU \
         -src ./share/bxg1.mp4 \
         -sbgr ./share/bxg1.png \
         -dst ./share/dst.mp4 \
         -dbgr ./share/replace.jpg \
         -in_width 512 \
         -in_height 512 \
         -method 1 \
         -interval 4 \
         -cpu_thread 2