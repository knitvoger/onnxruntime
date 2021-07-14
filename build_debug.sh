./build.sh --build_shared_lib --parallel --skip_tests --config Debug  --build_dir build_cpu_debug
cp build_cpu_debug/Debug/libonnxruntime.so.1.8.0 ~/git/GPU-Wavenet/onnxwrapper/libonnxruntime.so