# flash-attn build
cd flash-attention
python setup.py install

# ops build
cd csrc
cd rotary && python setup.py install && cd ..
cd layer_norm && python setup.py install && cd ..
cd fused_dense_lib && python setup.py install && cd ..