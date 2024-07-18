2024-07-18 23:18:18.493995: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.

2024-07-18 23:18:18.498687: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.

2024-07-18 23:18:18.510164: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered

2024-07-18 23:18:18.533923: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered

2024-07-18 23:18:18.540729: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

2024-07-18 23:18:18.555297: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.

To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

2024-07-18 23:18:22.127116: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:589 in _run_script                                      

                                                                                

  /mount/src/imageprocessing/app.py:15 in <module>                              

                                                                                

     12                                                                         

     13 # Load the tokenizer                                                    

     14 with open('tokenizer.pkl', 'rb') as f:                                  

  ❱  15 │   tokenizer = pickle.load(f)                                          

     16                                                                         

     17 # Load the captioning model                                             

     18 model = load_model('best_model.h5')                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'keras.src.preprocessing'
