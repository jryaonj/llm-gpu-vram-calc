# LLM calculation

to estimate the speed, using following one

## prompting token speed

prompting speed determines how fast an LLM can **read** some existed content.

determined by the process_power of device, and the scale of LLM-model

### equation
the calculation is like

```
prompt_speed = gpu/tpu/xpu/cpu_pp  /  full_parameters_of_a_model * num_ratio * quantization_ratio
tok/s,aggregated    FP16 TFLOPS                                                 model_param_fp_to_fp16
```

### example 1 - Qwen3-4B-AWQ on 1x NVIDIA RTX 3090

* Qwen3-4B-AWQ model_parameters = 4.02B
* RTX3090 fp16 = 35.58 TFLOPS 

modelfp_to_fp16 estimated as sqrt(2)

```
prompt_speed =  35.58 / 4.02 * 1000 / sqrt(2)
             =  6258 tok/s
```

thus in the most ideal situation, the total (multiple task aggreated) throughput prompt speed is **6258 toks/s**

## generate token speed

generating speed determines how fast an LLM can **output** new generated content.

determined by the bandwith of device, and the scale of LLM-model

### equation

```
generate_speed = gpu/tpu/xpu/cpu_membw  /  active_parameters_of_a_model * quantization_ratio
tok/s,aggregated     GB/s                                                 model_param_fp_to_byte
```

### example 1 - Qwen3-4B-AWQ on 1x NVIDIA RTX 3090

* Qwen3-4B-AWQ active_model_parameters = 4.02B
* AWQ byte ~= int4 = 0.5 byte 
* RTX3090 membw = 936.2 GB/s

```
generate_speed =  936.2 / 4.02 /0.5
               =  466 tok/s
```

## [difficult] LLM memory consumption

LLM memory consumption is determined by the model, the task (max_input_content_length), the quantization method, the parallel number

we would introduce below calculation

* model-weight
* model-kvcache (determin by per request maxinum length)

after that , we can do an estimation on how much token could a typical device/GPU process at once, aka the capacity for a typical model on specified device/GPU

### model-weight memory usage

using the following equation to do so

```
model_weight_size = num_of_model_parameter * size_per_param
    GB                      
```

typically, size_per_param is determined by different model_weight_quantization method
for **Qwen3-30B-A3B-AWQ** ( param=30.53, AWQ-group = 64 ), such **size_per_param** is 

```
size_per_param = raw_value + index_stuff
                  int4 / fp8     3/AWQ-group
               = 0.5  + 3/64
```

thus the model size 
```
model_weight_size = 30.53 * ( 0.5  + 3/64)
                  = 16.70 GB
```


### key-value cache memory usage

k-v cache consumption determines how many token such model need to process at once, which is a project-oriented stuff.
when the capacity of the device is not enough, then such a LLM calculation won't be carried on.

k-v cache is determined by these factors

* model_parameters ( layer, num_kv_heads, head-dim )
* model_weight_quantization (less model usage)
* kv_quantization (allow more token)


```
kvcache_size_per_req = head_dim * num_kv_head * layer * 2 * num_max_length / 2^30 * quantization_ratio
       GB                                         (k+v, each has one)   byte_to_gigabyte
```


for **Qwen3-30B-A3B** model, `layer=48, num_kv_heads=4, head_dim=128`, 
suppose a single request max length is same as train length `max_len=32768`, and kv-cache quantization method is `fp8_e4m3`, then

```
kvcache_size = 128 * 4 * 48 * 2 * 32768 / 2^30 * 1 (fp8=1byte)
             = 1.5 GB
```

### make full capacity estimation 

after we know how to calculate the memory consumption, we could make an estimation on how much token could a typical system support

suppose now we have the following device, and select such model

**NVIDIA RTX 3090**
* vmem = 24 GB
* fp16 = 35.58 TFLOPS (boosted)
* membs = 936.2 GB/s

**Qwen3-30B-A3B-AWQ**
* model_param =  30.53 B
* model_active_param = 3 B
* awq_group = 64
* layer=48, num_kv_heads=4, head_dim=128

we set the model serve within kv-cache quantization `fp8_e4m3` and `max_model_length=32768`,
`gpu_util=0.825` (82.5% vmem is used in LLM)

then, we could easily get those result

```
generate_speed = 624 tok/s (upper bound)
prompt_speed   = 824 tok/s (upper bound)
model_weight_size = 16.70 GB
model_kv_size_per_request = 1.50 GB (per 32768 tok, fp8_e4m3 quantization)
total_token_capacity = ( vmem * gpu_util - model_weight_size ) / model_kv_size_per_request * max_model_length
                     = ( 24 * 0.825 - 16.7 ) / 1.50 * 32768
                     = 67720 token
```

and the parallel ratio on worst case, is

```
p_ratio = total_token_capacity / max_model_length
        = 67720 / 32768
        = 2.06x
```
