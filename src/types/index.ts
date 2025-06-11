export type QuantType = 'fp64' | 'fp32' | 'fp16' | 'fp8' | 'int8' | 'int4';

export interface GPUCard {
  name: string;
  vramGb: number;               // total device VRAM in GB  
  memoryBandwidthGBs: number;   // GB/s
  kvQuantType?: string;        // quantization type for KV cache, e.g., 'fp8', 'int8'
  processPower: Partial<Record<QuantType, number>>;
  // processPower: {
  //   fp16: number;               // TFLOPS
  //   fp8?: number;               // optional, if different from fp16
  //   int8?: number;
  // };
}

export interface ModelDef {
    name: string;                 // model name
    paramsB: number;              // total parameters in billions
    hiddenSize: number;           // hidden layer size
    activeParamsB: number;        // active parameters in billions
    totalParamsB: number;         // total parameters in billions
    modelSizeGB: number;          // model size in GB
    perKVsizeFp8: number;         // per KV size in bytes when using fp8
    quantType?: QuantType;        // quantization type
    quantBits?: number;           // quantization bits
    awqGroup?: number;            // AWQ group size
    awqScale?: number;            // scale factor bits per 32-group
    awqZeroPoint?: number;        // zero point bits per 32-group
    layers?: number;              // number of layers
    numKVHeads?: number;          // number of KV heads
    headDim?: number;             // head dimension
}

export interface CalcResults {
  usableVram: number;
  usableKvCacheVram: number;      // usable VRAM for KV cache
  reservedVram: any;
  modelVram: number;
  kvCacheVram: number;
  totalVram: number;
  genSpeed: number;
  promptSpeed: number;
  membwScaling: number;
  ppScaling: number;
  sharedGen: number;
  sharedPrompt: number;
  maxTokenCountSimultaneous: number,
  fullLengthGenCount: number,
  error: string | null;
}