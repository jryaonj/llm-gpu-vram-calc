import type { ModelDef } from '../types';

/**
 * Model definitions for Qwen3 series models.
 * Each model definition includes parameters such as size, quantization type, and architecture details.
 */

export const modelDefs: ModelDef[] = [
  {
    name: 'Qwen3-0.6B',
    paramsB: 0.6,
    hiddenSize: 8 * 256,
    activeParamsB: 0.6,
    totalParamsB: 0.6,
    modelSizeGB: 0.31,
    perKVsizeFp8: (28 * 8 * 256) / 2, // 28672 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 28,
    numKVHeads: 8,
    headDim: 256,
  },
  {
    name: 'Qwen3-1.7B',
    paramsB: 1.7,
    hiddenSize: 8 * 128,
    activeParamsB: 1.7,
    totalParamsB: 1.7,
    modelSizeGB: 0.90,
    perKVsizeFp8: (28 * 8 * 128) / 2, // 14336 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 28,
    numKVHeads: 8,
    headDim: 128,
  },
  {
    name: 'Qwen3-4B',
    paramsB: 4.0,
    hiddenSize: 8 * 128,
    activeParamsB: 4.0,
    totalParamsB: 4.0,
    modelSizeGB: 2.10,
    perKVsizeFp8: (36 * 8 * 128) / 2, // 18432 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 36,
    numKVHeads: 8,
    headDim: 128,
  },
  {
    name: 'Qwen3-30B-A3B',
    paramsB: 30.53,
    hiddenSize: 4 * 128,
    activeParamsB: 3.0,
    totalParamsB: 30.53,
    modelSizeGB: 16.70,
    perKVsizeFp8: (48 * 4 * 128) / 2, // 12288 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 64,
    layers: 48,
    numKVHeads: 4,
    headDim: 128,
  },
  {
    name: 'Qwen3-8B',
    paramsB: 8.2,
    hiddenSize: 8 * 128,
    activeParamsB: 8.2,
    totalParamsB: 8.2,
    modelSizeGB: 4.86,
    perKVsizeFp8: (36 * 8 * 128) / 2, // 18432 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 36,
    numKVHeads: 8,
    headDim: 128,
  },
  {
    name: 'Qwen3-14B',
    paramsB: 14.8,
    hiddenSize: 8 * 128,
    activeParamsB: 14.8,
    totalParamsB: 14.8,
    modelSizeGB: 10.15,
    perKVsizeFp8: (40 * 8 * 128) / 2, // 20480 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 16,
    layers: 40,
    numKVHeads: 8,
    headDim: 128,
  },
  {
    name: 'Qwen3-32B',
    paramsB: 32.8,
    hiddenSize: 8 * 128,
    activeParamsB: 32.8,
    totalParamsB: 32.8,
    modelSizeGB: 19.45,
    perKVsizeFp8: (64 * 8 * 128) / 2, // 32768 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 64,
    numKVHeads: 8,
    headDim: 128,
  },
  {
    name: 'Qwen3-235B-A22B',
    paramsB: 235.09,
    hiddenSize: 4 * 128,
    activeParamsB: 22.0,
    totalParamsB: 235.09,
    modelSizeGB: 128.56,
    perKVsizeFp8: (94 * 4 * 128) / 2, // 24064 bytes
    quantType: 'fp8',
    quantBits: 8,
    awqGroup: 32,
    layers: 94,
    numKVHeads: 4,
    headDim: 128,
  },
];