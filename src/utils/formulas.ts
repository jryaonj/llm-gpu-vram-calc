import type { QuantType } from '../types';

export function computeModelVramGB(
  paramsB: number,
  quant: QuantType
): number {
  // Your exact formula here
  const bytesPerParam = quant === 'fp16'
    ? 2 : quant === 'fp8'
    ? 1 : quant === 'int8'
    ? 1 : 0.5;
  return (paramsB * 1e9 * bytesPerParam) / (1024 ** 3);
}

export function computeKvCacheVramGB(
  hiddenSize: number,
  maxLength: number,
  quant: QuantType
): number {
  const bytes = quant === 'fp16'
    ? 2 : quant === 'fp8'
    ? 1 : quant === 'int8'
    ? 1 : 0.5;
  const totalBytes = 2 * hiddenSize * bytes * maxLength;
  return totalBytes / (1024 ** 3);
}