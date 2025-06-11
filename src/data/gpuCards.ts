import type { GPUCard } from '../types';

export const gpuCards: GPUCard[] = [
  // Populate from XLS:
  { name: 'A100 80GB', vramGb: 80, memoryBandwidthGBs: 2039, processPower: { fp16: 312, fp8: 624 } },
  { name: 'A100 40GB', vramGb: 40, memoryBandwidthGBs: 1555, processPower: { fp16: 312, fp8: 624 } },
  { name: 'H100 80GB', vramGb: 80, memoryBandwidthGBs: 3350, processPower: { fp16: 989, fp8: 1978 } },
  { name: 'H100 PCIe', vramGb: 80, memoryBandwidthGBs: 2000, processPower: { fp16: 989, fp8: 1978 } },
  { name: 'L40', vramGb: 48, memoryBandwidthGBs: 864, processPower: { fp16: 90.5 } },
  { name: 'L40S', vramGb: 48, memoryBandwidthGBs: 864, processPower: { fp16: 91.5 } },
  { name: 'RTX 4090', vramGb: 24, memoryBandwidthGBs: 1008, processPower: { fp16: 82.6 } },
];