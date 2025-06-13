import { useState, useEffect } from 'react';
import { Settings, ChevronDown, Cpu, BarChart3, Zap, HardDrive, Users } from 'lucide-react';

import type { GPUCard, ModelDef, CalcResults } from '../types/index.ts';
import { gpuCards } from '../data/gpuCards.ts';
import { modelDefs } from '../data/modelDefs.ts';

// Helper functions
function computeModelVramGB(model: ModelDef, quant: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
  if (quant === model.quantType) {
    return model.modelSizeGB;
  } else {
    const bytesPerParam = quant === 'fp16' ? 2 : quant === 'fp8' ? 1 : quant === 'int8' ? 1 : 0.5;
    const quantFactor = quant === 'fp8' ? (0.5 + 3/32)/0.5 : quant === 'int4' ? (0.5 + 3/32)/0.5 : quant === 'fp16' ? 1 : 1;
    return model.totalParamsB * bytesPerParam * quantFactor;
  }
}

function computeKvCacheVramGB(_hiddenSize: number, maxLength: number, quant: 'fp16' | 'fp8' | 'int8' | 'int4', model: ModelDef, _card: GPUCard): number {
  const bytesPerValue = quant === 'fp16' ? 2 : quant === 'fp8' ? 1 : quant === 'int8' ? 1 : quant === 'int4' ? 0.5 : 4;
  const baseKVSize = model.perKVsizeFp8;
  return (baseKVSize * bytesPerValue * maxLength * 2) / (1024 * 1024 * 1024);
}

// Estimate model size (GB) from total parameters and quantization type
function estimateModelSizeGB(totalParamsB: number, quantType: 'fp16' | 'fp8' | 'int8' | 'int4'): number {
  const bytesPerParam = quantType === 'fp16' ? 2 : quantType === 'fp8' ? 1 : quantType === 'int8' ? 1 : 0.5;
  // Safety factor for fp8 / int4 worst-case
  const quantFactor = (quantType === 'fp8' || quantType === 'int4') ? ((0.5 + 3 / 32) / 0.5) : 1;
  const sizeGB = totalParamsB * bytesPerParam * quantFactor;
  return Math.round(sizeGB * 100) / 100;
}

// Compute per-token KV cache size (bytes) for FP8 given transformer hyper-parameters
function computeKvSizeFp8FromParams(layers: number, numKVHeads: number, headDim: number): number {
  return layers * numKVHeads * headDim;
}

// Model color mapping
const modelColors: Record<string, string> = {
  'Qwen3-0.6B': '#a855f7',
  'Qwen3-1.7B': '#a855f7',
  'Qwen3-4B':   '#a855f7',
  'Qwen3-8B':   '#a855f7',
  'Qwen3-14B':  '#a855f7',
  'Qwen3-32B':  '#a855f7',
  'Qwen3-30B-A3B': '#a855f7',
  'Qwen3-235B-A22B': '#a855f7',
};

// GPU vendor colors
const getGpuColor = (name: string): string => {
  if (name.includes('NVIDIA')) return '#76b900'; // NVIDIA green
  if (name.includes('AMD')) return '#ed1c24'; // AMD red
  return '#6b7280'; // default gray
};

const stripVendor = (name: string): string => {
  return name.replace(/^NVIDIA\s+/i, '').replace(/^AMD\s+/i, '').trim();
};

const inferArchitecture = (gpu: GPUCard): string => {
  if (gpu.architecture) return gpu.architecture;
  const n = gpu.name.toLowerCase();
  if (n.includes('h100') || n.includes('h20') || n.includes('hopper')) return 'Hopper';
  if (n.includes('l40') || n.includes('a40') || n.includes('a800') || n.includes('a100')) return 'Ampere';
  if (n.match(/rtx4/)) return 'Ada';
  if (n.match(/rtx3/)) return 'Ampere';
  if (n.match(/rtx2/)) return 'Turing';
  if (n.includes('v100')) return 'Volta';
  if (n.includes('1080')) return 'Pascal';
  if (n.includes('rx7') || n.includes('r7')) return 'RDNA3';
  return '';
};

export default function LLMVRAMCalculator() {
  // Default selections
  const defaultCard = gpuCards.find(c => c.name === 'NVIDIA RTX3090 24G') ?? gpuCards[0];
  const defaultModel = modelDefs.find(m => m.name === 'Qwen3-8B') ?? modelDefs[0];

  // State
  const [selectedCard, setSelectedCard] = useState<GPUCard | null>(defaultCard);
  const [selectedModel, setSelectedModel] = useState<ModelDef | null>(defaultModel);
  const [quantType, setQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('int4');
  const [kvQuantType, setKvQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>(defaultCard?.kvQuantType as any || 'fp8');
  const [maxLength, setMaxLength] = useState<number>(8192);
  const [userCount, setUserCount] = useState<number>(10);
  const [vramUtilProportion, setVramUtilProportion] = useState<number>(0.9);
  const [minReserveVramGB, setMinReserveVramGB] = useState<number>(2);
  const [parallelGPUs, setParallelGPUs] = useState<number>(1);
  const [results, setResults] = useState<CalcResults | null>(null);
  const [dropdownOpenGPU, setDropdownOpenGPU] = useState(false);
  const [dropdownOpenModel, setDropdownOpenModel] = useState(false);
  const [useCustomGPU, setUseCustomGPU] = useState(false);
  const [customVramGB, setCustomVramGB] = useState<number>(24);
  const [customMemoryBandwidthGBs, setCustomMemoryBandwidthGBs] = useState<number>(600);
  const [customProcessPowerFP16, setCustomProcessPowerFP16] = useState<number>(30);
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [customTotalParamsB, setCustomTotalParamsB] = useState<number>(7);
  const [customModelSizeGB, setCustomModelSizeGB] = useState<number>(4);
  const [customActiveParamsB, setCustomActiveParamsB] = useState<number>(7);
  const [customPerKVsizeFp8, setCustomPerKVsizeFp8] = useState<number>(65536);

  // Custom parameters for detailed estimation
  const [customLayers, setCustomLayers] = useState<number>(32);
  const [customNumKVHeads, setCustomNumKVHeads] = useState<number>(8);
  const [customHeadDim, setCustomHeadDim] = useState<number>(128);

  const [kvSizeUserModified, setKvSizeUserModified] = useState<boolean>(false);
  const [modelSizeUserModified, setModelSizeUserModified] = useState<boolean>(false);

  const [customKvQuantType, setCustomKvQuantType] = useState<'fp16' | 'fp8' | 'int8' | 'int4'>('fp8');

  /* ─── Auto estimation effects ─────────────────────────────── */
  // Update KV size when hyper-params change and user didn't override
  useEffect(() => {
    if (useCustomModel && !kvSizeUserModified) {
      const estKv = computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim);
      setCustomPerKVsizeFp8(estKv);
    }
  }, [useCustomModel, kvSizeUserModified, customLayers, customNumKVHeads, customHeadDim]);

  // Update model size when total params or quant changes and user didn't override
  useEffect(() => {
    if (useCustomModel && !modelSizeUserModified) {
      const estSize = estimateModelSizeGB(customTotalParamsB, quantType);
      setCustomModelSizeGB(estSize);
    }
  }, [useCustomModel, modelSizeUserModified, customTotalParamsB, quantType]);

  // Sync kvQuantType with selected card when not using custom GPU
  useEffect(() => {
    if (!useCustomGPU && selectedCard?.kvQuantType) {
      setKvQuantType(selectedCard.kvQuantType as any);
    }
  }, [selectedCard, useCustomGPU]);

  // Sync kvQuantType when custom GPU kv setting changes
  useEffect(() => {
    if (useCustomGPU) {
      setKvQuantType(customKvQuantType);
    }
  }, [useCustomGPU, customKvQuantType]);

  // Calculation effect
  useEffect(() => {
    const effectiveModel: ModelDef | null = useCustomModel ? {
      name: 'Custom Model',
      modelSizeGB: customModelSizeGB,
      totalParamsB: customTotalParamsB,
      activeParamsB: customActiveParamsB,
      perKVsizeFp8: customPerKVsizeFp8,
      quantType: quantType,
      quantBits: quantType === 'int4' ? 4 : quantType === 'fp8' ? 8 : quantType === 'int8' ? 8 : 16,
      hiddenSize: 4096,
      paramsB: customTotalParamsB,
      layers: customLayers,
      numKVHeads: customNumKVHeads,
      headDim: customHeadDim
    } as ModelDef : selectedModel;

    const effectiveCard: GPUCard | null = useCustomGPU ? {
      name: 'Custom GPU',
      vramGb: customVramGB,
      memoryBandwidthGBs: customMemoryBandwidthGBs,
      processPower: { fp16: customProcessPowerFP16 },
      kvQuantType: customKvQuantType,
      architecture: 'Custom'
    } as GPUCard : selectedCard;

    if (!effectiveModel || !effectiveCard) {
      setResults(null);
      return;
    }

    try {
      const modelVram = computeModelVramGB(effectiveModel, quantType);
      const kvCacheVram = computeKvCacheVramGB(effectiveModel.hiddenSize, maxLength, kvQuantType, effectiveModel, effectiveCard);
      const totalCardVram = effectiveCard.vramGb * parallelGPUs;
      const usableVram = Math.max(0, totalCardVram * vramUtilProportion - minReserveVramGB);
      const totalVramReq = modelVram + kvCacheVram;

      if (totalVramReq > usableVram) {
        setResults({
          error: `Insufficient VRAM: Need ${totalVramReq.toFixed(2)}GB but only ${usableVram.toFixed(2)}GB available`,
          totalVram: totalVramReq,
          usableVram: usableVram,
          usableKvCacheVram: 0,
          reservedVram: minReserveVramGB,
          modelVram: modelVram,
          kvCacheVram: kvCacheVram,
          genSpeed: 0,
          promptSpeed: 0,
          sharedGen: 0,
          sharedPrompt: 0,
          ppScaling: 1,
          membwScaling: 1,
          maxTokenCountSimultaneous: 0,
          fullLengthGenCount: 0
        });
        return;
      }

      const ppScaling = Math.pow(parallelGPUs, 0.6);
      const membwScaling = Math.pow(parallelGPUs, 0.8);
      const processPowerFP16 = (effectiveCard.processPower.fp16 || 0) * ppScaling;
      const memoryBandwidth = effectiveCard.memoryBandwidthGBs * membwScaling;
      const activeParams = effectiveModel.activeParamsB;
      const totalParams = effectiveModel.totalParamsB;
      const quantBytes = quantType === 'fp16' ? 2 : quantType === 'fp8' ? 1 : quantType === 'int8' ? 1 : 0.5;
      const promptSpeed = (processPowerFP16 * 1000) / (totalParams * Math.sqrt(2));
      const genSpeed = memoryBandwidth / (activeParams * quantBytes);
      const sharedPrompt = promptSpeed / userCount;
      const sharedGen = genSpeed / userCount;
      const usableKvCacheVram = Math.max(0, usableVram - modelVram);
      const fullLengthGenCount = usableKvCacheVram / kvCacheVram;
      const maxTokenCountSimultaneous = maxLength * fullLengthGenCount;

      setResults({
        totalVram: totalVramReq,
        usableVram: usableVram,
        usableKvCacheVram: usableKvCacheVram,
        reservedVram: minReserveVramGB,
        modelVram: modelVram,
        kvCacheVram: kvCacheVram,
        genSpeed: genSpeed,
        promptSpeed: promptSpeed,
        sharedGen: sharedGen,
        sharedPrompt: sharedPrompt,
        ppScaling: ppScaling,
        membwScaling: membwScaling,
        maxTokenCountSimultaneous: maxTokenCountSimultaneous,
        fullLengthGenCount: fullLengthGenCount,
        error: null
      });
    } catch (error) {
      console.error('Calculation error:', error);
      setResults({
        error: 'Calculation error occurred',
        totalVram: 0,
        usableVram: 0,
        usableKvCacheVram: 0,
        reservedVram: 0,
        modelVram: 0,
        kvCacheVram: 0,
        genSpeed: 0,
        promptSpeed: 0,
        sharedGen: 0,
        sharedPrompt: 0,
        ppScaling: 1,
        membwScaling: 1,
        maxTokenCountSimultaneous: 0,
        fullLengthGenCount: 0
      });
    }
  }, [selectedModel, selectedCard, quantType, kvQuantType, maxLength, userCount, vramUtilProportion, minReserveVramGB, parallelGPUs, useCustomGPU, customVramGB, customMemoryBandwidthGBs, customProcessPowerFP16, useCustomModel, customTotalParamsB, customModelSizeGB, customActiveParamsB, customPerKVsizeFp8, customLayers, customNumKVHeads, customHeadDim, customKvQuantType]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Left Side - Configuration */}
      <div className="space-y-6">
                {/* Model Configuration */}
                <div className="bg-white rounded-xl border border-gray-200 p-6">
          {/* Header with inline custom-Model toggle on the right */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Model Configuration</h3>
                <p className="text-sm text-gray-500">Select model and quantization settings</p>
              </div>
            </div>
            <label className="toggle-clean">
              <input type="checkbox" checked={useCustomModel} onChange={e=>setUseCustomModel(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>
          
          <div className="space-y-4">
            {useCustomModel && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="form-label">Total Params (B)</label>
                  <input type="number" value={customTotalParamsB} onChange={e=>setCustomTotalParamsB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label flex items-center gap-1">Model Size (GB)
                    <button type="button" onClick={()=>{ const est = estimateModelSizeGB(customTotalParamsB, quantType); setCustomModelSizeGB(est); setModelSizeUserModified(false); }} className="btn btn-xs btn-ghost">↻</button>
                  </label>
                  <input type="number" value={customModelSizeGB} onChange={e=>{setCustomModelSizeGB(+e.target.value); setModelSizeUserModified(true);}} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div className="col-span-2">
                  <label className="form-label">Active Params (B)</label>
                  <input type="number" value={customActiveParamsB} onChange={e=>setCustomActiveParamsB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label">Layers</label>
                  <input type="number" value={customLayers} onChange={e=>setCustomLayers(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label">KV Heads</label>
                  <input type="number" value={customNumKVHeads} onChange={e=>setCustomNumKVHeads(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label">Head Dim</label>
                  <input type="number" value={customHeadDim} onChange={e=>setCustomHeadDim(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label flex items-center gap-1">KV Size (KB/Token)
                    <button type="button" onClick={()=>{const est = computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim); setCustomPerKVsizeFp8(est); setKvSizeUserModified(false);}} className="btn btn-xs btn-ghost">↻</button>
                  </label>
                  <input type="number" value={(customPerKVsizeFp8/1024).toFixed(0)} onChange={e=>{setCustomPerKVsizeFp8(+e.target.value*1024); setKvSizeUserModified(true);}} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
              </div>
            )}

            {!useCustomModel && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Model</label>
              <div className="dropdown-clean" tabIndex={0}>
                <button
                  type="button"
                  className="dropdown-trigger"
                  onClick={() => setDropdownOpenModel(!dropdownOpenModel)}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: modelColors[selectedModel?.name || ''] || '#6b7280' }}
                    ></div>
                    <span className="flex-1 truncate">{selectedModel ? selectedModel.name : 'Select a model'}</span>
                  </div>
                  <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpenModel ? 'rotate-180' : ''}`} />
                </button>
                {dropdownOpenModel && (
                  <div className="dropdown-menu max-h-60 overflow-y-auto" onBlur={() => setDropdownOpenModel(false)}>
                    {modelDefs.map((model) => (
                      <div
                        key={model.name}
                        className="dropdown-item"
                        onClick={() => {
                          setSelectedModel(model);
                          setDropdownOpenModel(false);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: modelColors[model.name] || '#6b7280' }}
                          ></div>
                          <div>
                            <div className="text-sm font-medium">{model.name}</div>
                            <div className="text-xs text-gray-500">
                              {model.totalParamsB}B • {model.modelSizeGB.toFixed(1)}GB
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Model Quantization</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'int4', label: 'INT4', desc: 'AWQ/GPTQ' },
                  { value: 'fp8', label: 'FP8', desc: 'High perf' },
                  { value: 'fp16', label: 'FP16', desc: 'Full prec' }
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => setQuantType(option.value as any)}
                    className={`p-3 rounded-lg border-2 text-center transition-all ${
                      quantType === option.value
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-gray-200 hover:border-gray-300 text-gray-600'
                    }`}
                  >
                    <div className="font-medium text-sm">{option.label}</div>
                    <div className="text-xs opacity-75">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* GPU Configuration */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          {/* Header with inline custom-GPU toggle on the right */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                <Cpu className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">GPU Configuration</h3>
                <p className="text-sm text-gray-500">Select GPU model or set custom specifications</p>
              </div>
            </div>
            {/* Small switch – no label to save vertical space */}
            <label className="toggle-clean">
              <input type="checkbox" checked={useCustomGPU} onChange={e=>setUseCustomGPU(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>
          
          <div className="space-y-4">
            {useCustomGPU && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="form-label">VRAM (GB)</label>
                  <input type="number" value={customVramGB} onChange={e=>setCustomVramGB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label">Mem BW (GB/s)</label>
                  <input type="number" value={customMemoryBandwidthGBs} onChange={e=>setCustomMemoryBandwidthGBs(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div className="col-span-2">
                  <label className="form-label">FP16 TFLOPS</label>
                  <input type="number" value={customProcessPowerFP16} onChange={e=>setCustomProcessPowerFP16(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div className="col-span-2">
                  <label className="form-label">KV Cache Quant</label>
                  <select value={customKvQuantType} onChange={e=>setCustomKvQuantType(e.target.value as any)} className="select select-bordered w-full">
                    <option value="fp8">FP8</option>
                    <option value="fp16">FP16</option>
                    <option value="int8">INT8</option>
                    <option value="int4">INT4</option>
                  </select>
                </div>
              </div>
            )}

            {!useCustomGPU && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Select GPU Model</label>
              <div className="dropdown-clean" tabIndex={0}>
                <button
                  type="button"
                  className="dropdown-trigger"
                  onClick={() => setDropdownOpenGPU(!dropdownOpenGPU)}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getGpuColor(selectedCard?.name || '') }}
                    ></div>
                    {selectedCard ? (
                      <div className="flex flex-col text-left">
                        <span className="text-sm font-medium leading-none">{stripVendor(selectedCard.name)}</span>
                        <span className="text-xs text-gray-500">{inferArchitecture(selectedCard)}{selectedCard.cores ? ` • ${selectedCard.cores.toLocaleString()} cores` : ''}</span>
                      </div>
                    ) : <span className="flex-1 truncate">Select a GPU</span>}
                  </div>
                  <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpenGPU ? 'rotate-180' : ''}`} />
                </button>

                {dropdownOpenGPU && (
                  <div className="dropdown-menu max-h-60 overflow-y-auto" onBlur={() => setDropdownOpenGPU(false)}>
                    {gpuCards.map((gpu) => (
                      <div
                        key={gpu.name}
                        className="dropdown-item"
                        onClick={() => {
                          setSelectedCard(gpu);
                          setKvQuantType((gpu.kvQuantType as any) || 'fp16');
                          setDropdownOpenGPU(false);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getGpuColor(gpu.name) }}
                          ></div>
                          <div>
                            <div className="text-sm font-medium">{stripVendor(gpu.name)}</div>
                            <div className="text-xs text-gray-500">
                              {inferArchitecture(gpu)}{gpu.cores ? ` • ${gpu.cores.toLocaleString()} cores` : ''}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            )}
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
              <Settings className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Deployment Parameters</h3>
              <p className="text-sm text-gray-500">Real-world deployment tweaks</p>
            </div>
          </div>
          
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Users className="w-4 h-4 inline mr-1" />
                  Concurrent Users
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={1}
                    max={100}
                    value={userCount}
                    onChange={e => setUserCount(+e.target.value)}
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={userCount}
                    onChange={e=>setUserCount(Math.min(100, Math.max(1, +e.target.value)))}
                    className="input input-bordered w-20 text-center"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Zap className="w-4 h-4 inline mr-1" />
                  Parallel GPUs
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={1}
                    max={8}
                    value={parallelGPUs}
                    onChange={e => setParallelGPUs(+e.target.value)}
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <input
                    type="number"
                    min={1}
                    max={8}
                    value={parallelGPUs}
                    onChange={e=>setParallelGPUs(Math.min(8, Math.max(1, +e.target.value)))}
                    className="input input-bordered w-20 text-center"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <HardDrive className="w-4 h-4 inline mr-1" />
                Max Context Length: {maxLength.toLocaleString()} tokens
              </label>
              <div className="relative">
                <input
                  type="range"
                  min={8192}
                  max={131072}
                  step={8192}
                  value={maxLength}
                  onChange={e => setMaxLength(+e.target.value)}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>8K</span>
                  <span>16K</span>
                  <span>32K</span>
                  <span>64K</span>
                  <span>128K</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  VRAM Utilization: {Math.round(vramUtilProportion * 100)}%
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={10}
                    max={100}
                    value={vramUtilProportion * 100}
                    onChange={e => setVramUtilProportion(+e.target.value / 100)}
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <input
                    type="number"
                    min={10}
                    max={100}
                    value={Math.round(vramUtilProportion*100)}
                    onChange={e=>setVramUtilProportion(Math.min(100,Math.max(10,+e.target.value))/100)}
                    className="input input-bordered w-20 text-center"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Reserve VRAM: {minReserveVramGB}GB
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={8}
                    step={0.5}
                    value={minReserveVramGB}
                    onChange={e => setMinReserveVramGB(+e.target.value)}
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <input
                    type="number"
                    min={0}
                    max={8}
                    step={0.5}
                    value={minReserveVramGB}
                    onChange={e=>setMinReserveVramGB(Math.min(8, Math.max(0, +e.target.value)))}
                    className="input input-bordered w-20 text-center"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Results */}
      <div className="space-y-6">
        {/* Error Display */}
        {results?.error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-red-500 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-red-900">Configuration Issue</h3>
                <p className="text-sm text-red-700">{results.error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Performance Results */}
        {results && !results.error ? (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-indigo-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Performance Results</h3>
                <p className="text-sm text-gray-500">Computational power analysis</p>
              </div>
            </div>
            
            {/* Main Metrics */}
            <div className="grid grid-cols-1 gap-6 mb-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-indigo-600 mb-2">
                  {results.totalVram.toFixed(1)} GB
                </div>
                <div className="text-gray-600 mb-3">Total VRAM Required</div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-indigo-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, (results.totalVram / (((useCustomGPU ? customVramGB : (selectedCard?.vramGb || 24))) * parallelGPUs)) * 100)}%` }}
                  />
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  Using {((results.totalVram / (((useCustomGPU ? customVramGB : (selectedCard?.vramGb || 24))) * parallelGPUs)) * 100).toFixed(1)}% of GPU memory
                </div>
              </div>
            </div>

            {/* Detailed Breakdown */}
            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-600 mb-1">
                  {results.modelVram.toFixed(1)} GB
                </div>
                <div className="text-sm text-blue-700">Model Weights</div>
                <div className="text-xs text-blue-500 mt-1">
                  {quantType === 'int4' ? '4-bit' : quantType === 'fp8' ? '8-bit' : '16-bit'} quantization
                </div>
              </div>
              
              <div className="bg-green-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600 mb-1">
                  {results.kvCacheVram.toFixed(1)} GB
                </div>
                <div className="text-sm text-green-700">KV Cache</div>
                <div className="text-xs text-green-500 mt-1">
                  {maxLength.toLocaleString()} tokens context
                </div>
                {/* KV quantization bits info */}
                <div className="text-xs text-green-500">
                  Quant: {
                    useCustomGPU ? (
                      customKvQuantType === 'fp8' ? '8-bit' :
                      customKvQuantType === 'int8' ? '8-bit' :
                      customKvQuantType === 'int4' ? '4-bit' :
                      '16-bit'
                    ) : (
                      (selectedCard?.kvQuantType === 'fp8' || selectedCard?.kvQuantType === 'int8') ? '8-bit' :
                      selectedCard?.kvQuantType === 'int4' ? '4-bit' :
                      '16-bit'
                    )
                  } ({kvQuantType.toUpperCase()})
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="space-y-4">
              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-3">Throughput Performance</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.genSpeed.toFixed(0)} tok/s</div>
                    <div className="text-sm text-gray-600">Generation Speed</div>
                  </div>
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.promptSpeed.toFixed(0)} tok/s</div>
                    <div className="text-sm text-gray-600">Prompt Processing</div>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-3">Shared Performance ({userCount} users)</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.sharedGen.toFixed(1)} tok/s</div>
                    <div className="text-sm text-gray-600">Per User Generation</div>
                  </div>
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.sharedPrompt.toFixed(0)} tok/s</div>
                    <div className="text-sm text-gray-600">Per User Prompt</div>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-3">Capacity</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.fullLengthGenCount.toFixed(2)}</div>
                    <div className="text-sm text-gray-600">Full-length Sequences</div>
                  </div>
                  <div>
                    <div className="text-xl font-bold text-gray-900">{results.maxTokenCountSimultaneous.toFixed(0)}</div>
                    <div className="text-sm text-gray-600">Total Tokens Capacity</div>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-3">Hardware Info</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">GPU Model</span>
                    <span className="text-sm font-medium text-gray-900">{useCustomGPU ? 'Custom GPU' : stripVendor(selectedCard?.name || '')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Memory Bandwidth</span>
                    <span className="text-sm font-medium text-gray-900">{useCustomGPU ? customMemoryBandwidthGBs : selectedCard?.memoryBandwidthGBs} GB/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">FP16 Performance</span>
                    <span className="text-sm font-medium text-gray-900">{useCustomGPU ? customProcessPowerFP16.toFixed(1) : (selectedCard?.processPower?.fp16 || 0).toFixed(1)} TFLOPS</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : !results?.error && (
          <div className="bg-gray-50 rounded-xl border-2 border-dashed border-gray-300 p-12 text-center">
            <BarChart3 className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-600 mb-2">Ready to Calculate</h3>
            <p className="text-gray-500">Select a GPU and model to see performance results</p>
          </div>
        )}
      </div>
    </div>
  );
}
