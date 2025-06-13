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
      // pick the larger of the %-based reserve and the fixed minimum
      const proportionalReserve = totalCardVram * (1 - vramUtilProportion);
      const effectiveReserve   = Math.max(proportionalReserve, minReserveVramGB);
      const usableVram         = Math.max(0, totalCardVram - effectiveReserve);
      const totalVramReq = modelVram + kvCacheVram;

      if (totalVramReq > usableVram) {
        setResults({
          error: `💥 💥 Insufficient VRAM! Need! Need ${totalVramReq.toFixed(2)}GB but only ${usableVram.toFixed(2)}GB available! 😱`,
          totalVram: totalVramReq,
          usableVram: usableVram,
          usableKvCacheVram: 0,
          reservedVram: effectiveReserve,
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
        reservedVram: effectiveReserve,
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
        error: '💥 Calculation error occurred! Something went wrong! 🚨',
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
        {/* 🤖 AI Model Beast Config 🔥 */}
        <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl border-2 border-green-300 p-6 shadow-xl">
          {/* Header with inline custom-Model toggle on the right */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-2xl">🤖</span>
              </div>
              <div>
                <h3 className="font-bold text-green-900 text-lg">🤖 AI Model Beast Config 🔥</h3>
                <p className="text-sm text-green-700">⚡ Pick your AI champion & tune it like a boss! 💪</p>
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
                  <label className="form-label text-green-800 font-medium">🧠 Total Params (B)</label>
                  <input type="number" value={customTotalParamsB} onChange={e=>setCustomTotalParamsB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label flex items-center gap-1 text-green-800 font-medium">📊 Model Size (GB)
                    <button type="button" onClick={()=>{ const est = estimateModelSizeGB(customTotalParamsB, quantType); setCustomModelSizeGB(est); setModelSizeUserModified(false); }} className="btn btn-xs btn-ghost">🔄</button>
                  </label>
                  <input type="number" value={customModelSizeGB} onChange={e=>{setCustomModelSizeGB(+e.target.value); setModelSizeUserModified(true);}} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div className="col-span-2">
                  <label className="form-label text-green-800 font-medium">⚡ Active Params (B)</label>
                  <input type="number" value={customActiveParamsB} onChange={e=>setCustomActiveParamsB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-green-800 font-medium">🏗️ Layers</label>
                  <input type="number" value={customLayers} onChange={e=>setCustomLayers(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-green-800 font-medium">🎯 KV Heads</label>
                  <input type="number" value={customNumKVHeads} onChange={e=>setCustomNumKVHeads(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-green-800 font-medium">🧱 Head Dim</label>
                  <input type="number" value={customHeadDim} onChange={e=>setCustomHeadDim(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label flex items-center gap-1 text-green-800 font-medium">💾 KV Size (KB/Token)
                    <button type="button" onClick={()=>{const est = computeKvSizeFp8FromParams(customLayers, customNumKVHeads, customHeadDim); setCustomPerKVsizeFp8(est); setKvSizeUserModified(false);}} className="btn btn-xs btn-ghost">🔄</button>
                  </label>
                  <input type="number" value={(customPerKVsizeFp8/1024).toFixed(0)} onChange={e=>{setCustomPerKVsizeFp8(+e.target.value*1024); setKvSizeUserModified(true);}} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
              </div>
            )}

            {!useCustomModel && (
            <div>
              <label className="block text-sm font-bold text-green-800 mb-2">🎯 Select Your AI Champion</label>
              <div className="dropdown-clean" tabIndex={0}>
                <button
                  type="button"
                  className="dropdown-trigger bg-white hover:bg-green-50 border-green-300 w-full p-3 rounded-lg border-2 flex items-center justify-between"
                  onClick={() => setDropdownOpenModel(!dropdownOpenModel)}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: modelColors[selectedModel?.name || ''] || '#6b7280' }}
                    ></div>
                    <span className="flex-1 truncate">{selectedModel ? `🤖 ${selectedModel.name}` : '🤖 Pick your AI beast'}</span>
                  </div>
                  <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpenModel ? 'rotate-180' : ''}`} />
                </button>
                {dropdownOpenModel && (
                  <div className="absolute z-10 w-full mt-2 bg-white border-2 border-green-300 rounded-lg shadow-xl max-h-60 overflow-y-auto">
                    {modelDefs.map((model) => (
                      <div
                        key={model.name}
                        className="p-3 hover:bg-green-100 cursor-pointer border-b border-green-100 last:border-b-0"
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
                            <div className="text-sm font-medium">🤖 {model.name}</div>
                            <div className="text-xs text-gray-500">
                              🧠 {model.totalParamsB}B • 📊 {model.modelSizeGB.toFixed(1)}GB
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
              <label className="block text-sm font-bold text-green-800 mb-2">🔧 🔧 Model Quantization Magic Magic</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'int4', label: 'INT4', desc: '🚀 Speed Demon', emoji: '🚀' },
                  { value: 'fp8', label: 'FP8', desc: '⚡ High Perf', emoji: '⚡' },
                  { value: 'fp16', label: 'FP16', desc: '💎 Full Prec', emoji: '💎' }
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => setQuantType(option.value as any)}
                    className={`p-3 rounded-xl border-2 text-center transition-all hover:scale-105 ${
                      quantType === option.value
                        ? 'border-green-500 bg-green-100 text-green-800 shadow-lg'
                        : 'border-green-200 hover:border-green-400 text-green-700 bg-white'
                    }`}
                  >
                    <div className="text-lg mb-1">{option.emoji}</div>
                    <div className="font-bold text-sm">{option.label}</div>
                    <div className="text-xs opacity-75">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 🎮 GPU Beast Selection 💪 */}
        <div className="bg-gradient-to-br from-blue-50 to-cyan-100 rounded-2xl border-2 border-blue-300 p-6 shadow-xl">
          {/* Header with inline custom-GPU toggle on the right */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-2xl">🎮</span>
              </div>
              <div>
                <h3 className="font-bold text-blue-900 text-lg">🎮 GPU Beast Selection 💪</h3>
                <p className="text-sm text-blue-700">🔥 Choose your graphics powerhouse! ⚡💎</p>
              </div>
            </div>
            <label className="toggle-clean">
              <input type="checkbox" checked={useCustomGPU} onChange={e=>setUseCustomGPU(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>
          
          <div className="space-y-4">
            {useCustomGPU && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="form-label text-blue-800 font-medium">💾 VRAM (GB)</label>
                  <input type="number" value={customVramGB} onChange={e=>setCustomVramGB(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-blue-800 font-medium">🚄 🚄 Memory Bandwidth (GB/s)</label>
                  <input type="number" value={customMemoryBandwidthGBs} onChange={e=>setCustomMemoryBandwidthGBs(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-blue-800 font-medium">⚡ FP16 Power (TFLOPS)</label>
                  <input type="number" value={customProcessPowerFP16} onChange={e=>setCustomProcessPowerFP16(+e.target.value)} className="input input-bordered w-full px-2 py-2 border rounded-md" />
                </div>
                <div>
                  <label className="form-label text-blue-800 font-medium">🎯 🎯 KV Cache Quant</label>
                  <select value={customKvQuantType} onChange={e=>setCustomKvQuantType(e.target.value as any)} className="input input-bordered w-full px-2 py-2 border rounded-md">
                    <option value="fp16">💎 FP16</option>
                    <option value="fp8">⚡ FP8</option>
                    <option value="int8">🔧 INT8</option>
                    <option value="int4">🚀 INT4</option>
                  </select>
                </div>
              </div>
            )}

            {!useCustomGPU && (
            <div>
              <label className="block text-sm font-bold text-blue-800 mb-2">🎯 Select Your GPU Beast</label>
              <div className="dropdown-clean" tabIndex={0}>
                <button
                  type="button"
                  className="dropdown-trigger bg-white hover:bg-blue-50 border-blue-300 w-full p-3 rounded-lg border-2 flex items-center justify-between"
                  onClick={() => setDropdownOpenGPU(!dropdownOpenGPU)}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getGpuColor(selectedCard?.name || '') }}
                    ></div>
                    <span className="flex-1 truncate">{selectedCard ? `🎮 ${stripVendor(selectedCard.name)}` : '🎮 Pick your GPU beast'}</span>
                  </div>
                  <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpenGPU ? 'rotate-180' : ''}`} />
                </button>
                {dropdownOpenGPU && (
                  <div className="absolute z-10 w-full mt-2 bg-white border-2 border-blue-300 rounded-lg shadow-xl max-h-60 overflow-y-auto">
                    {gpuCards.map((card) => (
                      <div
                        key={card.name}
                        className="p-3 hover:bg-blue-100 cursor-pointer border-b border-blue-100 last:border-b-0"
                        onClick={() => {
                          setSelectedCard(card);
                          setDropdownOpenGPU(false);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getGpuColor(card.name) }}
                          ></div>
                          <div>
                            <div className="text-sm font-medium">🎮 {stripVendor(card.name)}</div>
                            <div className="text-xs text-gray-500">
                              💾 {card.vramGb}GB • 🏗️ {inferArchitecture(card)}
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

        {/* ⚙️ Advanced Beast Settings 🔧 */}
        <div className="bg-gradient-to-br from-purple-50 to-pink-100 rounded-2xl border-2 border-purple-300 p-6 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-2xl">⚙️</span>
            </div>
            <div>
              <h3 className="font-bold text-purple-900 text-lg">⚙️ Advanced Beast Settings 🔧</h3>
              <p className="text-sm text-purple-700">🔥 Fine-tune your AI monster! 💪⚡</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-purple-800 mb-2">
                  📏 📏 Max Context: {maxLength.toLocaleString()} tokens
                </label>
                <input
                  type="range"
                  min={512}
                  max={131072}
                  step={512}
                  value={maxLength}
                  onChange={e => setMaxLength(+e.target.value)}
                  className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-bold text-purple-800 mb-2">
                  👥 👥 Users: {userCount} legends
                </label>
                <input
                  type="range"
                  min={1}
                  max={100}
                  value={userCount}
                  onChange={e => setUserCount(+e.target.value)}
                  className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-sm font-bold text-purple-800 mb-2">
                  🔥 🔥 Parallel GPUs: {parallelGPUs} beasts
                </label>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={parallelGPUs}
                  onChange={e => setParallelGPUs(+e.target.value)}
                  className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-purple-800 mb-2">
                  💪 💪 VRAM Usage: {Math.round(vramUtilProportion*100)}% beast mode
                </label>
                <input
                  type="range"
                  min={10}
                  max={100}
                  value={Math.round(vramUtilProportion*100)}
                  onChange={e => setVramUtilProportion(+e.target.value/100)}
                  className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-bold text-purple-800 mb-2">
                  🛡️ 🛡️ Reserve VRAM: {minReserveVramGB}GB safety net
                </label>
                <input
                  type="range"
                  min={0}
                  max={12}
                  step={0.5}
                  value={minReserveVramGB}
                  onChange={e => setMinReserveVramGB(+e.target.value)}
                  className="w-full h-3 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Results */}
      <div className="space-y-6">
        {/* Error Display */}
        {results?.error && (
          <div className="bg-gradient-to-br from-red-50 to-pink-100 border-2 border-red-300 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-red-400 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-2xl">💥</span>
              </div>
              <div>
                <h3 className="font-bold text-red-900 text-lg">💥 Houston, We Have a Problem! 🚨</h3>
                <p className="text-sm text-red-700">⚠️ {results.error}</p>
              </div>
            </div>
          </div>
        )}

        {/* 📊 Beast Mode Results! 🔥 */}
        {results && !results.error ? (
          <div className="bg-gradient-to-br from-indigo-50 to-blue-100 rounded-2xl border-2 border-indigo-300 p-6 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-400 to-blue-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-2xl">📊</span>
              </div>
              <div>
                <h3 className="font-bold text-indigo-900 text-lg">📊 Beast Mode Results! 🔥</h3>
                <p className="text-sm text-indigo-700">💪 Your AI powerhouse analysis! ⚡💎</p>
              </div>
            </div>
            
            {/* Main Metrics */}
            <div className="grid grid-cols-1 gap-6 mb-8">
              <div className="text-center bg-gradient-to-br from-yellow-50 to-orange-100 rounded-xl p-6 border-2 border-yellow-300">
                <div className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
                  🚀 {results.totalVram.toFixed(1)} GB
                </div>
                <div className="text-indigo-800 mb-3 font-bold text-lg">💾 Total VRAM Beast Requirement</div>
                <div className="w-full bg-gray-200 rounded-full h-4 shadow-inner">
                  <div 
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-4 rounded-full transition-all duration-500 shadow-lg"
                    style={{ width: `${Math.min(100, (results.totalVram / (((useCustomGPU ? customVramGB : (selectedCard?.vramGb || 24))) * parallelGPUs)) * 100)}%` }}
                  />
                </div>
                <div className="text-sm text-indigo-600 mt-3 font-medium">
                  🔥 Using {((results.totalVram / (((useCustomGPU ? customVramGB : (selectedCard?.vramGb || 24))) * parallelGPUs)) * 100).toFixed(1)}% GPU memory with 1 full request<br/>
                  💪 {results.fullLengthGenCount.toFixed(2)} x full-length requests MAX!<br/>
                </div>
              </div>
            </div>

            {/* Detailed Breakdown */}
            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-cyan-100 rounded-xl p-4 border-2 border-blue-300">
                <div className="text-3xl font-bold text-blue-600 mb-1 flex items-center gap-2">
                  🧠 {results.modelVram.toFixed(1)} GB
                </div>
                <div className="text-sm text-blue-700 font-bold">🤖 🤖 Model Weights</div>
                <div className="text-xs text-blue-500 mt-1 font-medium">
                  ⚡ {quantType === 'int4' ? '4-bit 🚀' : quantType === 'fp8' ? '8-bit ⚡' : '16-bit 💎'} quantization
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl p-4 border-2 border-green-300">
                <div className="text-3xl font-bold text-green-600 mb-1 flex items-center gap-2">
                  💾 {results.kvCacheVram.toFixed(1)} GB
                </div>
                <div className="text-sm text-green-700 font-bold">🎯 🎯 KV Cache</div>
                <div className="text-xs text-green-500 mt-1 font-medium">
                  📏 {maxLength.toLocaleString()} tokens context
                </div>
                {/* KV quantization bits info */}
                <div className="text-xs text-green-500 font-medium">
                  🔧 Quant: {
                    useCustomGPU ? (
                      customKvQuantType === 'fp8' ? '8-bit ⚡' :
                      customKvQuantType === 'int8' ? '8-bit 🔧' :
                      customKvQuantType === 'int4' ? '4-bit 🚀' :
                      '16-bit 💎'
                    ) : (
                      (selectedCard?.kvQuantType === 'fp8' || selectedCard?.kvQuantType === 'int8') ? '8-bit ⚡' :
                      selectedCard?.kvQuantType === 'int4' ? '4-bit 🚀' :
                      '16-bit 💎'
                    )
                  } ({kvQuantType.toUpperCase()})
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="space-y-4">
              <div className="border-t-2 border-indigo-200 pt-4">
                <h4 className="font-bold text-indigo-900 mb-3 text-lg">🚄 Throughput Beast Performance</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-yellow-50 to-orange-100 rounded-lg p-3 border border-yellow-300">
                    <div className="text-2xl font-bold text-orange-600">⚡ {results.genSpeed.toFixed(0)} tok/s</div>
                    <div className="text-sm text-orange-700 font-medium">🔥 🔥 Generation Speed</div>
                  </div>
                  <div className="bg-gradient-to-br from-purple-50 to-pink-100 rounded-lg p-3 border border-purple-300">
                    <div className="text-2xl font-bold text-purple-600">🚀 {results.promptSpeed.toFixed(0)} tok/s</div>
                    <div className="text-sm text-purple-700 font-medium">💭 💭 Prompt Processing</div>
                  </div>
                </div>
              </div>

              <div className="border-t-2 border-indigo-200 pt-4">
                <h4 className="font-bold text-indigo-900 mb-3 text-lg">👥 Shared Beast Power ({userCount} legends)</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-green-50 to-teal-100 rounded-lg p-3 border border-green-300">
                    <div className="text-2xl font-bold text-green-600">💪 {results.sharedGen.toFixed(1)} tok/s</div>
                    <div className="text-sm text-green-700 font-medium">🎯 🎯 Per User Generation</div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg p-3 border border-blue-300">
                    <div className="text-2xl font-bold text-blue-600">⚡ {results.sharedPrompt.toFixed(0)} tok/s</div>
                    <div className="text-sm text-blue-700 font-medium">💭 💭 Per User Prompt</div>
                  </div>
                </div>
              </div>

              <div className="border-t-2 border-indigo-200 pt-4">
                <h4 className="font-bold text-indigo-900 mb-3 text-lg">📊 Beast 📊 Beast Capacity</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-red-50 to-pink-100 rounded-lg p-3 border border-red-300">
                    <div className="text-2xl font-bold text-red-600">🔥 {results.fullLengthGenCount.toFixed(2)}</div>
                    <div className="text-sm text-red-700 font-medium">📏 Full-length Sequences</div>
                  </div>
                  <div className="bg-gradient-to-br from-cyan-50 to-blue-100 rounded-lg p-3 border border-cyan-300">
                    <div className="text-2xl font-bold text-cyan-600">💎 {results.maxTokenCountSimultaneous.toFixed(0)}</div>
                    <div className="text-sm text-cyan-700 font-medium">🚀 Total Tokens 📊 Beast Capacity</div>
                  </div>
                </div>
              </div>

              <div className="border-t-2 border-indigo-200 pt-4">
                <h4 className="font-bold text-indigo-900 mb-3 text-lg">🎮 Hardware Beast Info</h4>
                <div className="space-y-2 bg-gradient-to-br from-gray-50 to-slate-100 rounded-lg p-4 border border-gray-300">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 font-medium">🎮 🎮 GPU Model</span>
                    <span className="text-sm font-bold text-gray-900">{useCustomGPU ? '🔧 Custom GPU Beast' : `🎮 ${stripVendor(selectedCard?.name || '')}`}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 font-medium">🚄 🚄 Memory Bandwidth</span>
                    <span className="text-sm font-bold text-gray-900">⚡ {useCustomGPU ? customMemoryBandwidthGBs : selectedCard?.memoryBandwidthGBs} GB/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 font-medium">💪 💪 FP16 Performance</span>
                    <span className="text-sm font-bold text-gray-900">🔥 {useCustomGPU ? customProcessPowerFP16.toFixed(1) : (selectedCard?.processPower?.fp16 || 0).toFixed(1)} TFLOPS</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : !results?.error && (
          <div className="bg-gradient-to-br from-gray-50 to-slate-100 rounded-2xl border-2 border-dashed border-gray-400 p-12 text-center shadow-xl">
            <div className="text-6xl mb-4">🚀</div>
            <h3 className="text-xl font-bold text-gray-700 mb-2">🔥 🔥 Ready to Calculate Beast Mode! Beast Mode! 💪</h3>
            <p className="text-gray-600 font-medium">🎯 Pick your GPU and AI model to see epic results! ⚡🤖💎</p>
          </div>
        )}
      </div>
    </div>
  );
}
