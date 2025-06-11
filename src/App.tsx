import './App.css'
import LLMVRAMCalculator from './components/LlmGpuVramCalculator'

function App() {
  return (
    <>
      <div className="llm-vram-calculator-app">
        <h1>LLM VRAM Calculator</h1>
        <LLMVRAMCalculator />
      </div>
    </>
  )
}

export default App
