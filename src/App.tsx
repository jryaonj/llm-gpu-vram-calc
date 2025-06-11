import './App.css'
import LLMVRAMCalculator from './components/LlmGpuVramCalculator'

function App() {
  return (
    <>
      <div className="llm-vram-calculator-app">
        {/* <h1>LLM VRAM Calculator</h1> */}
        <LLMVRAMCalculator />
      </div>

      <div className="border-t border-gray-200 mt-8 pt-4 text-center text-sm text-gray-500">
        © 2025 jryaonj · Powered by{' '}
        <a 
          href="https://react.dev" 
          target="_blank" 
          rel="noopener noreferrer"
          className="hover:text-gray-700"
        >
          React
        </a>
        {' '}& {' '}
        <a
          href="https://daisyui.com"
          target="_blank" 
          rel="noopener noreferrer"
          className="hover:text-gray-700"
        >
          DaisyUI
        </a>
      </div>

    </>
  )
}

export default App
