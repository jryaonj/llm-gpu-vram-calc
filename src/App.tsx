import './App.css'
import LLMVRAMCalculator from './components/LlmGpuVramCalculator'
import { Github, Heart } from 'lucide-react'

function App() {
  return (
    <>
      <div className="llm-vram-calculator-app">
        {/* <h1>LLM VRAM Calculator</h1> */}
        <LLMVRAMCalculator />
      </div>

      <footer className="border-t border-base-200 mt-8 pt-6 pb-4">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-base-content/70">
              <span>© 2025</span>
              <a 
                href="https://github.com/jryaonj"
                target="_blank"
                rel="noopener noreferrer"
                className="link link-hover font-medium"
              >
                jryaonj
              </a>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 text-sm text-base-content/70">
                <span>Made with</span>
                <Heart className="w-4 h-4 text-error animate-pulse" />
                <span>by</span>
                <a 
                  href="https://github.com/jryaonj/llm-gpu-vram-calc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="link link-hover font-medium flex items-center gap-1"
                >
                  <Github className="w-4 h-4" />
                  <span>llm-gpu-vram-calc</span>
                </a>
              </div>
            </div>

            <div className="flex items-center gap-4 text-sm text-base-content/70">
              <span>Powered by</span>
              <div className="flex items-center gap-3">
                <a 
                  href="https://react.dev" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="link link-hover font-medium transition-colors hover:text-primary"
                >
                  React
                </a>
                <span>·</span>
                <a
                  href="https://daisyui.com"
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="link link-hover font-medium transition-colors hover:text-primary"
                >
                  DaisyUI
                </a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </>
  )
}

export default App
