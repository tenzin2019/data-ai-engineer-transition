import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import About from './components/About'
import Experience from './components/Experience'
import Projects from './components/Projects'
import GenAIProjects from './components/GenAIProjects'
import Skills from './components/Skills'
import Education from './components/Education'
import Certification from './components/Certification'
import Contact from './components/Contact'
import Footer from './components/Footer'

function App() {
  const [isLoading, setIsLoading] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingMessage, setLoadingMessage] = useState('Initializing...')

  useEffect(() => {
    // Simulate progressive loading with meaningful messages
    const loadingSteps = [
      { progress: 20, message: 'Loading portfolio assets...' },
      { progress: 40, message: 'Preparing your experience...' },
      { progress: 60, message: 'Setting up interactive elements...' },
      { progress: 80, message: 'Almost ready...' },
      { progress: 100, message: 'Welcome!' }
    ]

    let currentStep = 0
    const interval = setInterval(() => {
      if (currentStep < loadingSteps.length) {
        const step = loadingSteps[currentStep]
        setLoadingProgress(step.progress)
        setLoadingMessage(step.message)
        currentStep++
      } else {
        clearInterval(interval)
        setTimeout(() => setIsLoading(false), 500)
      }
    }, 300)

    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <div className="h-screen w-screen bg-black flex items-center justify-center overflow-hidden">
        {/* Animated background particles */}
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-blue-500 rounded-full animate-pulse loading-particle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${2 + Math.random() * 2}s`,
                opacity: 0.6 + Math.random() * 0.4
              }}
            />
          ))}
          {/* Additional floating elements */}
          {[...Array(5)].map((_, i) => (
            <div
              key={`float-${i}`}
              className="absolute w-2 h-2 bg-purple-500 rounded-full animate-float"
              style={{
                left: `${20 + Math.random() * 60}%`,
                top: `${20 + Math.random() * 60}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${3 + Math.random() * 2}s`
              }}
            />
          ))}
        </div>

        <div className="text-center relative z-10 max-w-md mx-auto px-6">
          {/* Animated logo with brand reinforcement */}
          <div className="relative mb-8">
            <div className="w-24 h-24 mx-auto rounded-full bg-gradient-to-r from-blue-500 to-purple-500 p-1 animate-pulse">
              <div className="w-full h-full rounded-full bg-black flex items-center justify-center">
                <span className="text-2xl font-mono font-bold bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
                  TJ
                </span>
              </div>
            </div>
            
            {/* Orbiting elements for visual interest */}
            <div className="absolute inset-0 animate-spin" style={{ animationDuration: '8s' }}>
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-purple-500 rounded-full"></div>
            </div>
            <div className="absolute inset-0 animate-spin" style={{ animationDuration: '6s', animationDirection: 'reverse' }}>
              <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-blue-400 rounded-full"></div>
            </div>
          </div>
          
          {/* Loading text with personality */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-blue-500 mb-2 animate-pulse">
              Loading Portfolio
            </h1>
            <p className="text-gray-400 text-sm animate-pulse">
              {loadingMessage}
            </p>
          </div>
          
          {/* Progress bar with smooth animation */}
          <div className="mb-6">
            <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden mx-auto">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${loadingProgress}%` }}
              ></div>
            </div>
            <p className="text-gray-500 text-xs mt-2">
              {loadingProgress}% complete
            </p>
          </div>
          
          {/* Engaging loading tips */}
          <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-300 text-sm mb-2">
              💡 While you wait:
            </p>
            <div className="text-gray-400 text-xs space-y-1">
              <p>• I specialize in Data Science & AI Engineering</p>
              <p>• 5+ years of experience in ML & cloud technologies</p>
              <p>• Passionate about building scalable AI solutions</p>
            </div>
          </div>
          
          {/* Social proof indicator */}
          <div className="mt-6 text-center">
            <p className="text-gray-500 text-xs">
              Trusted by professionals at Fair Work Commission
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="fixed inset-0 bg-gradient-to-b from-blue-900/20 to-black pointer-events-none" />
      <Navbar />
      <main className="relative z-10">
        <Hero />
        <About />
        <Experience />
        <Projects />
        <GenAIProjects />
        <Skills />
        <Education />
        <Certification />
        <Contact />
      </main>
      <Footer />
    </div>
  )
}

export default App 