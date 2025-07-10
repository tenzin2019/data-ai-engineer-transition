import React from 'react';

const About = () => {
  return (
    <section id="about" className="py-20 bg-black">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          About Me
        </h2>
        
        {/* Hero Statement - Customer-Focused Opening */}
        <div className="text-center mb-16">
          <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-8 border border-blue-500/20">
            <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
              I help companies build AI systems that actually work in production
            </h3>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              No more experimental models that never see the light of day. I bridge the gap between 
              cutting-edge AI research and real-world business solutions that scale.
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          {/* Personal Story & Journey - More Human & Concise */}
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                <span className="text-blue-500 mr-3">▸</span>
                My Story
              </h3>
              <div className="space-y-6 text-gray-300 leading-relaxed">
                <p>
                  Picture this: A Data Scientist at the Fair Work Commission, building ML models that 
                  actually help people. That's me. But I noticed something frustrating - too many AI 
                  projects die in the lab, never making it to production.
                </p>
                <p>
                  So I decided to change that. I'm transitioning from <span className="text-yellow-400 font-semibold">experimental ML</span> to 
                  <span className="text-blue-400 font-semibold"> production AI systems</span>. From <span className="text-yellow-400 font-semibold">data analysis</span> to 
                  <span className="text-purple-400 font-semibold"> AI infrastructure engineering</span>.
                </p>
                <p>
                  My Master's in IT with AI specialization from Western Sydney University gave me the 
                  foundation. Now I'm focused on making AI work in the real world, not just in research papers.
                </p>
              </div>
            </div>

            {/* Mission - Human & Specific */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
              <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                <span className="text-green-500 mr-2">🎯</span>
                What Drives Me
              </h4>
              <p className="text-gray-300 mb-4">
                I'm tired of seeing brilliant AI research collect dust. My mission is simple: 
                <span className="text-blue-400 font-semibold"> Build AI systems that solve real problems and actually get deployed</span>.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-blue-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Turn experimental models into production-ready solutions</span>
                </div>
                <div className="flex items-start">
                  <span className="text-purple-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Build MLOps pipelines that don't break at 3 AM</span>
                </div>
                <div className="flex items-start">
                  <span className="text-green-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Help teams ship AI features faster than their competitors</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Personality & Proof */}
          <div className="space-y-6">
            {/* What I'm NOT - Personality Through Contrast */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
              <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                <span className="text-red-500 mr-2">❌</span>
                What I'm NOT
              </h4>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">A researcher who builds models that never leave the lab</span>
                </div>
                <div className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Someone who promises AI magic without the engineering</span>
                </div>
                <div className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">A consultant who disappears after the handover</span>
                </div>
              </div>
            </div>

            {/* Proof Points - Specific Achievements */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
              <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                <span className="text-green-500 mr-2">🏆</span>
                Proof It Works
              </h4>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-blue-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">60% faster</span> ML deployments with MLOps automation</span>
                </div>
                <div className="flex items-start">
                  <span className="text-purple-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">4+ years</span> building production ML systems</span>
                </div>
                <div className="flex items-start">
                  <span className="text-green-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">100%</span> of my models made it to production</span>
                </div>
                <div className="flex items-start">
                  <span className="text-yellow-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">Azure certified</span> in AI and ML engineering</span>
                </div>
              </div>
            </div>

            {/* Personality - Beyond the Code */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
              <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                <span className="text-purple-500 mr-2">🎭</span>
                When I'm Not Coding
              </h4>
              <p className="text-gray-300 mb-3">
                Because robots are boring, and humans are interesting:
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  <span>Contributing to open-source AI projects (because sharing is caring)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  <span>Mentoring junior data scientists (paying it forward)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  <span>Exploring the latest in GenAI (because the future is now)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  <span>Building things that actually work (my favorite hobby)</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Strong Call to Action - Multiple Options */}
        <div className="mt-16 text-center">
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-lg p-8 border border-blue-500/20">
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Build AI That Actually Works?
            </h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              Whether you need to deploy your first ML model or scale your AI infrastructure, 
              let's talk about making your AI dreams a production reality.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="#projects"
                className="inline-block px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                See My Work
              </a>
              <a
                href="#contact"
                className="inline-block px-8 py-3 border-2 border-blue-600 text-blue-600 rounded-lg hover:bg-blue-600 hover:text-white transition-all duration-200 transform hover:scale-105"
              >
                Let's Talk AI
              </a>
              <a
                href="#experience"
                className="inline-block px-8 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all duration-200 transform hover:scale-105"
              >
                My Experience
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About; 