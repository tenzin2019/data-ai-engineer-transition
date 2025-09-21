import React from 'react';

const About = () => {
  return (
    <section id="about" className="py-20 bg-black">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          About Me
        </h2>
        
        {/* Hero Statement - Customer-Focused Opening */}
        <div className="text-center mb-16">
          <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-8 border border-blue-500/20">
            <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
              Turning Data into Real Solutions
            </h3>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Bridging AI innovation and business needs to deliver real-world impact.
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
                  I help businesses turn data into actionable solutions — combining machine learning, data science, and AI solution to solve real-world problems.
                </p>
                <p>
                  My expertise spans the full project lifecycle: from data exploration and predictive modeling to deploying scalable systems in production. 
                </p>
                <p>
                  Whether it’s building ML models, automating workflows, or driving insights, I focus on solutions that deliver lasting business impact—not just experiments.
                </p>
                <p>
                  With a Master's in IT with AI specialization and 5+ years of experience, I build real-world AI and data science solutions that go from idea to deployment.
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
                My mission is simple: 
                <span className="text-blue-400 font-semibold"> to build systems that solve real problems and actually get deployed.</span>.
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
                  <span className="text-gray-300">Help teams ship data-driven features faster than their competitors</span>
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
                  <span className="text-gray-300">Not about AI hype — just simple, practical solutions that work</span>
                </div>
                <div className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Someone who promises AI/ML magic without the engineering</span>
                </div>
                <div className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Someone who disappears after the handover</span>
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
                  <span className="text-gray-300"><span className="text-white font-semibold">5+ years</span> building production ML systems</span>
                </div>
                <div className="flex items-start">
                  <span className="text-green-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">Automated</span> refund payment eligibility process</span>
                </div>
                <div className="flex items-start">
                  <span className="text-yellow-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300"><span className="text-white font-semibold">Azure certified</span> Azure AI Engineer Associate</span>
                </div>
              </div>
            </div>

            {/* Personality - Beyond the Code */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
              <h4 className="text-xl font-bold text-white mb-4 flex items-center">
                <span className="text-purple-500 mr-2">🎭</span>
                When I'm Not Coding
              </h4>
          
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  <span>At the gym, listening to music, relaxing in the Sauna or sometimes just catching up on sleep</span>
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