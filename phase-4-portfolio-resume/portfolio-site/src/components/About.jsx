import React from 'react';

const About = () => {
  return (
    <section id="about" className="py-20 bg-black">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          About Me
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          {/* Personal Story & Journey */}
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-white mb-6">My Journey</h3>
              <div className="space-y-6 text-gray-300 leading-relaxed">
                <p>
                  I'm a passionate Data Scientist actively transitioning into <span className="text-blue-400 font-semibold">AI Engineering</span> and <span className="text-purple-400 font-semibold">Generative AI</span>, with over 4 years of experience 
                  in machine learning, data analysis, and cloud technologies. My journey began with a Master's in 
                  Information Technology with AI specialization from Western Sydney University.
                </p>
                <p>
                  Currently working at the Fair Work Commission as a Data Scientist, I've developed expertise in 
                  building end-to-end ML pipelines, implementing MLOps practices, and deploying AI solutions on 
                  cloud platforms like Azure. I'm now focused on <span className="text-green-400 font-semibold">scaling AI systems</span> and 
                  <span className="text-green-400 font-semibold"> implementing GenAI solutions</span>.
                </p>
                <p>
                  My transition involves moving from <span className="text-yellow-400 font-semibold">experimental ML models</span> to <span className="text-blue-400 font-semibold">production AI systems</span>, 
                  from <span className="text-yellow-400 font-semibold">data analysis</span> to <span className="text-blue-400 font-semibold">AI infrastructure engineering</span>, and from 
                  <span className="text-yellow-400 font-semibold">model development</span> to <span className="text-purple-400 font-semibold">GenAI application deployment</span>.
                </p>
              </div>
            </div>

            {/* Mission & Values */}
            <div className="bg-gray-900 rounded-lg p-6">
              <h4 className="text-xl font-bold text-white mb-4">My Mission</h4>
              <p className="text-gray-300 mb-4">
                I'm dedicated to bridging the gap between cutting-edge AI research and production-ready applications. 
                My mission is to build scalable, reliable AI systems that solve real-world problems and drive business value.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="text-blue-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Creating AI solutions that are both innovative and practical</span>
                </div>
                <div className="flex items-start">
                  <span className="text-purple-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Building robust MLOps pipelines for reliable AI deployment</span>
                </div>
                <div className="flex items-start">
                  <span className="text-green-500 mr-3 mt-1">▸</span>
                  <span className="text-gray-300">Mentoring and sharing knowledge with the AI community</span>
                </div>
              </div>
            </div>
          </div>

          {/* Professional Summary & Goals */}
          <div className="space-y-6">
            <div className="bg-gray-900 rounded-lg p-6">
              <h4 className="text-xl font-bold text-white mb-4">What I Do</h4>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">▸</span>
                  Design and scale production AI infrastructure
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">▸</span>
                  Implement GenAI applications and LLM pipelines
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">▸</span>
                  Build MLOps and AIOps automation systems
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">▸</span>
                  Optimize AI model performance and deployment
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">▸</span>
                  Architect cloud-native AI solutions on Azure
                </li>
              </ul>
            </div>

            <div className="bg-gray-900 rounded-lg p-6">
              <h4 className="text-xl font-bold text-white mb-4">What I'm Looking For</h4>
              <p className="text-gray-300 mb-3">
                I'm seeking opportunities to:
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-purple-500 mr-2">▸</span>
                  Lead AI Engineering and GenAI initiatives
                </li>
                <li className="flex items-start">
                  <span className="text-purple-500 mr-2">▸</span>
                  Architect scalable AI infrastructure
                </li>
                <li className="flex items-start">
                  <span className="text-purple-500 mr-2">▸</span>
                  Build enterprise GenAI applications
                </li>
                <li className="flex items-start">
                  <span className="text-purple-500 mr-2">▸</span>
                  Implement AIOps and automation systems
                </li>
              </ul>
            </div>

            {/* Personal Touch */}
            <div className="bg-gray-900 rounded-lg p-6">
              <h4 className="text-xl font-bold text-white mb-4">Beyond the Code</h4>
              <p className="text-gray-300 mb-3">
                When I'm not architecting AI systems or optimizing ML pipelines, you'll find me:
              </p>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  Exploring the latest developments in Generative AI and LLMs
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  Contributing to open-source AI projects and the tech community
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  Mentoring aspiring data scientists and AI engineers
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">▸</span>
                  Staying current with emerging AI technologies and best practices
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-12 text-center">
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-lg p-8">
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Build the Future of AI Together?
            </h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              Whether you're looking to implement AI solutions, build scalable ML infrastructure, 
              or explore the possibilities of Generative AI, I'd love to connect and discuss how 
              we can bring your AI vision to life.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="#projects"
                className="inline-block px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 transform hover:scale-105"
              >
                View My Projects
              </a>
              <a
                href="#contact"
                className="inline-block px-8 py-3 border border-blue-600 text-blue-600 rounded-lg hover:bg-blue-600 hover:text-white transition-colors duration-200 transform hover:scale-105"
              >
                Let's Connect
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About; 