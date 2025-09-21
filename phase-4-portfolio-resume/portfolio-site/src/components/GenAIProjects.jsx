import React from 'react';

const GenAIProjects = () => {
  const genaiProjects = [
    {
      title: 'Intelligent Document Analysis System',
      description: 'Production-ready GenAI platform that processes complex business documents using Azure OpenAI and Document Intelligence. Successfully deployed with advanced text extraction, AI-powered analysis, and automated insight generation capabilities.',
      tech: ['Azure OpenAI GPT-4', 'Azure Document Intelligence', 'Python', 'Streamlit', 'FastAPI', 'PostgreSQL', 'Docker', 'SQLAlchemy'],
      image: '/placeholder-doc-analysis.jpg',
      status: 'Completed',
      progress: 100,
      features: [
        'Multi-format document processing (PDF, DOCX, XLSX, TXT)',
        'AI-powered summarization and entity extraction',
        'Sentiment analysis and key phrase identification',
        'Automated recommendation generation',
        'Real-time processing with progress tracking',
        'Comprehensive analytics dashboard',
        'Production-ready Docker deployment',
        'Advanced error handling and validation'
      ],
      githubLink: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-3-specialization/intelligent-document-analysis',
      demoLink: '#contact'
    },
    {
      title: 'AI-Powered Content Generation Platform',
      description: 'Advanced content creation system using Large Language Models for automated blog writing, social media posts, and marketing copy. Features fine-tuned models for industry-specific content generation.',
      tech: ['OpenAI GPT-4', 'LangChain', 'Python', 'FastAPI', 'Vector DB', 'RAG'],
      image: '/placeholder-genai-content.jpg',
      status: 'Under Development',
      progress: 75,
      features: [
        'Multi-format content generation (blogs, social media, emails)',
        'Industry-specific model fine-tuning',
        'Real-time content optimization',
        'Brand voice consistency',
        'SEO-optimized output'
      ]
    },
    {
      title: 'Conversational AI Assistant',
      description: 'Enterprise-grade chatbot system with advanced natural language understanding, context awareness, and seamless integration with business systems.',
      tech: ['Rasa', 'BERT', 'Python', 'Docker', 'Kubernetes', 'WebSocket'],
      image: '/placeholder-chatbot.jpg',
      status: 'Planning Phase',
      progress: 30,
      features: [
        'Multi-turn conversation handling',
        'Intent recognition and entity extraction',
        'Integration with CRM and ERP systems',
        'Multi-language support',
        'Analytics and performance monitoring'
      ]
    },
    {
      title: 'AI-Driven Code Generation Tool',
      description: 'Intelligent code generation platform that creates production-ready code from natural language descriptions, with support for multiple programming languages.',
      tech: ['GitHub Copilot API', 'CodeT5', 'Python', 'React', 'TypeScript'],
      image: '/placeholder-code-gen.jpg',
      status: 'Research Phase',
      progress: 20,
      features: [
        'Natural language to code conversion',
        'Multi-language support (Python, JavaScript, Java)',
        'Code review and optimization suggestions',
        'Integration with development workflows',
        'Custom model training for specific domains'
      ]
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'Completed':
        return 'text-green-500 bg-green-900/30 border border-green-700/50';
      case 'Under Development':
        return 'text-yellow-400 bg-yellow-900/30';
      case 'In Development':
        return 'text-blue-400 bg-blue-900/30';
      case 'Near Completion':
        return 'text-green-400 bg-green-900/30';
      case 'Planning Phase':
        return 'text-purple-400 bg-purple-900/30';
      case 'Research Phase':
        return 'text-green-400 bg-green-900/30';
      default:
        return 'text-gray-400 bg-gray-900/30';
    }
  };

  const getProgressColor = (progress) => {
    if (progress >= 75) return 'bg-green-500';
    if (progress >= 50) return 'bg-yellow-500';
    if (progress >= 25) return 'bg-blue-500';
    return 'bg-purple-500';
  };

  return (
    <section id="genai-projects" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-purple-500 to-pink-500 text-transparent bg-clip-text">
          Generative AI Projects
        </h2>
        <p className="text-center text-gray-400 mb-12 max-w-3xl mx-auto">
          Cutting-edge GenAI initiatives showcasing my transition into advanced AI engineering. 
          These projects demonstrate the future of AI-powered applications and intelligent systems.
        </p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {genaiProjects.map((project, index) => (
            <div
              key={index}
              className="group relative bg-gray-900 rounded-lg overflow-hidden transform transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl border border-gray-800"
            >
              {/* Project Header */}
              <div className="p-6 border-b border-gray-800">
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-xl font-bold text-white group-hover:text-purple-400 transition-colors duration-200">
                    {project.title}
                  </h3>
                  <span className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(project.status)}`}>
                    {project.status}
                  </span>
                </div>
                
                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>Progress</span>
                    <span>{project.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ${getProgressColor(project.progress)}`}
                      style={{ width: `${project.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Project Content */}
              <div className="p-6">
                <p className="text-gray-400 mb-6 leading-relaxed">
                  {project.description}
                </p>
                
                {/* Features List */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-white mb-3">Key Features:</h4>
                  <ul className="space-y-2">
                    {project.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start text-sm text-gray-300">
                        <span className="text-purple-400 mr-2 mt-1">▸</span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Technology Stack */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-white mb-3">Technology Stack:</h4>
                  <div className="flex flex-wrap gap-2">
                    {project.tech.map((tech, techIndex) => (
                      <span
                        key={techIndex}
                        className="px-3 py-1 text-xs bg-purple-900/50 text-purple-300 rounded-full border border-purple-700/50"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Development Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      project.status === 'Completed' 
                        ? 'bg-green-500' 
                        : project.status === 'Near Completion' 
                        ? 'bg-green-400 animate-pulse' 
                        : 'bg-yellow-400 animate-pulse'
                    }`}></div>
                    <span className="text-sm text-gray-400">
                      {project.status === 'Completed' 
                        ? 'Production Ready' 
                        : project.status === 'Near Completion' 
                        ? 'Ready for Demo' 
                        : 'Active Development'}
                    </span>
                  </div>
                  <div className="flex space-x-2">
                    {project.githubLink && (
                      <a
                        href={project.githubLink}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-4 py-2 text-sm bg-purple-600/20 text-purple-400 rounded-lg border border-purple-600/50 hover:bg-purple-600/30 transition-colors duration-200"
                      >
                        View Code
                      </a>
                    )}
                    <a
                      href={project.demoLink || '#contact'}
                      className={`px-4 py-2 text-sm rounded-lg border transition-colors duration-200 ${
                        project.status === 'Completed'
                          ? 'bg-green-600/20 text-green-400 border-green-600/50 hover:bg-green-600/30'
                          : 'bg-blue-600/20 text-blue-400 border-blue-600/50 hover:bg-blue-600/30'
                      }`}
                    >
                      {project.status === 'Completed' 
                        ? 'Live Demo' 
                        : project.status === 'Near Completion' 
                        ? 'Request Demo' 
                        : 'Coming Soon'}
                    </a>
                  </div>
                </div>
              </div>

              {/* Overlay for Project Status */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="bg-black/80 backdrop-blur-sm rounded-lg p-4">
                    {project.status === 'Completed' ? (
                      <>
                        <p className="text-white text-sm font-medium">✅ Production Ready</p>
                        <p className="text-gray-300 text-xs mt-1">
                          Successfully deployed GenAI solution with full functionality
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="text-white text-sm font-medium">🚧 Under Active Development</p>
                        <p className="text-gray-300 text-xs mt-1">
                          This project is being developed with cutting-edge GenAI technologies
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-12 text-center">
          <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-lg p-8 border border-purple-800/50">
            <h3 className="text-2xl font-bold text-white mb-4">
              Interested in GenAI Development?
            </h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              I'm actively working on these cutting-edge GenAI projects. If you're interested in 
              collaborating on AI innovation or want to discuss potential applications, let's connect!
            </p>
            <a
              href="#contact"
              className="inline-block px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-200 transform hover:scale-105"
            >
              Let's Build the Future Together
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default GenAIProjects; 