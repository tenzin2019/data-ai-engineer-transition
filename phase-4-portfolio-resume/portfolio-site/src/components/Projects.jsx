import React from 'react';

const Projects = () => {
  const projects = [
    {
      title: 'Loan Default Prediction',
      description: 'MLOps project implementing end-to-end machine learning pipeline with experiment tracking and model deployment.',
      tech: ['Python', 'MLflow', 'Docker', 'FastAPI'],
      image: '/placeholder-loan.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-1-mlops/loan-default-prediction',
      features: [
        'Automated data preprocessing and feature engineering',
        'Experiment tracking with MLflow',
        'Model deployment with FastAPI and Docker',
        'End-to-end reproducible pipeline',
        'Performance monitoring and logging'
      ],
      progress: 100
    },
    {
      title: 'Financial Behavior Insights',
      description: 'Cloud-native AI application analyzing financial patterns using Azure ML and cloud services.',
      tech: ['Azure ML', 'Python', 'Docker', 'Azure Functions'],
      image: '/placeholder-finance.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-2-cloud-ai/financial-behavior-insights',
      features: [
        'Cloud-based data ingestion and processing',
        'Predictive analytics for financial behavior',
        'Integration with Azure ML pipelines',
        'Interactive dashboards and reporting',
        'Scalable deployment with Docker and Azure Functions'
      ],
      progress: 100
    },
    {
      title: 'AUS Market Analysis',
      description: 'Data analysis and visualization project for Australian market trends and patterns.',
      tech: ['Python', 'Pandas', 'Plotly', 'Jupyter'],
      image: '/placeholder-market.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-1-mlops/aus-market-analysis',
      features: [
        'Comprehensive data cleaning and wrangling',
        'Exploratory data analysis (EDA)',
        'Interactive visualizations with Plotly',
        'Jupyter Notebook documentation',
        'Market trend insights and reporting'
      ],
      progress: 100
    }
  ];

  const getProgressColor = (progress) => {
    if (progress >= 75) return 'bg-green-500';
    if (progress >= 50) return 'bg-yellow-500';
    if (progress >= 25) return 'bg-blue-500';
    return 'bg-purple-500';
  };

  return (
    <section id="projects" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Data Science Projects
        </h2>
        <p className="text-center text-gray-400 mb-12 max-w-3xl mx-auto">
          A selection of end-to-end data science projects, from MLOps to cloud-native analytics, demonstrating real-world impact and production-ready solutions.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <div
              key={index}
              className="group relative bg-gray-900 rounded-lg overflow-hidden transform transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl border border-gray-800"
            >
              {/* Project Header */}
              <div className="p-6 border-b border-gray-800">
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors duration-200">
                    {project.title}
                  </h3>
                  <a
                    href={project.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3 py-1 text-xs font-medium rounded-full bg-blue-900/30 text-blue-400 border border-blue-700/30 hover:bg-blue-900/50 transition-colors duration-200"
                  >
                    View Repo
                  </a>
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
                        <span className="text-blue-400 mr-2 mt-1">▸</span>
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
                        className="px-3 py-1 text-xs bg-blue-900/50 text-blue-300 rounded-full border border-blue-700/50"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
                {/* Project Link Button */}
                <div className="flex items-center justify-end">
                  <a
                    href={project.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-4 py-2 text-sm bg-blue-600/20 text-blue-400 rounded-lg border border-blue-600/50 hover:bg-blue-600/30 transition-colors duration-200"
                  >
                    View Project
                  </a>
                </div>
              </div>
              {/* Overlay for Hover */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
            </div>
          ))}
        </div>
        {/* Call to Action */}
        <div className="mt-12 text-center">
          <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-lg p-8 border border-blue-800/50">
            <h3 className="text-2xl font-bold text-white mb-4">
              Want to see more?
            </h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              Explore the full code and documentation for each project on GitHub, or reach out to discuss how these solutions can be adapted for your business.
            </p>
            <a
              href="#contact"
              className="inline-block px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 transform hover:scale-105"
            >
              Contact Me
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Projects; 