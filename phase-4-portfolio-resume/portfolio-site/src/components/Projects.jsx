const Projects = () => {
  const projects = [
    {
      title: 'Loan Default Prediction',
      description: 'MLOps project implementing end-to-end machine learning pipeline with experiment tracking and model deployment.',
      tech: ['Python', 'MLflow', 'Docker', 'FastAPI'],
      image: '/placeholder-loan.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-1-mlops/loan-default-prediction'
    },
    {
      title: 'Financial Behavior Insights',
      description: 'Cloud-native AI application analyzing financial patterns using Azure ML and cloud services.',
      tech: ['Azure ML', 'Python', 'Docker', 'Azure Functions'],
      image: '/placeholder-finance.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-2-cloud-ai/financial-behavior-insights'
    },
    {
      title: 'AUS Market Analysis',
      description: 'Data analysis and visualization project for Australian market trends and patterns.',
      tech: ['Python', 'Pandas', 'Plotly', 'Jupyter'],
      image: '/placeholder-market.jpg',
      link: 'https://github.com/tenzin2019/data-ai-engineer-transition/tree/main/phase-1-mlops/aus-market-analysis'
    }
  ];

  return (
    <section id="projects" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Featured Projects
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div
              key={index}
              className="group relative bg-gray-900 rounded-lg overflow-hidden transform transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl"
            >
              <div className="aspect-w-16 aspect-h-9 bg-gray-800">
                <div className="w-full h-48 bg-gradient-to-br from-blue-900 to-purple-900 opacity-50" />
              </div>
              <div className="p-6">
                <h3 className="text-xl font-bold mb-2 text-white group-hover:text-blue-400 transition-colors duration-200">
                  {project.title}
                </h3>
                <p className="text-gray-400 mb-4">
                  {project.description}
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.tech.map((tech, techIndex) => (
                    <span
                      key={techIndex}
                      className="px-3 py-1 text-sm bg-blue-900/50 text-blue-300 rounded-full"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
                <a
                  href={project.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-blue-400 hover:text-blue-300 transition-colors duration-200"
                >
                  View Project
                  <svg
                    className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-200"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </a>
              </div>
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Projects; 