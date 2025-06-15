import { useEffect, useRef } from 'react';

const Experience = () => {
  const timelineRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in');
          }
        });
      },
      { threshold: 0.1 }
    );

    const timelineItems = document.querySelectorAll('.timeline-item');
    timelineItems.forEach((item) => observer.observe(item));

    return () => {
      timelineItems.forEach((item) => observer.unobserve(item));
    };
  }, []);

  const experiences = [
    {
      title: "Data Scientist",
      company: "Tibet Data",
      period: "2022 - Present",
      description: "Leading data science initiatives and developing machine learning solutions for business problems.",
      achievements: [
        "Developed and deployed ML models for predictive analytics",
        "Implemented MLOps practices for model lifecycle management",
        "Led data-driven decision making processes"
      ]
    },
    {
      title: "Data Analyst",
      company: "Tibet Data",
      period: "2021 - 2022",
      description: "Analyzed complex datasets and created visualizations to drive business insights.",
      achievements: [
        "Created interactive dashboards for business metrics",
        "Conducted statistical analysis for business optimization",
        "Collaborated with cross-functional teams on data projects"
      ]
    },
    {
      title: "Research Assistant",
      company: "University of Washington",
      period: "2020 - 2021",
      description: "Assisted in research projects focusing on data analysis and machine learning applications.",
      achievements: [
        "Contributed to academic research papers",
        "Implemented machine learning algorithms",
        "Conducted data collection and preprocessing"
      ]
    }
  ];

  return (
    <section id="experience" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Career Journey
        </h2>
        <div className="relative" ref={timelineRef}>
          {/* Timeline line */}
          <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-0.5 bg-gradient-to-b from-blue-500 to-purple-500" />
          
          {/* Timeline items */}
          <div className="space-y-12">
            {experiences.map((exp, index) => (
              <div
                key={index}
                className={`timeline-item relative flex items-center ${
                  index % 2 === 0 ? 'justify-start' : 'justify-end'
                } opacity-0`}
              >
                <div className={`w-5/12 ${index % 2 === 0 ? 'pr-8' : 'pl-8'}`}>
                  <div className="bg-gray-900 rounded-lg p-6 transform transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xl font-bold text-white">{exp.title}</h3>
                      <span className="text-sm text-blue-400">{exp.period}</span>
                    </div>
                    <h4 className="text-lg text-gray-300 mb-2">{exp.company}</h4>
                    <p className="text-gray-400 mb-4">{exp.description}</p>
                    <ul className="space-y-2">
                      {exp.achievements.map((achievement, idx) => (
                        <li key={idx} className="flex items-start">
                          <svg
                            className="w-5 h-5 text-blue-500 mr-2 mt-1"
                            fill="none"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="text-gray-300">{achievement}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                {/* Timeline dot */}
                <div className="absolute left-1/2 transform -translate-x-1/2 w-4 h-4 bg-blue-500 rounded-full" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience; 