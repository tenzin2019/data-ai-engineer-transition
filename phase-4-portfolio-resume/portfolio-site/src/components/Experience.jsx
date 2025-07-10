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
      company: "Fair Work Commission",
      period: "July 2023 - Present (2 years)",
      location: "Canberra, Australian Capital Territory, Australia",
      description: "",
      achievements: []
    },
    {
      title: "Junior Data Scientist",
      company: "Fair Work Commission",
      period: "July 2021 - July 2023 (2 years 1 month)",
      location: "Canberra, Australian Capital Territory, Australia",
      description: "",
      achievements: []
    },
    {
      title: "Data Scientist",
      company: "PIT-M3D International",
      period: "March 2020 - June 2021 (1 year 4 months)",
      location: "Sydney, New South Wales, Australia",
      description: "",
      achievements: [
        "Led extensive data mining operations",
        "Developed and maintained data pipelines",
        "Collaborated with cross-functional teams",
        "Implemented data visualization tools",
        "Optimized data processing workflows",
        "Developed and maintained data pipelines",
        "Collaborated with cross-functional teams",
      ]
    },
    {
      title: "Global Scope Business Consultant",
      company: "Rec Alley",
      period: "November 2019 - December 2019 (2 months)",
      location: "Gregory Hills, NSW",
      description: "",
      achievements: []
    }
  ];

  return (
    <section id="experience" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Career Journey
        </h2>
        
        {/* Mobile-first timeline layout */}
        <div className="space-y-8 md:hidden">
          {experiences.map((exp, index) => (
            <div
              key={index}
              className="timeline-item opacity-0 bg-gray-900 rounded-lg p-6 transform transition-all duration-300 hover:scale-105 hover:shadow-xl"
            >
              <div className="flex flex-col space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
                  <h3 className="text-xl font-bold text-white">{exp.title}</h3>
                  <span className="text-sm text-blue-400 mt-1 sm:mt-0">{exp.period}</span>
                </div>
                <h4 className="text-lg text-gray-300">{exp.company}</h4>
                <p className="text-gray-400 text-sm">{exp.location}</p>
                {exp.description && (
                  <p className="text-gray-400">{exp.description}</p>
                )}
                {exp.achievements.length > 0 && (
                  <ul className="space-y-2 mt-4">
                    {exp.achievements.map((achievement, idx) => (
                      <li key={idx} className="flex items-start">
                        <svg
                          className="w-4 h-4 text-blue-500 mr-2 mt-1 flex-shrink-0"
                          fill="none"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="text-gray-300 text-sm">{achievement}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Desktop timeline layout */}
        <div className="relative hidden md:block" ref={timelineRef}>
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
                    <p className="text-gray-400 mb-4">{exp.location}</p>
                    {exp.description && (
                      <p className="text-gray-400 mb-4">{exp.description}</p>
                    )}
                    {exp.achievements.length > 0 && (
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
                    )}
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