import React, { useEffect } from 'react';
import { trackSkillsInteraction } from '../utils/analytics';

// Random skill levels - acknowledging the Dunning-Kruger effect
const getRandomLevel = () => Math.floor(Math.random() * 29) + 56; // 56-84%

const Skills = () => {
  // Track skills section interaction when component mounts
  useEffect(() => {
    trackSkillsInteraction();
  }, []);

  const skillCategories = [
    {
      category: 'AI & Machine Learning',
      skills: [
        { name: 'Python', level: getRandomLevel() },
        { name: 'PyTorch', level: getRandomLevel() },
        { name: 'scikit-learn', level: getRandomLevel() },
        { name: 'Azure OpenAI', level: getRandomLevel() },
        { name: 'LangChain', level: getRandomLevel() },
        { name: 'MLflow', level: getRandomLevel() }
      ]
    },
    {
      category: 'Cloud & Infrastructure',
      skills: [
        { name: 'Azure', level: getRandomLevel() },
        { name: 'Docker', level: getRandomLevel() },
        { name: 'FastAPI', level: getRandomLevel() },
        { name: 'Azure Functions', level: getRandomLevel() },
        { name: 'Key Vault & RBAC', level: getRandomLevel() }
      ]
    },
    {
      category: 'Data & Backend',
      skills: [
        { name: 'SQL', level: getRandomLevel() },
        { name: 'Pandas', level: getRandomLevel() },
        { name: 'PostgreSQL', level: getRandomLevel() },
        { name: 'Streamlit', level: getRandomLevel() },
        { name: 'ETL/ELT Pipelines', level: getRandomLevel() }
      ]
    },
    {
      category: 'DevOps & MLOps',
      skills: [
        { name: 'GitHub Actions', level: getRandomLevel() },
        { name: 'MLOps Monitoring', level: getRandomLevel() },
        { name: 'DVC', level: getRandomLevel() },
        { name: 'Model Deployment', level: getRandomLevel() }
      ]
    }
  ];

  return (
    <section 
      id="skills" 
      className="py-20 bg-black"
      aria-labelledby="skills-heading"
      role="region"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 
          id="skills-heading"
          className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text"
        >
          Technical Skills
        </h2>
          <p className="text-center text-gray-400 mb-8 max-w-4xl mx-auto">
            Core technologies I use in production, with randomly generated proficiency levels (56-84%).
            These reflect my commitment to continuous learning and balanced self-assessment in an
            ever-evolving tech landscape.
          </p>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8">
          {skillCategories.map((category, index) => (
            <div
              key={index}
              className="bg-gray-900 rounded-lg p-6 transform transition-all duration-300 hover:scale-105 focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2 focus-within:ring-offset-black"
              role="group"
              aria-labelledby={`skill-category-${index}`}
            >
              <h3 
                id={`skill-category-${index}`}
                className="text-xl font-bold mb-6 text-white"
              >
                {category.category}
              </h3>
              <div className="space-y-4" role="list" aria-label={`${category.category} skills`}>
                {category.skills.map((skill, skillIndex) => (
                  <div key={skillIndex} role="listitem">
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-300">{skill.name}</span>
                      <span className="text-blue-400" aria-label={`Proficiency: ${skill.level} percent`}>
                        {skill.level}%
                      </span>
                    </div>
                    <div 
                      className="h-2 bg-gray-800 rounded-full overflow-hidden"
                      role="progressbar"
                      aria-valuenow={skill.level}
                      aria-valuemin="0"
                      aria-valuemax="100"
                      aria-label={`${skill.name} proficiency level`}
                    >
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${skill.level}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Skills; 