import React from 'react';

const getRandomLevel = () => Math.floor(Math.random() * 45) + 55; // 55-99

const Skills = () => {
  const skillCategories = [
    {
      category: 'Machine Learning',
      skills: [
        { name: 'Python', level: getRandomLevel() },
        { name: 'PyTorch', level: getRandomLevel() },
        { name: 'scikit-learn', level: getRandomLevel() },
        { name: 'MLflow', level: getRandomLevel() }
      ]
    },
    {
      category: 'Cloud & DevOps',
      skills: [
        { name: 'Azure ML', level: getRandomLevel() },
        { name: 'Docker', level: getRandomLevel() },
        { name: 'GitHub Actions', level: getRandomLevel() },
        { name: 'FastAPI', level: getRandomLevel() }
      ]
    },
    {
      category: 'Data Engineering',
      skills: [
        { name: 'SQL', level: getRandomLevel() },
        { name: 'Pandas', level: getRandomLevel() },
        { name: 'DVC', level: getRandomLevel() },
        { name: 'Airflow', level: getRandomLevel() }
      ]
    }
  ];

  return (
    <section id="skills" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Technical Skills
        </h2>
        <p className="text-center text-gray-400 mb-8 max-w-2xl mx-auto">
          Skill levels shown below are just estimates and change every time you load the page. I believe there is always room for improvement — reflecting how all developers feel: the more you know, the less you think you know! However, I am always open to learning new skills and technologies.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {skillCategories.map((category, index) => (
            <div
              key={index}
              className="bg-gray-900 rounded-lg p-6 transform transition-all duration-300 hover:scale-105"
            >
              <h3 className="text-xl font-bold mb-6 text-white">
                {category.category}
              </h3>
              <div className="space-y-4">
                {category.skills.map((skill, skillIndex) => (
                  <div key={skillIndex}>
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-300">{skill.name}</span>
                      <span className="text-blue-400">{skill.level}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
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