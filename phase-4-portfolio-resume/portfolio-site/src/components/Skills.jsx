const Skills = () => {
  const skillCategories = [
    {
      category: 'Machine Learning',
      skills: [
        { name: 'Python', level: 90 },
        { name: 'PyTorch', level: 85 },
        { name: 'scikit-learn', level: 90 },
        { name: 'MLflow', level: 85 }
      ]
    },
    {
      category: 'Cloud & DevOps',
      skills: [
        { name: 'Azure ML', level: 85 },
        { name: 'Docker', level: 80 },
        { name: 'GitHub Actions', level: 75 },
        { name: 'FastAPI', level: 85 }
      ]
    },
    {
      category: 'Data Engineering',
      skills: [
        { name: 'SQL', level: 85 },
        { name: 'Pandas', level: 90 },
        { name: 'DVC', level: 80 },
        { name: 'Airflow', level: 75 }
      ]
    }
  ];

  return (
    <section id="skills" className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
          Technical Skills
        </h2>
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
                        style={{ width: '0%' }}
                        onMouseEnter={(e) => {
                          e.target.style.width = `${skill.level}%`;
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.width = '0%';
                        }}
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