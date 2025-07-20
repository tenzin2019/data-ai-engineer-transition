import React from 'react';

const Education = () => (
  <section id="education" className="py-20 bg-black">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <h2 className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
        Education
      </h2>
      <div className="space-y-8">
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg transform transition-all duration-300 hover:scale-105">
          <div className="flex items-center mb-2">
            <img src="https://upload.wikimedia.org/wikipedia/en/f/f0/Western_Sydney_University_Crest.png" alt="Western Sydney University" className="w-10 h-10 mr-4" />
            <div>
              <h3 className="text-xl font-bold text-white">Western Sydney University</h3>
              <p className="text-gray-400">Master's degree, Information Technology / Artificial Intelligence</p>
              <p className="text-gray-400">2018 - 2020 &nbsp; | &nbsp; Grade: 6</p>
              <p className="text-blue-400 mt-1">Team Management, Artificial Intelligence (AI) and +3 skills</p>
            </div>
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg transform transition-all duration-300 hover:scale-105">
          <div className="flex items-center mb-2">
            <img src="https://upload.wikimedia.org/wikipedia/en/d/dd/Official_Logo_of_CHRIST%28Deemed_to_be_University%29%2C_bangalore.jpg" alt="Christ University, Bangalore" className="w-10 h-10 mr-4" />
            <div>
              <h3 className="text-xl font-bold text-white">Christ University, Bangalore</h3>
              <p className="text-gray-400">Bachelor of Science - BS, Computer Science</p>
              <p className="text-gray-400">2010 - 2013</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

export default Education; 