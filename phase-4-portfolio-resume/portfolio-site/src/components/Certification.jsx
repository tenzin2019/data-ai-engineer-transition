import React from 'react';

const Certification = () => (
  <section id="certification" className="py-20 bg-black">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <h2 className="text-4xl font-bold text-center mb-4 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
        Licenses & Certifications
      </h2>
      <div className="space-y-8">
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg transform transition-all duration-300 hover:scale-105">
          <div className="flex items-center mb-4">
            <img 
              src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" 
              alt="Microsoft" 
              className="w-12 h-12 mr-4"
            />
            <div>
              <h3 className="text-xl font-bold text-white">Microsoft Certified: Azure AI Engineer Associate</h3>
              <p className="text-gray-400">Microsoft</p>
            </div>
          </div>
          <p className="text-gray-400">Issued May 2024 - Expires May 2026</p>
          <p className="text-gray-400">Credential ID: 939651E830D55C01</p>
          <a href="https://learn.microsoft.com/en-us/users/tenzinjamyang-3790/credentials/939651e830d55c01?ref=https%3A%2F%2Fwww.linkedin.com%2F" className="text-blue-400 hover:underline">Show credential</a>
        </div>
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg transform transition-all duration-300 hover:scale-105">
          <div className="flex items-center mb-4">
            <img 
              src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" 
              alt="IBM" 
              className="w-12 h-12 mr-4"
            />
            <div>
              <h3 className="text-xl font-bold text-white">IBM Generative AI Applications Specialist</h3>
              <p className="text-gray-400">Coursera</p>
            </div>
          </div>
          <p className="text-gray-400">Issued September 2024</p>
          <p className="text-gray-400">Credential ID: 587c62d5-6c0f-4d44-bfa5-c600268f6886</p>
          <a href="https://www.credly.com/badges/587c62d5-6c0f-4d44-bfa5-c600268f6886/linked_in_profile" className="text-blue-400 hover:underline">Show credential</a>
        </div>
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg transform transition-all duration-300 hover:scale-105">
          <div className="flex items-center mb-4">
            <img 
              src="https://datasciencedojo.com/wp-content/uploads/DSD-Logo-Updated-2048x523.png.webp" 
              alt="Data Science Dojo" 
              className="w-12 h-12 mr-4"
            />
            <div>
              <h3 className="text-xl font-bold text-white">Large Language Models for Everyone</h3>
              <p className="text-gray-400">Data Science Dojo</p>
            </div>
          </div>
          <p className="text-gray-400">Issued May 2024</p>
          <p className="text-gray-400">Credential ID: ded24dcb8eb409</p>
          <a href="https://verify.datasciencedojo.com/verify/ded24dcb8eb409" className="text-blue-400 hover:underline">Show credential</a>
        </div>
      </div>
    </div>
  </section>
);

export default Certification; 