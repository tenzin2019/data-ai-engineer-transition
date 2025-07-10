import React from 'react';

const Certification = () => (
  <section id="certification" className="py-20 bg-black">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
        Licenses & Certifications
      </h2>
      <div className="space-y-8">
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold text-white mb-2">Large Language Models for Everyone</h3>
          <p className="text-gray-400">Data Science Dojo</p>
          <p className="text-gray-400">Issued May 2024</p>
          <p className="text-gray-400">Credential ID: ded24dcb8eb409</p>
          <a href="https://verify.datasciencedojo.com/verify/ded24dcb8eb409" className="text-blue-400 hover:underline">Show credential</a>
        </div>
        <div className="bg-gray-900 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold text-white mb-2">Microsoft Certified: Azure AI Engineer Associate</h3>
          <p className="text-gray-400">Microsoft</p>
          <p className="text-gray-400">Issued May 2024 - Expires May 2026</p>
          <p className="text-gray-400">Credential ID: 939651E830D55C01</p>
          <a href="https://learn.microsoft.com/en-us/users/tenzinjamyang-3790/credentials/939651e830d55c01?ref=https%3A%2F%2Fwww.linkedin.com%2F" className="text-blue-400 hover:underline">Show credential</a>
        </div>
      </div>
    </div>
  </section>
);

export default Certification; 