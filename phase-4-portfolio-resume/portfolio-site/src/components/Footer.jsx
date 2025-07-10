const Footer = () => {
  return (
    <footer className="bg-black border-t border-gray-800">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-mono font-bold bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text">
              TJ
            </span>
            <span className="text-gray-400">© 2024 Tenzin Jamyang. All rights reserved.</span>
          </div>
          <div className="mt-4 md:mt-0">
            <nav className="flex space-x-6">
              <a
                href="#experience"
                className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
              >
                Experience
              </a>
              <a
                href="#projects"
                className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
              >
                Projects
              </a>
              <a
                href="#skills"
                className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
              >
                Skills
              </a>
              <a
                href="#contact"
                className="text-gray-400 hover:text-blue-400 transition-colors duration-200"
              >
                Contact
              </a>
            </nav>
          </div>
        </div>
        <div className="mt-8 border-t border-gray-800 pt-8">
          <p className="text-center text-gray-400 text-sm">
            Built with React and TailwindCSS • Deployed on GitHub Pages
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 