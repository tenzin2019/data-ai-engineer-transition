import { useState, useEffect } from 'react';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isProjectsOpen, setIsProjectsOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMobileProjectsOpen, setIsMobileProjectsOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleProjectsClick = (e) => {
    e.preventDefault();
    setIsProjectsOpen(!isProjectsOpen);
  };

  const handleMobileProjectsClick = (e) => {
    e.preventDefault();
    setIsMobileProjectsOpen(!isMobileProjectsOpen);
  };

  const handleProjectLinkClick = (sectionId) => {
    console.log('Project link clicked:', sectionId);
    setIsProjectsOpen(false);
    setIsMobileMenuOpen(false);
    setIsMobileProjectsOpen(false);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    } else {
      console.log('Element not found:', sectionId);
    }
  };

  const handleNavLinkClick = (sectionId) => {
    console.log('Nav link clicked:', sectionId);
    setIsMobileMenuOpen(false);
    setIsMobileProjectsOpen(false);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    } else {
      console.log('Element not found:', sectionId);
    }
  };

  const toggleMobileMenu = () => {
    console.log('Toggle mobile menu');
    setIsMobileMenuOpen(!isMobileMenuOpen);
    setIsProjectsOpen(false);
  };

  // Navigation items with correct lowercase IDs
  const navItems = [
    { name: 'Home', id: 'home' },
    { name: 'About', id: 'about' },
    { name: 'Experience', id: 'experience' },
    { name: 'Skills', id: 'skills' },
    { name: 'Education', id: 'education' },
    { name: 'Certification', id: 'certification' },
    { name: 'Contact', id: 'contact' }
  ];

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${
      isScrolled ? 'bg-black/80 backdrop-blur-lg' : 'bg-transparent'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex-shrink-0">
            <a
              href="https://github.com/tenzin2019"
              target="_blank"
              rel="noopener noreferrer"
              className="text-2xl font-mono font-bold bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text hover:opacity-80 transition-opacity duration-200"
            >
              TJ
            </a>
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-8">
              {navItems.map((item) => (
                <a
                  key={item.name}
                  href={`#${item.id}`}
                  className="text-gray-300 hover:text-white px-3 py-2 text-sm font-medium transition-colors duration-200"
                >
                  {item.name}
                </a>
              ))}
              
              {/* Projects Dropdown */}
              <div className="relative">
                <button
                  onClick={handleProjectsClick}
                  className="text-gray-300 hover:text-white px-3 py-2 text-sm font-medium transition-colors duration-200 flex items-center"
                >
                  Projects
                  <svg
                    className={`ml-1 w-4 h-4 transition-transform duration-200 ${isProjectsOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                
                {/* Dropdown Menu */}
                {isProjectsOpen && (
                  <div className="absolute top-full left-0 mt-2 w-48 bg-gray-900 rounded-lg shadow-xl border border-gray-700 py-2 z-50">
                    <button
                      onClick={() => handleProjectLinkClick('projects')}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800 transition-colors duration-200"
                    >
                      Data Science Project
                    </button>
                    <button
                      onClick={() => handleProjectLinkClick('genai-projects')}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800 transition-colors duration-200"
                    >
                      GenAI Projects
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button 
              onClick={toggleMobileMenu}
              className="text-gray-300 hover:text-white p-3 transition-colors duration-200 touch-target focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-black rounded-lg"
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={isMobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            >
              {isMobileMenuOpen ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div className="md:hidden relative z-50" id="mobile-menu">
          <div className="px-2 pt-2 pb-3 space-y-1 bg-black/95 backdrop-blur-lg border-t border-gray-700" role="navigation" aria-label="Mobile navigation menu">
            {navItems.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNavLinkClick(item.id)}
                className="block w-full text-left text-gray-300 hover:text-white px-4 py-3 text-base font-medium transition-colors duration-200 min-h-[44px] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-black rounded-lg"
                aria-label={`Navigate to ${item.name} section`}
              >
                {item.name}
              </button>
            ))}
            
            {/* Mobile Projects Dropdown */}
            <div className="border-t border-gray-700 pt-2">
              <button
                onClick={handleMobileProjectsClick}
                className="block w-full text-left text-gray-300 hover:text-white px-3 py-2 text-base font-medium transition-colors duration-200 flex items-center justify-between"
              >
                Projects
                <svg
                  className={`w-4 h-4 transition-transform duration-200 ${isMobileProjectsOpen ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {isMobileProjectsOpen && (
                <div className="pl-4 space-y-1">
                  <button
                    onClick={() => handleProjectLinkClick('projects')}
                    className="block w-full text-left px-3 py-2 text-sm text-gray-400 hover:text-white transition-colors duration-200"
                  >
                    Data Science Projects
                  </button>
                  <button
                    onClick={() => handleProjectLinkClick('genai-projects')}
                    className="block w-full text-left px-3 py-2 text-sm text-gray-400 hover:text-white transition-colors duration-200"
                  >
                    GenAI Projects
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Click outside to close dropdowns - only for desktop projects dropdown */}
      {isProjectsOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => {
            setIsProjectsOpen(false);
          }}
        />
      )}
    </nav>
  );
};

export default Navbar; 