import React, { useEffect, useRef, useState } from "react";
import CustomBarChart from "../components/CustomBarChart";
import { useNavigate } from "react-router-dom";
import Plasma from "../components/Plasma";
import profile from "../assets/profile-pic-2.webp"

const buttons = [
  "Edit profile",
  "Earnings",
  "Settings",
  "Comments",
  "Your blogs",
];

const ProfilePage = () => {
  const navigate = useNavigate();
  const containerRef = useRef(null);
  const [highlightStyle, setHighlightStyle] = useState({});
  const [isHovering, setIsHovering] = useState(false);
  const btnRefs = useRef([]);

  useEffect(() => {
    console.log(containerRef.current);
  }, []);

  const handleHover = (index) => {
    const el = btnRefs.current[index];
    if (el) {
      const { offsetLeft, offsetTop, offsetWidth, offsetHeight } = el;
      setHighlightStyle({
        left: offsetLeft,
        top: offsetTop,
        width: offsetWidth,
        height: offsetHeight,
        opacity: 1,
      });
      setIsHovering(true);
    }
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };

  return (
    <main className="relative overflow-x-hidden min-h-screen flex flex-col text-white">
      {/* Prism Background */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <Plasma
          color="#FFBF00"
          speed={0.8}
          direction="forward"
          scale={2}
          opacity={100}
          mouseInteractive={false}
        />
      </div>

      {/* Top Navigation Section */}
      <section className="flex flex-col sm:flex-row mt-24 px-6 sm:px-10 py-10 items-center gap-6 w-full font-bold relative z-10">
        <img
          src={profile}
          alt="profile-pic"
          className="cursor-pointer h-[80px] sm:h-[100px] w-[80px] sm:w-[100px] rounded-full object-cover"
        />

        <div
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
          className="relative flex flex-wrap sm:flex-nowrap justify-center sm:justify-between gap-3 sm:gap-6 border border-white/10 backdrop-blur-full rounded-full px-1 py-1 shadow-[0_20px_80px_rgba(0,0,0,0.35)]"
        >
          <div
            className="absolute border-3 border-white rounded-full transition-all duration-300 ease-in-out z-0 pointer-events-none"
            style={{
              ...highlightStyle,
              opacity: isHovering ? 1 : 0,
              transition: "all 300ms ease-in-out",
            }}
          />
          {buttons.map((label, index) => (
            <button
              key={label}
              ref={(el) => (btnRefs.current[index] = el)}
              onClick={() => {
                if (label === "Your blogs") {
                  navigate("/senddocument");
                }
              }}
              onMouseEnter={() => handleHover(index)}
              className="relative z-10 text-lg sm:text-xl md:text-2xl px-4 py-2 transition-colors duration-300 cursor-pointer hover:text-amber-300 whitespace-nowrap"
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      {/* Analytics Section */}
      <section className="mt-10 px-6 sm:px-10 relative z-10">
        <div className="rounded-2xl backdrop-blur-full border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] p-6 sm:p-10">
          <h2 className="text-xl sm:text-2xl font-semibold text-white mb-4">
            Your Analytics
          </h2>
          <CustomBarChart />
        </div>
      </section>

      {/* AI Predictions Section */}
      <section className="mt-10 px-6 sm:px-10 mb-20 relative z-10">
        <div className="rounded-2xl backdrop-blur-full border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] p-6 sm:p-10">
          <h2 className="text-xl sm:text-2xl font-semibold text-white mb-4">
            AI-Powered Predictions
          </h2>

          <p className="text-white text-base sm:text-lg mb-4">
            Here’s what our AI thinks about your blog performance:
          </p>

          <div className="bg-black/40 border border-gray-700 backdrop-blur-md rounded-xl p-4 text-gray-300 text-sm sm:text-base">
            <p className="mb-2">📈 Your content is gaining steady engagement.</p>
            <p className="mb-2">
              🧠 Our AI predicts a 15% increase in weekly readers if you
              maintain current posting frequency.
            </p>
            <p className="mb-2">
              🔥 Suggested action: Add more trending topics in your tags to
              boost discoverability.
            </p>
          </div>

          <button className="mt-6 w-full sm:w-auto px-6 py-2 cursor-pointer text-[#7C7C6F] text-base sm:text-lg font-bold hover:text-white transition duration-300 border-2 border-amber-300 rounded-full">
            Generate New Prediction
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto w-full bg-transparent border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-full opacity-100 text-white py-8 relative z-10">
        <div className="max-w-6xl mx-auto px-6 sm:px-10">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0 text-center md:text-left">
              <h2 className="text-xl sm:text-2xl font-bold">blogchey.</h2>
              <p className="text-gray-400 mt-1 text-sm sm:text-base">
                The future of intelligent reading
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-4">
              <a
                href="#"
                className="text-gray-400 hover:text-white transition text-sm sm:text-base"
              >
                Terms
              </a>
              <a
                href="#"
                className="text-gray-400 hover:text-white transition text-sm sm:text-base"
              >
                Privacy
              </a>
              <a
                href="#"
                className="text-gray-400 hover:text-white transition text-sm sm:text-base"
              >
                Contact
              </a>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-6 pt-6 text-center text-gray-400 text-xs sm:text-sm">
            <p>© {new Date().getFullYear()} blogchey. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  );
};

export default ProfilePage;
