import RotatingText from "./RotatingText";
import SpotlightCard from "./SpotlightCard";
import AnimatedContent from "./AnimatedContent";
import SplitText from "./SplitText";
import { useEffect } from "react";
import axios from "axios";
import Plasma from "./Plasma";

const Intro = () => {
  useEffect(() => {
    const ingester = async () => {
      try {
        const response = await axios.post("/fastapi/ingest");
        console.log("Ingestion started:", response.data);
      } catch (error) {
        console.error("Ingestion error:", error);
      }
    };
    ingester();
  }, []);

  const handleAnimationComplete = () => {
    console.log("All letters have animated!");
  };

  return (
    <main className="relative overflow-x-hidden">
      {/* HERO */}
      <section className="flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8 relative">
        {/* Plasma background */}
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

        {/* Hero content */}
        <div className="w-full max-w-6xl mx-auto text-center relative z-10">
          <div className="flex flex-col items-center sm:flex-row justify-center">
            <span className="text-white font-bold text-7xl sm:text-8xl md:text-8xl lg:text-8xl xl:text-9xl mb-4 sm:mb-0 sm:mr-3 md:mr-5">
              blog
            </span>
            <span className="mt-2 sm:mt-0">
              <RotatingText
                texts={["chey.", "it out."]}
                mainClassName="px-2 sm:px-3 md:px-4 bg-gradient-to-r from-white to-yellow-400 text-black font-bold text-6xl sm:text-7xl md:text-7xl lg:text-7xl xl:text-8xl overflow-hidden py-1 sm:py-1.5 md:py-2 justify-center rounded-lg shadow-[0_8px_40px_rgba(255,191,0,0.25)]"
                staggerFrom={"last"}
                initial={{ y: "105%" }}
                animate={{ y: 0 }}
                exit={{ y: "-120%" }}
                staggerDuration={0.055}
                splitLevelClassName="overflow-hidden pb-0.5 sm:pb-1 md:pb-1"
                transition={{ type: "spring", damping: 30, stiffness: 400 }}
                rotationInterval={2500}
              />
            </span>
          </div>

          <div className="mt-7 flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-5">
            <SplitText
              text="Crafted for readers. Powered by intelligence."
              className="text-[13px] sm:text-2xl md:text-[28px] lg:text-[28px] bg-gradient-to-r from-yellow-400 to-white rounded-lg p-2 font-extrabold text-center"
              delay={50}
              duration={0.5}
              ease="power3.out"
              splitType="chars"
              from={{ opacity: 0, y: 40 }}
              to={{ opacity: 1, y: 0 }}
              threshold={0.1}
              rootMargin="-100px"
              textAlign="center"
              onLetterAnimationComplete={handleAnimationComplete}
            />
          </div>
        </div>
      </section>

      {/* POPULAR POSTS */}
      <section className="relative w-full px-4 sm:px-6 lg:px-8 py-12 sm:py-20 md:py-28">
        <div className="max-w-7xl mx-auto rounded-2xl backdrop-blur-full border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] p-6 sm:p-10">
          <AnimatedContent distance={20} duration={0.8} threshold={0.1}>
            <h1 className="text-white text-4xl sm:text-5xl md:text-6xl text-center mb-8 sm:mb-12">
              Popular posts.
            </h1>
          </AnimatedContent>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {/* Post 1 */}
            <AnimatedContent distance={30} direction="vertical" duration={0.6} delay={0.2}>
              <SpotlightCard
                className="text-white h-full min-h-[320px] sm:min-h-[360px] md:min-h-[400px]"
                spotlightColor="rgba(255, 200, 0, 0.2)"
              >
                <div className="h-full flex flex-col p-5 sm:p-6">
                  <div className="flex items-center mb-4">
                    <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 mr-3"></div>
                    <div>
                      <p className="font-medium text-sm sm:text-base">Sarah Johnson</p>
                      <p className="text-xs text-neutral-400">2 days ago</p>
                    </div>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold mb-3">The Future of Web Development</h3>
                  <p className="text-neutral-300 mb-4 text-sm sm:text-base">
                    Exploring how AI and new frameworks are changing the way we build for the web in 2024.
                  </p>
                  <div className="mt-auto flex flex-wrap gap-2">
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#webdev</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#ai</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#trends</span>
                  </div>
                </div>
              </SpotlightCard>
            </AnimatedContent>

            {/* Post 2 */}
            <AnimatedContent distance={30} direction="vertical" duration={0.6} delay={0.3}>
              <SpotlightCard
                className="text-white h-full min-h-[320px] sm:min-h-[360px] md:min-h-[400px]"
                spotlightColor="rgba(255, 200, 0, 0.2)"
              >
                <div className="h-full flex flex-col p-5 sm:p-6">
                  <div className="flex items-center mb-4">
                    <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-blue-500 to-teal-500 mr-3"></div>
                    <div>
                      <p className="font-medium text-sm sm:text-base">Michael Chen</p>
                      <p className="text-xs text-neutral-400">1 week ago</p>
                    </div>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold mb-3">Minimalism in UI Design</h3>
                  <p className="text-neutral-300 mb-4 text-sm sm:text-base">
                    Why less is more in modern interface design and how to implement it effectively in your projects.
                  </p>
                  <div className="mt-auto flex flex-wrap gap-2">
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#design</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#ui</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#ux</span>
                  </div>
                </div>
              </SpotlightCard>
            </AnimatedContent>

            {/* Post 3 */}
            <AnimatedContent distance={30} direction="vertical" duration={0.6} delay={0.4}>
              <SpotlightCard
                className="text-white h-full min-h-[320px] sm:min-h-[360px] md:min-h-[400px]"
                spotlightColor="rgba(255, 200, 0, 0.2)"
              >
                <div className="h-full flex flex-col p-5 sm:p-6">
                  <div className="flex items-center mb-4">
                    <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-orange-500 to-yellow-500 mr-3"></div>
                    <div>
                      <p className="font-medium text-sm sm:text-base">David Wilson</p>
                      <p className="text-xs text-neutral-400">3 days ago</p>
                    </div>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold mb-3">Building a Personal Brand</h3>
                  <p className="text-neutral-300 mb-4 text-sm sm:text-base">
                    Practical tips for developers and creators to establish their online presence and grow their audience.
                  </p>
                  <div className="mt-auto flex flex-wrap gap-2">
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#career</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#branding</span>
                    <span className="text-xs px-2 py-1 bg-neutral-800 rounded-full">#growth</span>
                  </div>
                </div>
              </SpotlightCard>
            </AnimatedContent>
          </div>
        </div>
      </section>

      {/* ABOUT */}
      <section className="relative w-full px-4 sm:px-6 lg:px-8 py-12 sm:py-20 md:py-28">
        <div className="max-w-6xl mx-auto rounded-2xl backdrop-blur-full border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] p-6 sm:p-10">
          <AnimatedContent distance={20} duration={0.8} threshold={0.1}>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-center mb-12 text-white">
              About blogchey.
            </h1>
          </AnimatedContent>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <AnimatedContent distance={30} direction="vertical" delay={0.2}>
              <div className="space-y-6">
                <h2 className="text-2xl sm:text-3xl font-semibold text-white">
                  The Future of Reading.   Powered by Artifical Intelligence.
                </h2>
                <p className="text-lg text-white">
                  Blogchey is an intelligent platform that revolutionizes your reading experience. Our AI learns your preferences to curate
                  content you'll love, while our natural language processing makes every article more accessible and engaging.
                </p>
                <p className="text-lg text-white">
                  We combine human creativity with machine intelligence to deliver a seamless reading experience that adapts to your unique style.
                </p>
                <div className="flex flex-wrap gap-3 mt-6">
                  <span className="px-4 py-2 bg-gray-900 text-white rounded-full text-sm font-medium">Smart Recommendations</span>
                  <span className="px-4 py-2 bg-gray-900 text-white rounded-full text-sm font-medium">Personalized Content</span>
                  <span className="px-4 py-2 bg-gray-900 text-white rounded-full text-sm font-medium">AI-Assisted Reading</span>
                </div>
              </div>
            </AnimatedContent>

            <AnimatedContent distance={30} direction="vertical" delay={0.3}>
              <div className="space-y-6">
                <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100">
                  <h3 className="text-xl font-bold text-gray-800 mb-3">Why Choose Blogchey?</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <span className="text-yellow-500 mr-2 mt-1">•</span>
                      <span className="text-gray-700">Smart summarization for quick understanding</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-yellow-500 mr-2 mt-1">•</span>
                      <span className="text-gray-700">Adaptive text complexity based on your level</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-yellow-500 mr-2 mt-1">•</span>
                      <span className="text-gray-700">Multilingual support with seamless translation</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-yellow-500 mr-2 mt-1">•</span>
                      <span className="text-gray-700">Interactive content that responds to your interests</span>
                    </li>
                  </ul>
                </div>
                <p className="text-gray-400 italic">
                  "Blogchey has transformed how I consume content online. It's like having a personal reading assistant that knows exactly
                  what I need." — Sarah T., Power User
                </p>
              </div>
            </AnimatedContent>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="w-full bg-transparent border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-full opacity-100 text-white py-8 mt-28">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h2 className="text-2xl font-bold">blogchey.</h2>
              <p className="text-gray-400 mt-1">
                The future of intelligent reading
              </p>
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-400 hover:text-white transition">
                Terms
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition">
                Privacy
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition">
                Contact
              </a>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-6 pt-6 text-center text-gray-400 text-sm">
            <p>© {new Date().getFullYear()} blogchey. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  );
};

export default Intro;
