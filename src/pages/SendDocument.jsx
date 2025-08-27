import AnimatedContent from "../components/AnimatedContent";
import SpotlightCard from "../components/SpotlightCard";
import { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Plasma from "../components/Plasma";
import arrow from "../assets/arrow.png"

const SendDocument = () => {
  const [blogs, setBlogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleDelete = async (blog_id) => {
    try {
      const option = prompt("Are you sure ?");
      if (option && option.toLowerCase() === "yes") {
        await axios.delete(`/api/delete/${blog_id}`);
        alert("Successfully deleted !");
        setBlogs((prevBlogs) =>
          prevBlogs.filter((blog) => blog._id !== blog_id)
        );
      }
    } catch (error) {
      console.log({ Error: error.message });
    }
  };

  useEffect(() => {
    const fetchBlogs = async () => {
      try {
        const response = await axios.get("/api/items");
        setBlogs(response.data);
      } catch (error) {
        setError(error.message || "Something went wrong");
      } finally {
        setLoading(false);
      }
    };

    fetchBlogs();
  }, []);

  if (loading) {
    return <p className="text-white text-center py-10">Loading...</p>;
  }

  if (error) {
    return <p className="text-red-500 text-center py-10">{error}</p>;
  }

  return (
    <main className="relative min-h-screen overflow-x-hidden text-white">
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

      <section className="w-full px-4 sm:px-6 lg:px-8 py-20 relative z-10">
        <div className="max-w-7xl mx-auto">
          <AnimatedContent
            distance={20}
            duration={0.8}
            threshold={0.1}
            delay={0.1}
          >
            <h1 className="mt-10 text-4xl sm:text-5xl md:text-6xl text-center mb-12 font-bold">
              {blogs.length === 0 ? "No blogs yet :(" : "Published blogs."}
            </h1>
          </AnimatedContent>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {blogs.length === 0
              ? null
              : blogs.map((blog) => (
                  <AnimatedContent
                    key={blog._id}
                    distance={30}
                    direction="vertical"
                    duration={0.6}
                    delay={0.2}
                    threshold={0.1}
                  >
                    <SpotlightCard
                      className="h-full min-h-[320px] sm:min-h-[360px] md:min-h-[400px] backdrop-blur-full border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)] rounded-2xl"
                      spotlightColor="rgba(255, 255, 255, 0.35)"
                    >
                      <div className="h-full flex flex-col p-6">
                        {/* Header */}
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 mr-3"></div>
                            <div>
                              <p className="font-medium text-sm sm:text-base">
                                You
                              </p>
                              <p className="text-xs text-neutral-400">
                                Just now
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => navigate(`/update/${blog._id}`)}
                              className="text-amber-300 text-xs sm:text-sm hover:text-white transition duration-300 border-2 cursor-pointer border-amber-300 rounded-full px-3 py-1"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => handleDelete(blog._id)}
                              className="text-amber-300 text-xs sm:text-sm hover:text-red-500 transition duration-300 border-2 cursor-pointer border-amber-300 rounded-full px-3 py-1"
                            >
                              Delete
                            </button>
                            <img
                              src={arrow}
                              alt="open"
                              className="w-[20px] cursor-pointer"
                              onClick={() => {
                                navigate(`/blogpage/${blog._id}`);
                              }}
                            />
                          </div>
                        </div>

                        {/* Blog Content */}
                        <h3 className="text-xl sm:text-2xl font-bold mb-3">
                          {blog.title}
                        </h3>
                        <p className="text-lg text-neutral-300 mb-4">
                          {blog.summary}
                        </p>

                        {/* Tags */}
                        <div className="mt-auto flex flex-wrap gap-2">
                          <span className="text-xs px-2 py-1 bg-black/40 border border-white/10 backdrop-blur-md rounded-full">
                            #
                          </span>
                          <span className="text-xs px-2 py-1 bg-black/40 border border-white/10 backdrop-blur-md rounded-full">
                            #ai
                          </span>
                          <span className="text-xs px-2 py-1 bg-black/40 border border-white/10 backdrop-blur-md rounded-full">
                            #trends
                          </span>
                        </div>
                      </div>
                    </SpotlightCard>
                  </AnimatedContent>
                ))}
          </div>
        </div>
      </section>
    </main>
  );
};

export default SendDocument;
