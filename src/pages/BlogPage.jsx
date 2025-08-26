import { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import HalfRating from "../components/HalfRating";
import RenderFeedbacks from "../components/RenderFeedbacks";
import Plasma from "../components/Plasma";

const BlogPage = () => {
  const [POCfeedbacks, setPOCfeedbacks] = useState([
    { name: "Kiran Kumar", desp: "Brilliant writing man! keep it up" },
    { name: "Karan RA", desp: "Loved the work!" },
  ]);
  const [blog, setBlog] = useState(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [feedback, setFeedback] = useState("");
  const { id } = useParams();

  // Chatbot
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;
    const userQuery = chatInput;
    setChatMessages((prev) => [...prev, { sender: "user", text: userQuery }]);
    setChatInput("");
    try {
      const response = await axios.post("/fastapi/post", {
        title: title,
        question: userQuery,
      });
      setChatMessages((prev) => [
        ...prev,
        { sender: "llm", text: response.data.message },
      ]);
    } catch (error) {
      console.log({ message: error });
    }
  };

  const handleQuickAction = async (type) => {
    let command = "";
    if (type === "summarize") command = `Summarize the blog of ${title}`;
    else if (type === "translate") command = `Translate the blog of ${title}`;
    setChatMessages((prev) => [...prev, { sender: "user", text: command }]);
    try {
      const response = await axios.post("/fastapi/post", {
        title: title,
        question: command,
      });
      setChatMessages((prev) => [
        ...prev,
        { sender: "llm", text: response.data.message },
      ]);
    } catch (error) {
      console.log({ message: error });
    }
  };

  const handleSubmitFeedback = async (e) => {
    if (!feedback.trim()) {
      alert("Empty feedback!");
      return;
    }
    const updatePOCfeedbacks = [
      ...POCfeedbacks,
      { name: "Gurucharan", desp: feedback },
    ];
    setPOCfeedbacks(updatePOCfeedbacks);
    e.preventDefault();
    try {
      await axios.post(`/api/feedbackpost/${id}`, { updatePOCfeedbacks });
    } catch (error) {
      console.log("Error : ", error);
    }
  };

  useEffect(() => {
    const fetchBlog = async () => {
      try {
        const response = await axios.get(`/api/items/${id}`);
        const blog = response.data;
        setBlog(blog);
        setTitle(blog.title);
        setContent(blog.content);
      } catch (error) {
        console.log({ Error: error.message });
      }
    };
    fetchBlog();
  }, []);

  return (
    <>
      <main className="relative overflow-x-hidden text-white px-6 py-12 pt-32">
        {/* Plasma prism background */}
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

        <section className="ml-10">
          <div
            className="flex flex-col md:flex-row gap-8 rounded-2xl p-8 
          backdrop-blur-full bg-white/5 border border-white/10 
          shadow-[0_20px_80px_rgba(0,0,0,0.35)]"
          >
            {/* Left Side - Blog Content */}
            <div className="flex-1 min-w-0">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 text-center md:text-left leading-tight break-words">
                {title}
              </h1>

              <div
                className="h-1 bg-amber-300 mb-10 rounded-full mx-auto md:mx-0 inline-block"
                style={{ width: `${Math.min(title.length * 12, 300)}px` }}
              ></div>

              <article className="prose prose-invert max-w-none text-lg leading-relaxed overflow-hidden">
                <div dangerouslySetInnerHTML={{ __html: content }} />
              </article>

              {/* Chatbot */}
              <div
                className="mt-8 p-4 rounded-xl 
              backdrop-blur-full bg-white/5 border border-white/10 flex flex-col h-[400px]"
              >
                <h2 className="text-xl font-bold mb-4">AI Assistant</h2>

                <div
                  className="flex-1 overflow-y-auto space-y-3 p-2 
                bg-black/20 rounded-lg border border-white/10"
                >
                  {chatMessages.map((msg, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded-lg max-w-[80%] ${
                        msg.sender === "user"
                          ? "bg-white font-semibold text-black ml-auto"
                          : "bg-white/10 font-semibold text-white"
                      }`}
                    >
                      {msg.text}
                    </div>
                  ))}
                </div>

                <form className="mt-4 flex items-center gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Type your command here..."
                    className="flex-1 font-semibold bg-transparent border-2 border-white rounded-full px-4 py-2 text-white focus:outline-none"
                  />

                  <button
                    type="button"
                    className="border-2 border-white rounded-full px-4 py-2 font-bold text-white hover:bg-white hover:text-black transition duration-300"
                    onClick={() => handleQuickAction("summarize")}
                  >
                    Summarize
                  </button>
                  <button
                    type="button"
                    className="border-2 border-white rounded-full px-4 py-2 font-bold text-white hover:bg-white hover:text-black transition duration-300"
                    onClick={() => handleQuickAction("translate")}
                  >
                    Translate
                  </button>
                  <button
                    type="button"
                    className="border-2 border-amber-300 rounded-full px-4 py-2 font-bold text-amber-300 hover:bg-amber-300 hover:text-black transition duration-300"
                    onClick={handleSendMessage}
                  >
                    Send
                  </button>
                </form>
              </div>
            </div>

            {/* Right Side */}
            <div className="w-full md:w-[350px] flex-shrink-0 border-t md:border-t-0 md:border-l border-white/10 pt-6 md:pt-0 md:pl-6">
              <div className="flex justify-center mb-6">
                <div className="flex items-center gap-2 px-4 py-2 border border-white/10 rounded-full font-bold backdrop-blur-full bg-white/5">
                  <span>Please rate</span>
                  <HalfRating />
                </div>
              </div>

              <div className="p-4 rounded-xl backdrop-blur-full bg-white/5 border border-amber-300">
                <textarea
                  onChange={(e) => setFeedback(e.target.value)}
                  value={feedback}
                  placeholder="Leave your feedback or thoughts..."
                  className="w-full bg-transparent border-2 border-white rounded-lg p-3 text-white focus:outline-none"
                  rows="4"
                ></textarea>

                <button
                  type="button"
                  className="mt-3 border-2 border-white rounded-full font-bold text-white px-6 py-2 hover:bg-white hover:text-black transition duration-300"
                  onClick={handleSubmitFeedback}
                >
                  Submit Feedback
                </button>
              </div>

              <div className="mt-5">
                <RenderFeedbacks value={POCfeedbacks} />
              </div>
            </div>
          </div>
        </section>
      </main>

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
    </>
  );
};

export default BlogPage;
