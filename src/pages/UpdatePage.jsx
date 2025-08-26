import RotatingText from "../components/RotatingText";
import JoditEditor from "jodit-react";
import { useRef, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Plasma from "../components/Plasma";
import axios from "axios";

const UpdatePage = () => {
  const editor = useRef(null);
  const navigate = useNavigate();
  const { id } = useParams();

  const [isSummarizing, setIsSummarizing] = useState(false);
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("");
  const [summary, setSummary] = useState("");

  useEffect(() => {
    const savedTitle = localStorage.getItem("Title");
    if (savedTitle) setTitle(savedTitle);

    const savedContent = localStorage.getItem("Content");
    if (savedContent) setContent(savedContent);

    const fetchBlog = async () => {
      try {
        const response = await axios.get(`/api/items/${id}`);
        const blog = response.data;
        if (!savedTitle) setTitle(blog.title);
        if (!savedContent) setContent(blog.content);
      } catch (error) {
        console.error("Error fetching blog:", error);
      }
    };

    fetchBlog();

    return () => {
      localStorage.removeItem("Title");
      localStorage.removeItem("Content");
    };
  }, [id]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSummarizing(true);

    const cleanContent = content.replace(
      /<img[^>]+src="data:image\/[^;]+;base64,[^"]+"[^>]*>/g,
      (m) => m
    );

    try {
      const strippedContent = content.replace(/<img[^>]*>/g, "[image]");
      const summaryRes = await axios.post("/openai-api/openai", {
        prompt: `Below is an HTML blog post. Ignore the [image] placeholders and summarize in 25 words only.\n\n${strippedContent}`,
      });
      const summaryText = summaryRes.data.choices[0].message.content;
      setSummary(summaryText);

      await axios.put(
        `/api/update/${id}`,
        { title, content: cleanContent, summary: summaryText },
        { headers: { "Content-Type": "application/json" } }
      );

      navigate("/senddocument");
    } catch (error) {
      console.log("Error:", error);
    } finally {
      setIsSummarizing(false);
      localStorage.removeItem("Title");
      localStorage.removeItem("Content");
    }
  };

  useEffect(() => {
    if (editor.current) editor.current.focus();
  }, []);

  const config = {
    height: 400,
    readonly: false,
    askBeforePasteFromHTML: true,
    defaultActionOnPaste: "insert_as_plaintext",
    uploader: {
      insertImageAsBase64URI: true,
      url: "http://localhost:5000/api/upload-image",
      format: "json",
      method: "POST",
      filesVariableName: () => "image",
      isSuccess: (resp) => resp && resp.data && resp.data.success,
      process: (resp) => ({ files: resp.data.files, path: "", baseurl: "" }),
      error: (e) => ({
        success: false,
        messages: [e.message || "Image upload failed"],
      }),
      headers: { Accept: "application/json" },
      prepareData: (formData) => formData,
    },
    image: {
      maxWidth: 1200,
      quality: 0.8,
      openOnDblClick: true,
      popup: true,
      showPreview: true,
      useImageEditor: true,
      accept: ".jpg,.jpeg,.png,.gif,.webp,.avif",
      editSrc: true,
    },
    style: { background: "transparent" },
    allowDragAndDropFileToEditor: true,
    draggableTags: ["img"],
    buttons: [
      "source",
      "|",
      "bold",
      "italic",
      "underline",
      "strikethrough",
      "|",
      "superscript",
      "subscript",
      "|",
      "ul",
      "ol",
      "outdent",
      "indent",
      "|",
      "font",
      "fontsize",
      "brush",
      "paragraph",
      "|",
      "image",
      "video",
      "file",
      "table",
      "link",
      "|",
      "align",
      "undo",
      "redo",
      "|",
      "hr",
      "copyformat",
      "symbol",
      "fullsize",
      "selectall",
      "print",
      "preview",
    ],
    sizeLG: 900,
    sizeMD: 700,
    sizeSM: 400,
    maxSize: 50 * 1024 * 1024,
    allowResizeX: true,
    allowResizeY: true,
  };

  return (
    <main className="relative">
      {/* Plasma background behind the whole page */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <Plasma
          color="#FFBF00"
          speed={0.8}
          direction="forward"
          scale={1.5}
          opacity={100}
          mouseInteractive={false}
        />
      </div>

      {/* Small CSS tweaks for Jodit transparency & images */}
      <style>{`
        .jodit-container:not(.jodit_inline),
        .jodit-workplace,
        .jodit-wysiwyg {
          background: transparent !important;
        }
        .jodit-wysiwyg {
          color: #ffffff;
        }
        .jodit-wysiwyg img {
          max-width: 100%;
          height: auto;
          display: block;
          margin: 10px 0;
        }
      `}</style>

      {/* HERO */}
      <section className="flex flex-col items-center justify-center min-h-screen px-4">
        <h1 className="text-4xl sm:text-6xl z-10 md:text-7xl lg:text-8xl xl:text-9xl bg-gradient-to-r from-amber-300 to-white bg-clip-text text-transparent drop-shadow-[0_2px_10px_rgba(0,0,0,0.4)]">
          Re-define your
        </h1>
        <div className="mt-4 sm:mt-6 md:mt-8">
          <RotatingText
            texts={["blog.", "creation."]}
            mainClassName="px-2 sm:px-3 md:px-4 bg-gradient-to-r from-white to-yellow-400 text-black font-bold text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl overflow-hidden py-1 sm:py-1.5 md:py-2 justify-center rounded-lg shadow-[0_8px_40px_rgba(255,191,0,0.25)]"
            staggerFrom={"first"}
            initial={{ y: "105%" }}
            animate={{ y: 0 }}
            exit={{ y: "-120%" }}
            staggerDuration={0.055}
            splitLevelClassName="overflow-hidden pb-0.5 sm:pb-1 md:pb-1"
            transition={{ type: "spring", damping: 30, stiffness: 400 }}
            rotationInterval={3000}
          />
        </div>
      </section>

      {/* UPDATE YOUR BLOG */}
      <section className="relative w-full px-4 sm:px-8 md:px-12 py-12 sm:py-20 md:py-28">
        <div className="max-w-5xl mx-auto rounded-2xl bg-black/1 backdrop-blur-md border border-white/10 shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
          {/* Heading */}
          <div className="px-6 sm:px-8 md:px-12 pt-10 pb-6 text-center">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-7xl bg-gradient-to-r from-white via-amber-200 to-amber-300 bg-clip-text text-transparent font-bold">
              Update your blog!
            </h1>
          </div>

          {/* Form */}
          <div className="px-6 sm:px-8 md:px-12 pb-10">
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value);
                  localStorage.setItem("Title", e.target.value);
                }}
                placeholder="Enter title"
                name="title"
                className="px-4 py-3 bg-black/60 backdrop-blur-sm shadow-inner font-semibold text-white rounded-lg focus:outline-none w-full text-lg sm:text-xl md:text-2xl placeholder-white/70 transition-all duration-300 mb-9 border border-white/10"
              />

              <div className="rounded-lg bg-black/50 backdrop-blur-sm border border-white/10 p-2">
                <JoditEditor
                  ref={editor}
                  value={content}
                  config={config}
                  onBlur={(newContent) => {
                    setContent(newContent);
                    localStorage.setItem("Content", newContent);
                  }}
                />
              </div>

              <input type="hidden" name="content" value={content} />

              <div className="w-full flex justify-center mt-8">
                <button
                  type="submit"
                  disabled={isSummarizing}
                  className="disabled:opacity-70 disabled:cursor-not-allowed cursor-pointer bg-white text-black text-lg sm:text-xl font-semibold rounded-full px-8 py-3 transition-transform duration-300 hover:scale-105 shadow-[0_10px_30px_rgba(255,255,255,0.15)]"
                >
                  {isSummarizing ? "Updating..." : "Re-publish"}
                </button>
              </div>
            </form>
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

export default UpdatePage;
