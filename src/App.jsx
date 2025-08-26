import Create from "./pages/Create";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import SendDocument from "./pages/SendDocument";
import UpdatePage from "./pages/UpdatePage";
import ProfilePage from "./pages/ProfilePage";
import React, { useState, useEffect } from "react";
import BlogPage from "./pages/BlogPage";
import OpenAiTest from "./components/OpenAiTest"
import Home from "./pages/Home";
import { BrowserRouter } from "react-router-dom";

const App = () => {
//   const [items, setItems] = useState([]);

// //   useEffect(() => {
// //     axios
// //       .get("/api/items")
// //       .then((response) => setItems(response.data))
// //       .catch((error) => console.error("Failed to fetch"))
// //   }, []);

//   console.log(items);

  return (
    <BrowserRouter basename="/blog-chey">
      <Navbar/>
        <Routes>
            <Route path="/" element={<Home/>}/>
            <Route path="/create" element={<Create/>}/>
            <Route path="/senddocument" element={<SendDocument/>}/>
            <Route path="/update/:id" element={<UpdatePage/>}/>
            <Route path="/profile" element={<ProfilePage/>}/>
            <Route path="/blogpage/:id" element={<BlogPage/>}/>
        </Routes>
    </BrowserRouter>
  );
};

export default App;
