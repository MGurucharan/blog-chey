import React from "react";
import { useState } from "react";
import axios from "axios";

const openAiTest = () => {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
        const response=await axios.post("/openai-api/openai",{
            prompt:input
        });
        setResponse(response.data.choices[0].message.content);
        console.log("Response",response.data);
    } catch (error) {
        console.log("Error:",error)
    }
  };

  return (
    <div className="text-white"> 
      <h1>OpenAI Integration with React</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows="5"
          cols="50"
          placeholder="Type your prompt here..."
        />
        <br />
        <button type="submit">Get Response</button>
      </form>
      <div>
        <h2>Response:</h2>
        <p>{response}</p>
      </div>
    </div>
  );
};

export default openAiTest;
