import { Schema, model } from "mongoose";

const itemSchema = new Schema({
  title: String,
  content: String,
  summary: String,
  feedback: Array
});

export default model("Item", itemSchema);