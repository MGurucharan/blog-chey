import express from "express";
import mongoose from "mongoose";
import Item from "./models/Item.js";
import bodyParser from "body-parser";
import { Collection } from "mongodb";
import multer from "multer";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";
import fs from "fs";
import sharp from "sharp"; // Add this import at the top
import { useParams } from "react-router-dom";

const app = express();
const PORT = process.env.PORT || 5000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(
  cors({
    origin: "http://localhost:5173",
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

//Server will be running on localhost:5000 port
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
app.use(express.json({ limit: "50mb" })); // or more depending on expected size
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use("/uploads", express.static("uploads"));

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/"); // This is your folder to save images
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + "-" + file.originalname); // Avoid file name collisions
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
});

//connects server to MongoDB
mongoose.connect("mongodb://localhost/dummy", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

app.post("/api/senddocument", upload.single("image"), async (req, res) => {
  try {
    let imageUrl = null;
    if (req.file) {
      imageUrl = `${req.protocol}://${req.get("host")}/uploads/${
        req.file.filename
      }`;
    }
    // Get form data from request body
    const title = req.body.title;
    const content = req.body.content;
    const summary=req.body.summary;
    await Item.insertMany([{ title, content, imageUrl , summary }]);
    res.status(200).json({ message: "Document sent", imageUrl });
  } catch (error) {
    console.log(error.message);
    res.status(500).json({ message: error.message });
  }
});

app.post("/api/upload-image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      throw new Error("No file uploaded");
    }

    const filePath = path.join(__dirname, "uploads", req.file.filename);

    try {
      // Process image with sharp (handles AVIF/WEBP)
      const processedImage = await sharp(filePath)
        .toFormat("jpeg") // Convert to JPEG for maximum compatibility
        .jpeg({ quality: 80 }) // Set quality (0-100)
        .toBuffer();

      const base64Image = processedImage.toString("base64");
      const dataURI = `data:image/jpeg;base64,${base64Image}`;

      // Clean up the temporary file
      fs.unlinkSync(filePath);

      res.status(200).json({
        data: {
          files: [dataURI],
          base64: dataURI,
          success: true,
        },
      });
    } catch (processingError) {
      console.error("Image processing failed:", processingError);
      // Fallback to original file if processing fails
      const fileData = fs.readFileSync(filePath);
      const base64Image = fileData.toString("base64");
      const dataURI = `data:${req.file.mimetype};base64,${base64Image}`;

      fs.unlinkSync(filePath);

      res.status(200).json({
        data: {
          files: [dataURI],
          base64: dataURI,
          success: true,
        },
      });
    }
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({
      data: {
        messages: ["Failed to process image. Please try a different format."],
        files: [],
        success: false,
      },
    });
  }
});

//gets all the items from the collection
app.get("/api/items", async (req, res) => {
  try {
    const items = await Item.find({}, "title content summary"); // gets all the data from the database
    res.json(items);
  } catch (error) {
    console.error(error);
    res.status(500).send("Server Error");
  }
});

app.get("/api/items/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const item = await Item.findById(id);
    if (!item) {
      return res.status(404).json({ message: "Item not found " });
    }
    res.json(item);
  } catch (error) {
    res.status(500).json({ message: error.message });
    console.log("Error!");
  }
});

app.put("/api/update/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const title = req.body.title;
    const content = req.body.content;
    const summary= req.body.summary;
    const result = await Item.updateOne(
      { _id: id },
      { $set: { title, content, summary} }
    );
    // if (result.modifiedCount == 0) {
    //   return res
    //     .status(404)
    //     .json({ message: "Blog not found or no changes made!!" });
    //     console.log("No changes made!!!");
    // }
    res.status(200).json({ message: "Successfully updated the Blog!" });
  } catch (error) {
    console.log({ Error: error.message });
  }
});

app.delete("/api/delete/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const result = await Item.deleteOne({ _id: id });
    if (result.deletedCount == 0) {
      return res.status(404).json({ message: "Blog not found." });
    }
    res.status(200).json({ message: "Successfully deleted the blog!" });
  } catch (error) {
    console.log({ Error: error.message });
    res
      .status(500)
      .json({ error: "Something went wrong while deleting the blog." });
  }
});

app.post("/api/feedbackpost/:id",async (req,res)=>{
    try {
        const {id}=req.params
        const feedback=req.body.updatePOCfeedbacks
        const result = await Item.updateOne({_id:id},{$push:{feedback:{$each:feedback}}})
        res.status(200).json({message:"Feedback added",result})
    } catch (error) {
        console.error(error)
        res.status(500).json({error:"Server error while adding feedback!"})
    }
})