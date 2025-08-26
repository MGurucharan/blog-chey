import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data = [
  { name: "Views", value: 2400 },
  { name: "Likes", value: 1398 },
  { name: "Comments", value: 980 },
  { name: "Shares", value: 390 },
  { name: "Bookmarks", value: 800 },
];

const CustomBarChart = () => {
  return (
    <div className=" font-bold w-full h-[350px] px-10 mt-44">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" />
          <XAxis dataKey="name" stroke="#ffffff" />
          <YAxis stroke="#ffffff" />
          <Tooltip
            contentStyle={{ backgroundColor: "#111", border: "1px solid #facc15" }}
            labelStyle={{ color: "#facc15" }}
            cursor={{ fill: "rgba(251, 191, 36, 0.1)" }}
          />
          <Bar dataKey="value" fill="#facc15" radius={[10, 10, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CustomBarChart;
