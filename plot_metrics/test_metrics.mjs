
import React from 'react';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const data = [
  { epoch: 0, psnr: 20, ssim: 0.5, l1: 0.1 },
  { epoch: 50, psnr: 25, ssim: 0.7, l1: 0.05 },
  { epoch: 100, psnr: 28, ssim: 0.8, l1: 0.03 },
  { epoch: 150, psnr: 29.5, ssim: 0.85, l1: 0.02 },
  { epoch: 200, psnr: 30.45, ssim: 0.882, l1: 0.017 },
];

const MetricsChart = () => (
  <LineChart width={600} height={300} data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="epoch" />
    <YAxis yAxisId="left" />
    <YAxis yAxisId="right" orientation="right" />
    <Tooltip />
    <Legend />
    <Line yAxisId="left" type="monotone" dataKey="psnr" stroke="#8884d8" name="PSNR" />
    <Line yAxisId="right" type="monotone" dataKey="ssim" stroke="#82ca9d" name="SSIM" />
    <Line yAxisId="right" type="monotone" dataKey="l1" stroke="#ffc658" name="L1 Distance" />
  </LineChart>
);

export default MetricsChart;
