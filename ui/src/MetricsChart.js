import React from 'react';
import {LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, ResponsiveContainer} from 'recharts';

const comparison_data = [
  { name: 'PSNR (dB)',   Pix2Pix: 27.89, Pix2PixHD: 29.34, DeepHadad: 30.45 },
  { name: 'SSIM',        Pix2Pix: 0.845, Pix2PixHD: 0.852, DeepHadad: 0.882 },
  { name: 'L1 Distance', Pix2Pix: 0.021, Pix2PixHD: 0.020, DeepHadad: 0.017 },
];

const metrics_data = [
  { epoch: 0,   psnr: 18.6, ssim: 0.55, l1: 0.1000 },
  { epoch: 5,   psnr: 20.1, ssim: 0.57, l1: 0.0900 },
  { epoch: 10,  psnr: 21.3, ssim: 0.59, l1: 0.0820 },
  { epoch: 15,  psnr: 22.2, ssim: 0.61, l1: 0.0750 },
  { epoch: 20,  psnr: 23.0, ssim: 0.63, l1: 0.0690 },
  { epoch: 25,  psnr: 23.7, ssim: 0.65, l1: 0.0640 },
  { epoch: 30,  psnr: 24.3, ssim: 0.67, l1: 0.0600 },
  { epoch: 35,  psnr: 24.8, ssim: 0.69, l1: 0.0560 },
  { epoch: 40,  psnr: 25.3, ssim: 0.71, l1: 0.0530 },
  { epoch: 45,  psnr: 25.7, ssim: 0.72, l1: 0.0500 },
  { epoch: 50,  psnr: 26.1, ssim: 0.73, l1: 0.0475 },
  { epoch: 55,  psnr: 26.4, ssim: 0.74, l1: 0.0450 },
  { epoch: 60,  psnr: 26.7, ssim: 0.75, l1: 0.0430 },
  { epoch: 65,  psnr: 27.0, ssim: 0.76, l1: 0.0410 },
  { epoch: 70,  psnr: 27.2, ssim: 0.77, l1: 0.0390 },
  { epoch: 75,  psnr: 27.4, ssim: 0.78, l1: 0.0375 },
  { epoch: 80,  psnr: 27.6, ssim: 0.79, l1: 0.0360 },
  { epoch: 85,  psnr: 27.8, ssim: 0.80, l1: 0.0345 },
  { epoch: 90,  psnr: 28.0, ssim: 0.81, l1: 0.0330 },
  { epoch: 95,  psnr: 28.2, ssim: 0.82, l1: 0.0315 },
  { epoch: 100, psnr: 28.4, ssim: 0.83, l1: 0.0300 },
  { epoch: 105, psnr: 28.5, ssim: 0.83, l1: 0.0290 },
  { epoch: 110, psnr: 28.6, ssim: 0.84, l1: 0.0280 },
  { epoch: 115, psnr: 28.7, ssim: 0.84, l1: 0.0270 },
  { epoch: 120, psnr: 28.8, ssim: 0.84, l1: 0.0260 },
  { epoch: 125, psnr: 28.9, ssim: 0.85, l1: 0.0250 },
  { epoch: 130, psnr: 29.0, ssim: 0.85, l1: 0.0240 },
  { epoch: 135, psnr: 29.1, ssim: 0.85, l1: 0.0230 },
  { epoch: 140, psnr: 29.2, ssim: 0.85, l1: 0.0220 },
  { epoch: 145, psnr: 29.3, ssim: 0.86, l1: 0.0210 },
  { epoch: 150, psnr: 29.4, ssim: 0.86, l1: 0.0200 },
  { epoch: 155, psnr: 29.5, ssim: 0.86, l1: 0.0195 },
  { epoch: 160, psnr: 29.6, ssim: 0.86, l1: 0.0190 },
  { epoch: 165, psnr: 29.7, ssim: 0.86, l1: 0.0185 },
  { epoch: 170, psnr: 29.8, ssim: 0.86, l1: 0.0180 },
  { epoch: 175, psnr: 29.9, ssim: 0.86, l1: 0.0175 },
  { epoch: 180, psnr: 30.0, ssim: 0.87, l1: 0.0173 },
  { epoch: 185, psnr: 30.1, ssim: 0.87, l1: 0.0171 },
  { epoch: 190, psnr: 30.2, ssim: 0.87, l1: 0.0169 },
  { epoch: 195, psnr: 30.3, ssim: 0.87, l1: 0.0168 },
  { epoch: 200, psnr: 30.4, ssim: 0.87, l1: 0.0167 },
];

const loss_data = [
  { epoch: 0,   generator: 2.500, discriminator: 0.760 },
  { epoch: 5,   generator: 2.300, discriminator: 0.660 },
  { epoch: 10,  generator: 2.150, discriminator: 0.610 },
  { epoch: 15,  generator: 2.000, discriminator: 0.580 },
  { epoch: 20,  generator: 1.880, discriminator: 0.550 },
  { epoch: 25,  generator: 1.760, discriminator: 0.525 },
  { epoch: 30,  generator: 1.650, discriminator: 0.500 },
  { epoch: 35,  generator: 1.550, discriminator: 0.480 },
  { epoch: 40,  generator: 1.460, discriminator: 0.460 },
  { epoch: 45,  generator: 1.380, discriminator: 0.445 },
  { epoch: 50,  generator: 1.310, discriminator: 0.430 },
  { epoch: 55,  generator: 1.250, discriminator: 0.415 },
  { epoch: 60,  generator: 1.190, discriminator: 0.400 },
  { epoch: 65,  generator: 1.140, discriminator: 0.390 },
  { epoch: 70,  generator: 1.090, discriminator: 0.380 },
  { epoch: 75,  generator: 1.050, discriminator: 0.370 },
  { epoch: 80,  generator: 1.010, discriminator: 0.360 },
  { epoch: 85,  generator: 0.975, discriminator: 0.350 },
  { epoch: 90,  generator: 0.940, discriminator: 0.340 },
  { epoch: 95,  generator: 0.910, discriminator: 0.335 },
  { epoch: 100, generator: 0.880, discriminator: 0.330 },
  { epoch: 105, generator: 0.855, discriminator: 0.325 },
  { epoch: 110, generator: 0.830, discriminator: 0.320 },
  { epoch: 115, generator: 0.810, discriminator: 0.315 },
  { epoch: 120, generator: 0.790, discriminator: 0.310 },
  { epoch: 125, generator: 0.775, discriminator: 0.305 },
  { epoch: 130, generator: 0.760, discriminator: 0.300 },
  { epoch: 135, generator: 0.745, discriminator: 0.295 },
  { epoch: 140, generator: 0.730, discriminator: 0.290 },
  { epoch: 145, generator: 0.720, discriminator: 0.285 },
  { epoch: 150, generator: 0.710, discriminator: 0.280 },
  { epoch: 155, generator: 0.700, discriminator: 0.275 },
  { epoch: 160, generator: 0.690, discriminator: 0.272 },
  { epoch: 165, generator: 0.685, discriminator: 0.270 },
  { epoch: 170, generator: 0.680, discriminator: 0.270 },
  { epoch: 175, generator: 0.675, discriminator: 0.270 },
  { epoch: 180, generator: 0.670, discriminator: 0.265 },
  { epoch: 185, generator: 0.665, discriminator: 0.264 },
  { epoch: 190, generator: 0.660, discriminator: 0.262 },
  { epoch: 195, generator: 0.655, discriminator: 0.260 },
  { epoch: 200, generator: 0.650, discriminator: 0.260 },
];

const normalized_metrics_data = metrics_data.map(item => ({
  ...item,
  normalized_psnr: item.psnr / 32,
  normalized_ssim: item.ssim / 1,
  normalized_l1: item.l1 / 0.1,
}));


const NetworksComparisonChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <BarChart data={comparison_data} margin={{top: 20, right: 30, left: 20, bottom: 5}}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
      <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
      <Tooltip />
      <Legend />
      <Bar yAxisId="left" dataKey="Pix2Pix" fill="#b77474" />
      <Bar yAxisId="left" dataKey="Pix2PixHD" fill="#82ca9d" />
      <Bar yAxisId="left" dataKey="DeepHadad" fill="#9975e5" />
    </BarChart>
  </ResponsiveContainer>
);


const MetricsPerEpochChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={metrics_data} margin={{top: 5, right: 30, left: 20, bottom: 5}}>
      <CartesianGrid strokeDasharray="3 3" stroke="#000000" />
      <XAxis dataKey="epoch" />
      <YAxis yAxisId="psnr" label={{ value: 'PSNR', angle: -90, position: 'insideLeft' }} />
      <YAxis yAxisId="ssim" orientation="right" label={{ value: 'SSIM', angle: 90, position: 'insideRight' }} />
      <YAxis yAxisId="l1" orientation="right" label={{ value: 'L1 Distance', angle: 90, position: 'insideRight' }} />
      <Tooltip />
      <Legend />
      <Line yAxisId="psnr" type="monotone" dataKey="psnr" stroke="#9975e5" name="PSNR" strokeWidth={3} />
      <Line yAxisId="ssim" type="monotone" dataKey="ssim" stroke="#82ca9d" name="SSIM" strokeWidth={3} />
      <Line yAxisId="l1" type="monotone" dataKey="l1" stroke="#b77474" name="L1 Distance" strokeWidth={3} />
    </LineChart>
  </ResponsiveContainer>
);


// const chartStyle = {
//   fontSize: '14px',
//   fontFamily: 'Arial, sans-serif',
// };


const NormalizedMetricsChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={normalized_metrics_data} margin={{top: 5, right: 30, left: 20, bottom: 5}}>
      <CartesianGrid strokeDasharray="3 3" stroke="#000000" />
      <XAxis dataKey="epoch" />
      <YAxis domain={[0, 1]} />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="normalized_psnr" stroke="#9975e5" name="PSNR / 32" strokeWidth={3} />
      <Line type="monotone" dataKey="normalized_ssim" stroke="#82ca9d" name="SSIM / 1" strokeWidth={3} />
      <Line type="monotone" dataKey="normalized_l1" stroke="#b77474" name="L1 / 0.1" strokeWidth={3} />
    </LineChart>
  </ResponsiveContainer>
);

// const NormalizedMetricsChart = () => (
//   <ResponsiveContainer width="100%" height={400}>
//     <LineChart data={normalized_metrics_data} margin={{top: 5, right: 30, left: 20, bottom: 5}}>
//       <CartesianGrid strokeDasharray="3 3" stroke="#000000" />
//       <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }} />
//       <YAxis domain={[0, 1]} label={{ value: 'Normalized Value', angle: -90, position: 'insideLeft', offset: 10, strokeWidth:3}}/>
//       <Tooltip />
//       <Legend />
//       <Line type="monotone" dataKey="normalized_psnr" stroke="#9975e5" name="PSNR / 32" strokeWidth={3} />
//       <Line type="monotone" dataKey="normalized_ssim" stroke="#82ca9d" name="SSIM / 1" strokeWidth={3} />
//       <Line type="monotone" dataKey="normalized_l1" stroke="#b77474" name="L1 / 0.1" strokeWidth={3} />
//     </LineChart>
//   </ResponsiveContainer>
// );

const LossFunctionsChart = () => (
  <ResponsiveContainer width="100%" height={400}>
    <LineChart data={loss_data} margin={{top: 5, right: 30, left: 20, bottom: 5}}>
      <CartesianGrid strokeDasharray="3 3" stroke="#000000" />
      <XAxis dataKey="epoch" />
      <YAxis domain={[0, 3]} />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="generator" stroke="#9975e5" name="Generator Loss" strokeWidth={3} />
      <Line type="monotone" dataKey="discriminator" stroke="#b77474" name="Discriminator Loss" strokeWidth={3} />
    </LineChart>
  </ResponsiveContainer>
);

// const NormalizedMetricsChart = () => (
//   <div style={chartStyle}>
//     <ResponsiveContainer width="100%" height={450}>
//       <LineChart data={normalized_metrics_data} margin={{top: 20, right: 30, left: 10, bottom: 10}}>
//         <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
//         <XAxis
//           dataKey="epoch"
//           label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }}
//           tick={{fontSize: 12}}
//         />
//         <YAxis
//           domain={[0, 1]}
//           label={{ value: 'Normalized Value', angle: -90, position: 'insideLeft', offset: 0 }}
//           tick={{fontSize: 12}}
//         />
//         <Tooltip />
//         <Legend verticalAlign="top" height={36} />
//         <Line type="monotone" dataKey="normalized_psnr" stroke="#8884d8" name="PSNR" strokeWidth={2} dot={false} />
//         <Line type="monotone" dataKey="normalized_ssim" stroke="#82ca9d" name="SSIM" strokeWidth={2} dot={false} />
//         <Line type="monotone" dataKey="normalized_l1" stroke="#ffc658" name="L1 Distance" strokeWidth={2} dot={false} />
//       </LineChart>
//     </ResponsiveContainer>
//   </div>
// );
//
// const LossFunctionsChart = () => (
//   <div style={chartStyle}>
//     <ResponsiveContainer width="100%" height={450}>
//       <LineChart data={loss_data} margin={{top: 20, right: 30, left: 10, bottom: 10}}>
//         <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
//         <XAxis
//           dataKey="epoch"
//           label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }}
//           tick={{fontSize: 12}}
//         />
//         <YAxis
//           label={{ value: 'Loss Value', angle: -90, position: 'insideLeft', offset: 0 }}
//           tick={{fontSize: 12}}
//         />
//         <Tooltip />
//         <Legend verticalAlign="top" height={36} />
//         <Line type="monotone" dataKey="generator" stroke="#8884d8" name="Generator Loss" strokeWidth={2} dot={false} />
//         <Line type="monotone" dataKey="discriminator" stroke="#82ca9d" name="Discriminator Loss" strokeWidth={2} dot={false} />
//       </LineChart>
//     </ResponsiveContainer>
//   </div>
// );

export {
  NetworksComparisonChart,
  MetricsPerEpochChart,
  LossFunctionsChart,
  NormalizedMetricsChart,
};